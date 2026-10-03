"""
Менеджер задач: создание, хранение, очистка.

Задачи живут в двух местах: активные — в памяти (для быстрого SSE),
завершённые — в SQLite (переживают перезапуск сервера). JobManager
координирует оба источника.

Задача (Job) — единица работы по транскрибации одного файла. Она
хранит состояние, прогресс, логи, сегменты и результат.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .storage import JobRepository


@dataclass
class Job:
    """
    Единица работы по транскрибации одного файла.

    Класс не потокобезопасен сам по себе — мутирующие методы предполагают
    вызов из одного потока (воркера). emit() безопасен к вызову из другого
    потока за счёт call_soon_threadsafe.

    При наличии _repo изменения автоматически сохраняются в БД: логи
    и сегменты сразу, прогресс — с дедупликацией по времени.

    Attributes:
        id: Уникальный 12-символьный идентификатор задачи.
        filename: Исходное имя загруженного файла.
        file_path: Путь к файлу на диске.
        file_size: Размер файла в байтах.
        model: Название модели Whisper.
        language: Код языка или None для автоопределения.
        status: pending, downloading, loading, transcribing, done, error, cancelled, interrupted.
        stage: Стадия для UI (совпадает со status в большинстве случаев).
        progress: Прогресс от 0 до 100.
        message: Человекочитаемое описание текущего действия.
        logs: Накопленные записи логов (в памяти).
        segments: Распознанные сегменты текста (в памяти).
        text: Полный распознанный текст.
        metadata: Итоговые метаданные (язык, длительность, скорость).
        error: Текст ошибки, если задача завершилась неудачно.
        created_at: Unix-время создания задачи.
        finished_at: Unix-время завершения.
    """

    id: str
    filename: str
    file_path: str
    file_size: int
    model: str
    language: Optional[str]

    status: str = "pending"
    stage: str = "pending"
    progress: int = 0
    message: str = ""

    logs: List[Dict] = field(default_factory=list)
    segments: List[Dict] = field(default_factory=list)
    text: str = ""
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None

    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    _subscribers: List[asyncio.Queue] = field(default_factory=list)
    _loop: Optional[asyncio.AbstractEventLoop] = None
    _cancelled: bool = False
    _repo: Optional["JobRepository"] = None
    _last_progress_save: float = 0.0

    def attach_repo(self, repo: "JobRepository") -> None:
        """
        Привязывает задачу к репозиторию для автосохранения.

        Args:
            repo: Экземпляр JobRepository.
        """
        self._repo = repo

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Привязывает задачу к event loop для рассылки SSE-событий.

        Args:
            loop: Event loop главного потока.
        """
        self._loop = loop

    def emit(self, event: Dict) -> None:
        """
        Рассылает событие всем SSE-подписчикам.

        Потокобезопасен. Если loop не привязан или закрыт, событие
        молча отбрасывается.

        Args:
            event: Словарь события с полем "type".
        """
        if self._loop is None:
            return
        for q in list(self._subscribers):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:
                pass

    def log(self, level: str, message: str) -> None:
        """
        Добавляет запись в лог и рассылает её подписчикам.

        В памяти хранится не больше 5000 записей; та же политика
        применяется в БД на стороне репозитория.

        Args:
            level: info, success, warn, error, debug.
            message: Текст записи.
        """
        entry = {"time": time.time(), "level": level, "message": message}
        self.logs.append(entry)
        if len(self.logs) > 5000:
            self.logs = self.logs[-4000:]
        if self._repo:
            try:
                self._repo.append_log(self.id, entry)
            except Exception:
                pass
        self.emit({"type": "log", "entry": entry})

    def set_progress(self, value: int, message: str = "", stage: str = None) -> None:
        """
        Обновляет прогресс и рассылает событие.

        Значение зажимается в [0, 100]. Пустые message и stage не затирают
        предыдущие. В БД сохраняется не чаще раза в секунду — при быстрых
        обновлениях прогресса (десятки раз в секунду) запись дебаунсится.

        Args:
            value: Прогресс в процентах (0-100).
            message: Описание текущего действия.
            stage: Новая стадия обработки.
        """
        self.progress = max(0, min(100, value))
        if message:
            self.message = message
        if stage:
            self.stage = stage

        now = time.time()
        if self._repo and (now - self._last_progress_save >= 1.0):
            self._last_progress_save = now
            try:
                self._repo.update_job(
                    self.id,
                    progress=self.progress,
                    message=self.message,
                    stage=self.stage,
                )
            except Exception:
                pass

        self.emit({
            "type": "progress",
            "value": self.progress,
            "message": self.message,
            "stage": self.stage,
        })

    def set_status(self, status: str, message: str = "") -> None:
        """
        Меняет статус задачи и рассылает событие.

        Args:
            status: Новый статус.
            message: Дополнительное описание.
        """
        self.status = status
        if message:
            self.message = message
        if self._repo:
            try:
                self._repo.update_job(
                    self.id, status=self.status, message=self.message,
                )
            except Exception:
                pass
        self.emit({"type": "status", "status": self.status, "message": self.message})

    def add_segment(self, start: float, end: float, text: str) -> None:
        """
        Добавляет сегмент и рассылает событие.

        Индекс назначается автоматически. Сегмент сразу сохраняется
        в БД — на случай, если сервер упадёт во время длинной задачи.

        Args:
            start: Начало сегмента в секундах.
            end: Конец сегмента в секундах.
            text: Распознанный текст.
        """
        seg = {"index": len(self.segments), "start": start, "end": end, "text": text}
        self.segments.append(seg)
        if self._repo:
            try:
                self._repo.append_segment(self.id, seg["index"], start, end, text)
            except Exception:
                pass
        self.emit({"type": "segment", "segment": seg})

    def persist_final(self) -> None:
        """
        Сохраняет финальное состояние задачи в БД.

        Вызывается один раз при завершении (успешном или нет). Записывает
        текстовый результат, метаданные, ошибку и время завершения —
        то, что не сохранялось по ходу обработки.
        """
        if not self._repo:
            return
        try:
            self._repo.update_job(
                self.id,
                status=self.status,
                stage=self.stage,
                progress=self.progress,
                message=self.message,
                text=self.text,
                metadata=self.metadata,
                error=self.error,
                finished_at=self.finished_at,
            )
        except Exception:
            pass

    def cancel(self) -> None:
        """Запрашивает отмену задачи."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Проверяет, запрошена ли отмена."""
        return self._cancelled

    def snapshot(self) -> Dict:
        """
        Возвращает полный снимок состояния задачи.

        Returns:
            Словарь со всеми полями, включая логи и сегменты.
        """
        return {
            "id": self.id,
            "filename": self.filename,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "model": self.model,
            "language": self.language,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "logs": self.logs,
            "segments": self.segments,
            "text": self.text,
            "metadata": self.metadata,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }

    def summary(self) -> Dict:
        """
        Возвращает краткую версию для списка истории.

        Returns:
            Словарь без логов и сегментов — только для отрисовки строки.
        """
        return {
            "id": self.id,
            "filename": self.filename,
            "file_size": self.file_size,
            "model": self.model,
            "status": self.status,
            "progress": self.progress,
            "segments_count": len(self.segments),
            "duration": self.metadata.get("duration"),
            "language": self.metadata.get("language"),
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    """
    Координатор активных (в памяти) и завершённых (в БД) задач.

    Активные задачи находятся в _jobs и обновляются в реальном времени.
    После завершения Job остаётся в памяти до перезапуска сервера или
    до явного удаления, но её состояние уже полностью сохранено в БД.
    Если задача запрошена по ID, а её нет в памяти — загружается из БД.

    Attributes:
        _jobs: Активные задачи в памяти.
        _repo: Репозиторий SQLite для хранения истории.
        _lock: RLock, защищающий _jobs от гонок.
    """

    def __init__(self, repo: Optional["JobRepository"] = None):
        """
        Создаёт пустой менеджер.

        Args:
            repo: Репозиторий SQLite. Если None — история не сохраняется
                (режим без БД, полезно для тестов).
        """
        self._jobs: Dict[str, Job] = {}
        self._repo = repo
        self._lock = threading.RLock()

    def create(self, **kwargs) -> Job:
        """
        Создаёт задачу, сохраняет её в БД и регистрирует в памяти.

        Args:
            **kwargs: Поля Job без id.

        Returns:
            Созданный экземпляр Job с привязанным репозиторием.
        """
        with self._lock:
            job = Job(id=uuid.uuid4().hex[:12], **kwargs)
            job.attach_repo(self._repo) if self._repo else None
            self._jobs[job.id] = job
            if self._repo:
                try:
                    self._repo.create_job({
                        "id": job.id,
                        "filename": job.filename,
                        "file_path": job.file_path,
                        "file_size": job.file_size,
                        "model": job.model,
                        "language": job.language,
                        "status": job.status,
                        "stage": job.stage,
                        "progress": job.progress,
                        "message": job.message,
                        "text": job.text,
                        "metadata": job.metadata,
                        "error": job.error,
                        "created_at": job.created_at,
                        "finished_at": job.finished_at,
                    })
                except Exception:
                    pass
            return job

    def get(self, job_id: str) -> Optional[Job]:
        """
        Возвращает Job из памяти (только активные и недавно завершённые).

        Args:
            job_id: Идентификатор задачи.

        Returns:
            Job или None, если задачи нет в памяти.
        """
        with self._lock:
            return self._jobs.get(job_id)

    def get_snapshot(self, job_id: str) -> Optional[Dict]:
        """
        Возвращает полный снимок задачи — из памяти или из БД.

        Args:
            job_id: Идентификатор задачи.

        Returns:
            Словарь-снимок задачи или None, если не найдена.
        """
        job = self.get(job_id)
        if job:
            return job.snapshot()
        if self._repo:
            return self._repo.get_job(job_id)
        return None

    def list_summaries(self, limit: int = 200, offset: int = 0) -> List[Dict]:
        """
        Возвращает краткий список задач.

        Если репозиторий есть — читает из БД (там все задачи, включая
        завершённые). Если репозитория нет — только из памяти.

        Args:
            limit: Максимум записей.
            offset: Смещение для пагинации.

        Returns:
            Список словарей-сводок, отсортированный от новых к старым.
        """
        if self._repo:
            return self._repo.list_summaries(limit=limit, offset=offset)
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [j.summary() for j in jobs[offset:offset + limit]]

    def delete(self, job_id: str, remove_file: bool = True) -> bool:
        """
        Удаляет задачу из памяти и из БД, опционально — файл с диска.

        Args:
            job_id: Идентификатор задачи.
            remove_file: Удалять ли связанный файл.

        Returns:
            True, если задача была найдена и удалена.
        """
        file_path = None
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job:
                file_path = job.file_path
            elif self._repo:
                snap = self._repo.get_job(job_id)
                if not snap:
                    return False
                file_path = snap.get("file_path")

        if self._repo:
            self._repo.delete_job(job_id)

        if remove_file and file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

        return True

    def cleanup_expired(self, retention_hours: int) -> int:
        """
        Удаляет завершённые задачи старше указанного срока.

        Args:
            retention_hours: Сколько часов хранить завершённые задачи.

        Returns:
            Количество удалённых задач.
        """
        if not self._repo:
            return 0
        cutoff = time.time() - retention_hours * 3600
        ids = self._repo.delete_finished_before(cutoff)
        with self._lock:
            for jid in ids:
                job = self._jobs.pop(jid, None)
                if job and job.file_path and os.path.exists(job.file_path):
                    try:
                        os.remove(job.file_path)
                    except OSError:
                        pass
        return len(ids)

    def cleanup_orphan_files(self, upload_dir: Path, max_age_hours: int = 48) -> int:
        """
        Удаляет файлы в upload_dir, на которые нет ссылок в БД.

        Такие файлы обычно остаются после неудачных загрузок или падений
        процесса. Файлы младше max_age_hours не трогаются.

        Args:
            upload_dir: Папка с загруженными файлами.
            max_age_hours: Минимальный возраст файла для удаления.

        Returns:
            Количество удалённых файлов.
        """
        if not upload_dir.exists():
            return 0

        active = set()
        if self._repo:
            with self._repo._lock:
                cur = self._repo._conn.cursor()
                cur.execute("SELECT file_path FROM jobs")
                active = {Path(r["file_path"]).name for r in cur.fetchall()}
                cur.close()
        with self._lock:
            active |= {Path(j.file_path).name for j in self._jobs.values()}

        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        for f in upload_dir.iterdir():
            if not f.is_file() or f.name in active:
                continue
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
        return removed


job_manager: Optional[JobManager] = None


def init_job_manager(repo: Optional["JobRepository"] = None) -> JobManager:
    """
    Инициализирует глобальный JobManager.

    Вызывается один раз при старте сервера — после того как репозиторий
    создан и все прерванные задачи помечены.

    Args:
        repo: Репозиторий SQLite (или None для работы без БД).

    Returns:
        Инициализированный JobManager.
    """
    global job_manager
    job_manager = JobManager(repo=repo)
    return job_manager


def get_job_manager() -> JobManager:
    """
    Возвращает глобальный JobManager.

    Returns:
        Singleton JobManager.

    Raises:
        RuntimeError: Если init_job_manager() не был вызван.
    """
    if job_manager is None:
        raise RuntimeError("JobManager не инициализирован. Вызовите init_job_manager().")
    return job_manager
