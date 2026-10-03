"""
Менеджер задач: создание, хранение, очистка.

Задача (Job) — это единица работы по транскрибации одного файла. Она
хранит состояние, прогресс, логи, сегменты и результат. JobManager
управляет жизненным циклом задач: создаёт, хранит, удаляет по запросу
и по истечении срока хранения.

Задачи живут в памяти процесса и не переживают перезапуск сервера.
Для сохранения истории между запусками потребуется внешнее хранилище
(SQLite, Redis) — сейчас это не реализовано.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Job:
    """
    Единица работы по транскрибации одного файла.

    Хранит входные параметры, текущее состояние обработки, накопленные
    логи и сегменты текста, а также метаданные по завершении. Умеет
    рассылать события SSE-подписчикам через emit().

    Задача не потокобезопасна сама по себе — предполагается, что все
    вызовы мутирующих методов идут из одного потока (воркера), а emit()
    безопасен к вызову из другого потока благодаря call_soon_threadsafe.

    Attributes:
        id: Уникальный 12-символьный идентификатор задачи.
        filename: Исходное имя загруженного файла.
        file_path: Путь к сохранённому файлу на диске.
        file_size: Размер файла в байтах.
        model: Название модели Whisper (например, "large-v3").
        language: Код языка или None для автоопределения.
        status: Общий статус задачи: pending, downloading, loading,
            transcribing, done, error, cancelled.
        stage: Более детальная стадия для UI (совпадает с status
            в большинстве случаев, но может уточняться).
        progress: Прогресс в процентах от 0 до 100.
        message: Человекочитаемое описание текущего действия.
        logs: Накопленные записи логов (ограничено 5000 последних).
        segments: Распознанные сегменты текста с таймкодами.
        text: Полный распознанный текст (склеен из сегментов).
        metadata: Итоговые метаданные (язык, длительность, скорость).
        error: Текст ошибки, если задача завершилась неудачно.
        created_at: Unix-время создания задачи.
        finished_at: Unix-время завершения (успешного или нет).
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

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Привязывает задачу к event loop, в котором будут рассылаться события.

        Обязательный шаг перед использованием emit() из воркера. Обычно
        вызывается один раз при создании задачи в HTTP-обработчике, где
        доступен текущий loop через asyncio.get_running_loop().

        Args:
            loop: Event loop главного потока, куда будут доставляться
                события SSE-подписчиков.
        """
        self._loop = loop

    def emit(self, event: Dict) -> None:
        """
        Рассылает событие всем активным SSE-подписчикам.

        Потокобезопасна: использует call_soon_threadsafe, чтобы передать
        событие в event loop главного потока. Если loop ещё не привязан
        через set_loop() или уже закрыт, событие молча отбрасывается —
        это нормально на ранних этапах жизненного цикла задачи.

        Args:
            event: Словарь с полем "type" и произвольной полезной нагрузкой.
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
        Добавляет запись в лог задачи и рассылает её подписчикам.

        Лог ограничен 5000 последними записями: при превышении старые
        удаляются пачкой по 1000, чтобы не резать историю на каждой
        строке.

        Args:
            level: Уровень: "info", "success", "warn", "error", "debug".
            message: Текст записи.
        """
        entry = {"time": time.time(), "level": level, "message": message}
        self.logs.append(entry)
        if len(self.logs) > 5000:
            self.logs = self.logs[-4000:]
        self.emit({"type": "log", "entry": entry})

    def set_progress(self, value: int, message: str = "", stage: str = None) -> None:
        """
        Обновляет прогресс задачи и рассылает событие подписчикам.

        Значение автоматически зажимается в диапазон [0, 100]. Пустые
        message и stage не затирают ранее установленные значения — это
        позволяет обновлять только процент без потери контекста.

        Args:
            value: Прогресс в процентах (0–100).
            message: Описание текущего действия (опционально).
            stage: Новая стадия обработки (опционально).
        """
        self.progress = max(0, min(100, value))
        if message:
            self.message = message
        if stage:
            self.stage = stage
        self.emit({
            "type": "progress", "value": self.progress,
            "message": self.message, "stage": self.stage,
        })

    def set_status(self, status: str, message: str = "") -> None:
        """
        Меняет статус задачи и рассылает событие подписчикам.

        Args:
            status: Новый статус из фиксированного набора: pending,
                downloading, loading, transcribing, done, error, cancelled.
            message: Дополнительное описание (опционально).
        """
        self.status = status
        if message:
            self.message = message
        self.emit({"type": "status", "status": self.status, "message": self.message})

    def add_segment(self, start: float, end: float, text: str) -> None:
        """
        Добавляет распознанный сегмент и рассылает событие подписчикам.

        Индекс сегмента назначается автоматически и отражает порядок
        поступления. UI использует его для нумерации строк.

        Args:
            start: Начало сегмента в секундах от начала аудио.
            end: Конец сегмента в секундах.
            text: Распознанный текст сегмента (уже обрезанный по краям).
        """
        seg = {"index": len(self.segments), "start": start, "end": end, "text": text}
        self.segments.append(seg)
        self.emit({"type": "segment", "segment": seg})

    def cancel(self) -> None:
        """
        Запрашивает отмену задачи.

        Воркер проверяет флаг через is_cancelled() между шагами обработки
        и корректно завершает работу. Мгновенной остановки не происходит —
        нельзя прервать уже запущенный батч инференса, но следующий шаг
        не начнётся.
        """
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """
        Проверяет, запрошена ли отмена задачи.

        Returns:
            True, если был вызван cancel().
        """
        return self._cancelled

    def snapshot(self) -> Dict:
        """
        Возвращает полный снимок состояния задачи.

        Используется при первичной подписке на SSE — клиент получает всё
        накопленное состояние одним куском, чтобы восстановить UI после
        переподключения.

        Returns:
            Словарь со всеми полями задачи, включая логи и сегменты.
        """
        return {
            "id": self.id,
            "filename": self.filename,
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
        Возвращает краткую версию состояния для списка истории.

        В отличие от snapshot(), не включает логи, сегменты и полный
        текст — только метаданные, необходимые для отображения строки
        в списке задач.

        Returns:
            Словарь с ключевыми полями: id, filename, статус, количество
            сегментов, длительность, язык и временные метки.
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
    Хранилище задач в памяти процесса.

    Управляет созданием, поиском, удалением и периодической очисткой
    задач. Все публичные методы потокобезопасны — защищены общим RLock.

    При удалении задачи опционально удаляется её файл с диска. Также
    поддерживается очистка «осиротевших» файлов — тех, что лежат в
    uploads/, но не связаны ни с одной активной задачей (например,
    остатки от упавших загрузок).

    Attributes:
        _jobs: Словарь id → Job.
        _lock: RLock, защищающий _jobs от одновременного доступа.
    """

    def __init__(self):
        """Создаёт пустое хранилище задач."""
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()

    def create(self, **kwargs) -> Job:
        """
        Создаёт задачу и регистрирует её в хранилище.

        Args:
            **kwargs: Поля Job без id — id генерируется автоматически
                как 12-символьный hex-идентификатор.

        Returns:
            Созданный экземпляр Job.
        """
        with self._lock:
            job = Job(id=uuid.uuid4().hex[:12], **kwargs)
            self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> Optional[Job]:
        """
        Возвращает задачу по идентификатору.

        Args:
            job_id: Идентификатор задачи.

        Returns:
            Экземпляр Job или None, если задача не найдена.
        """
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        """
        Возвращает все задачи, отсортированные по времени создания.

        Свежие задачи идут первыми — это ожидаемый порядок для списка
        истории в UI.

        Returns:
            Список задач от новых к старым.
        """
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str, remove_file: bool = True) -> bool:
        """
        Удаляет задачу и опционально её файл с диска.

        Args:
            job_id: Идентификатор задачи.
            remove_file: Удалять ли связанный файл. Ошибки удаления
                файла игнорируются — задача всё равно снимается с учёта.

        Returns:
            True, если задача была найдена и удалена, False в противном случае.
        """
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if not job:
            return False
        if remove_file and job.file_path and os.path.exists(job.file_path):
            try:
                os.remove(job.file_path)
            except OSError:
                pass
        return True

    def cleanup_expired(self, retention_hours: int) -> int:
        """
        Удаляет завершённые задачи старше указанного срока.

        Учитываются только задачи с заполненным finished_at — то есть
        завершённые (успешно или с ошибкой). Активные задачи не трогаются
        независимо от их возраста.

        Args:
            retention_hours: Сколько часов хранить задачи после завершения.

        Returns:
            Количество удалённых задач.
        """
        cutoff = time.time() - retention_hours * 3600
        removed = 0
        with self._lock:
            to_remove = [
                j.id for j in self._jobs.values()
                if j.finished_at and j.finished_at < cutoff
            ]
        for job_id in to_remove:
            if self.delete(job_id):
                removed += 1
        return removed

    def cleanup_orphan_files(self, upload_dir: Path, max_age_hours: int = 48) -> int:
        """
        Удаляет файлы в upload_dir, на которые нет ссылок в задачах.

        Такие файлы обычно остаются после неудачных загрузок или падений
        процесса. Файлы младше max_age_hours не трогаются, чтобы не
        удалить загрузку, которая идёт прямо сейчас.

        Args:
            upload_dir: Папка с загруженными файлами.
            max_age_hours: Минимальный возраст файла для удаления, в часах.

        Returns:
            Количество удалённых файлов.
        """
        if not upload_dir.exists():
            return 0
        active = {Path(j.file_path).name for j in self._jobs.values()}
        cutoff = time.time() - max_age_hours * 3600
        removed = 0
        for f in upload_dir.iterdir():
            if not f.is_file():
                continue
            if f.name in active:
                continue
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                pass
        return removed


job_manager = JobManager()
