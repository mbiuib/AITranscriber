"""
Хранилище задач на SQLite.

Хранит метаданные задач, логи и сегменты транскрибации. Исходные файлы
остаются на диске — в БД только пути к ним. Все записи идут через один
потокобезопасный коннект с WAL-режимом: параллельное чтение возможно,
параллельная запись — под общим RLock.

Схема:
    jobs          — метаданные задач (id, filename, status, progress, ...)
    job_logs      — записи логов (job_id, time, level, message)
    job_segments  — сегменты транскрибации (job_id, idx, start, end, text)

Внешние ключи с ON DELETE CASCADE: удаление задачи автоматически удаляет
её логи и сегменты.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 2


class JobRepository:
    """
    Репозиторий задач поверх SQLite.

    Все публичные методы потокобезопасны. Соединение создаётся один раз
    и переиспользуется, поэтому класс стоит держать как singleton
    (см. get_repository в config.py).

    Attributes:
        db_path: Путь к файлу БД.
        _conn: sqlite3.Connection с check_same_thread=False.
        _lock: RLock, защищающий запись и операции с курсором.
    """

    def __init__(self, db_path: Path):
        """
        Открывает соединение и создаёт схему, если её нет.

        Args:
            db_path: Путь к файлу SQLite. Родительская папка создаётся
                автоматически, если её нет.
        """
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._conn.row_factory = sqlite3.Row
        self._init_pragmas()
        self._init_schema()

    def _init_pragmas(self) -> None:
        """
        Устанавливает PRAGMA-настройки SQLite.

        WAL позволяет читать из БД, пока идёт запись — критично для UI,
        который параллельно опрашивает список задач. synchronous=NORMAL
        даёт компромисс между скоростью и надёжностью (в случае краша
        теряются только последние транзакции, а не вся БД).
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode = WAL")
            cur.execute("PRAGMA synchronous = NORMAL")
            cur.execute("PRAGMA foreign_keys = ON")
            cur.execute("PRAGMA busy_timeout = 5000")
            cur.close()

    def _init_schema(self) -> None:
        """
        Создаёт таблицы, индексы и выполняет миграции схемы.

        Миграции идемпотентны: добавление колонок проверяется через
        PRAGMA table_info перед ALTER TABLE, потому что SQLite не
        поддерживает ADD COLUMN IF NOT EXISTS.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    model TEXT NOT NULL,
                    language TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    stage TEXT NOT NULL DEFAULT 'pending',
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    created_at REAL NOT NULL,
                    finished_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_created
                    ON jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_jobs_status
                    ON jobs(status);
                CREATE INDEX IF NOT EXISTS idx_jobs_finished
                    ON jobs(finished_at);

                CREATE TABLE IF NOT EXISTS job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    time REAL NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_logs_job
                    ON job_logs(job_id, id);

                CREATE TABLE IF NOT EXISTS job_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_segments_job
                    ON job_segments(job_id, idx);
            """)
            self._migrate(cur)
            cur.close()

    def _migrate(self, cur: sqlite3.Cursor) -> None:
        """
        Добавляет недостающие колонки в таблицу jobs.

        Каждая новая колонка проверяется через PRAGMA table_info —
        если её нет, добавляется через ALTER TABLE. Это позволяет
        обновлять приложение без пересоздания БД.

        Args:
            cur: Активный курсор SQLite.
        """
        new_columns = {
            "source_metadata": "TEXT",
            "processing_stats": "TEXT",
            "tags": "TEXT",
            "notes": "TEXT",
            "settings_snapshot": "TEXT",
            "file_hash": "TEXT",
            "starred": "INTEGER NOT NULL DEFAULT 0",
        }
        cur.execute("PRAGMA table_info(jobs)")
        existing = {row["name"] for row in cur.fetchall()}

        for col, typ in new_columns.items():
            if col not in existing:
                cur.execute(f"ALTER TABLE jobs ADD COLUMN {col} {typ}")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_file_hash ON jobs(file_hash)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_starred "
            "ON jobs(starred) WHERE starred = 1"
        )
        cur.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def create_job(self, job: Dict[str, Any]) -> None:
        """
        Регистрирует новую задачу в БД.

        Args:
            job: Словарь с полями id, filename, file_path, model.
                Дополнительно могут быть переданы source_metadata,
                settings_snapshot, file_hash, tags.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO jobs (
                    id, filename, file_path, file_size, model, language,
                    status, stage, progress, message, text, metadata,
                    error, created_at, finished_at,
                    source_metadata, settings_snapshot, file_hash, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job["id"], job["filename"], job["file_path"],
                    job.get("file_size", 0), job["model"],
                    job.get("language"),
                    job.get("status", "pending"),
                    job.get("stage", "pending"),
                    job.get("progress", 0),
                    job.get("message", ""),
                    job.get("text", ""),
                    json.dumps(job.get("metadata", {}), ensure_ascii=False),
                    job.get("error"),
                    job.get("created_at", time.time()),
                    job.get("finished_at"),
                    json.dumps(job.get("source_metadata", {}), ensure_ascii=False),
                    json.dumps(job.get("settings_snapshot", {}), ensure_ascii=False),
                    job.get("file_hash"),
                    json.dumps(job.get("tags", []), ensure_ascii=False),
                ),
            )
            cur.close()

    def update_job(self, job_id: str, **fields: Any) -> None:
        """
        Обновляет произвольные поля задачи.

        Поле metadata автоматически сериализуется в JSON, если передано
        как dict.

        Args:
            job_id: Идентификатор задачи.
            **fields: Поля для обновления (status="done", progress=100, ...).
        """
        if not fields:
            return
        if "metadata" in fields and not isinstance(fields["metadata"], str):
            fields["metadata"] = json.dumps(fields["metadata"], ensure_ascii=False)

        keys = list(fields.keys())
        values = [fields[k] for k in keys] + [job_id]
        sql = f"UPDATE jobs SET {', '.join(f'{k} = ?' for k in keys)} WHERE id = ?"

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, values)
            cur.close()

    def update_notes(self, job_id: str, tags: Optional[List[str]] = None,
                     notes: Optional[str] = None) -> bool:
        """
        Обновляет теги и/или заметки задачи.

        Args:
            job_id: Идентификатор задачи.
            tags: Новый список тегов (None — не менять).
            notes: Новый текст заметок (None — не менять).

        Returns:
            True, если запись была найдена и обновлена.
        """
        fields = {}
        if tags is not None:
            fields["tags"] = json.dumps(tags, ensure_ascii=False)
        if notes is not None:
            fields["notes"] = notes
        if not fields:
            return False

        keys = list(fields.keys())
        values = [fields[k] for k in keys] + [job_id]
        sql = f"UPDATE jobs SET {', '.join(f'{k} = ?' for k in keys)} WHERE id = ?"

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(sql, values)
            affected = cur.rowcount
            cur.close()
            return affected > 0

    def find_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Ищет завершённую задачу по SHA-256 хешу исходного файла.

        Используется для дедупликации: если пользователь загружает тот
        же файл повторно, клиент может предложить открыть существующий
        результат вместо повторной обработки.

        Args:
            file_hash: Hex-строка SHA-256.

        Returns:
            Словарь с полями id, filename, created_at, duration,
            segments_count — или None, если дубликат не найден.
            Рассматриваются только задачи со статусом done.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT j.id, j.filename, j.created_at, j.metadata,
                       (SELECT COUNT(*) FROM job_segments WHERE job_id = j.id)
                           AS segments_count
                FROM jobs j
                WHERE j.file_hash = ? AND j.status = 'done'
                ORDER BY j.created_at DESC
                LIMIT 1
                """,
                (file_hash,),
            )
            row = cur.fetchone()
            cur.close()

        if not row:
            return None
        meta = json.loads(row["metadata"] or "{}")
        return {
            "id": row["id"],
            "filename": row["filename"],
            "created_at": row["created_at"],
            "duration": meta.get("duration"),
            "segments_count": row["segments_count"],
        }

    def append_log(self, job_id: str, entry: Dict[str, Any], max_logs: int = 5000) -> None:
        """
        Добавляет запись в лог задачи.

        Автоматически удаляет самые старые записи, если их стало больше
        max_logs. Это защищает БД от разрастания при очень долгих задачах.

        Args:
            job_id: Идентификатор задачи.
            entry: Словарь с полями time, level, message.
            max_logs: Максимум записей лога на одну задачу.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO job_logs (job_id, time, level, message) VALUES (?, ?, ?, ?)",
                (job_id, entry["time"], entry["level"], entry["message"]),
            )
            cur.execute(
                """
                DELETE FROM job_logs WHERE job_id = ? AND id NOT IN (
                    SELECT id FROM job_logs WHERE job_id = ? ORDER BY id DESC LIMIT ?
                )
                """,
                (job_id, job_id, max_logs),
            )
            cur.close()

    def append_segment(self, job_id: str, idx: int, start: float,
                       end: float, text: str) -> None:
        """
        Добавляет сегмент транскрибации.

        Args:
            job_id: Идентификатор задачи.
            idx: Порядковый номер сегмента.
            start: Начало сегмента в секундах.
            end: Конец сегмента в секундах.
            text: Распознанный текст.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO job_segments (job_id, idx, start, end, text) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, idx, start, end, text),
            )
            cur.close()

    def delete_job(self, job_id: str) -> bool:
        """
        Удаляет задачу вместе с логами и сегментами (через CASCADE).

        Args:
            job_id: Идентификатор задачи.

        Returns:
            True, если запись была найдена и удалена.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            affected = cur.rowcount
            cur.close()
            return affected > 0

    def mark_interrupted(self) -> int:
        """
        Помечает все незавершённые задачи как прерванные.

        Учитывает статус queued — задачи, стоявшие в очереди на момент
        остановки сервера, не могут быть возобновлены.

        Returns:
            Количество помеченных задач.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                UPDATE jobs
                SET status = 'interrupted',
                    error = 'Прервано перезапуском сервера'
                WHERE status IN ('queued', 'pending', 'downloading', 'loading', 'transcribing')
            """)
            affected = cur.rowcount
            cur.close()
            return affected

    def delete_finished_before(self, cutoff: float) -> List[str]:
        """
        Удаляет задачи, завершённые раньше указанного времени.

        Args:
            cutoff: Unix-время. Задачи с finished_at < cutoff удаляются.

        Returns:
            Список ID удалённых задач — чтобы вызывающий код мог почистить
            их файлы на диске.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT id FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?",
                (cutoff,),
            )
            ids = [row["id"] for row in cur.fetchall()]
            if ids:
                placeholders = ",".join("?" * len(ids))
                cur.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", ids)
            cur.close()
            return ids

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает полный снимок задачи с логами, сегментами и всеми
        расширенными полями.

        Args:
            job_id: Идентификатор задачи.

        Returns:
            Словарь со всеми полями задачи или None.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cur.fetchone()
            if not row:
                cur.close()
                return None

            job = dict(row)
            for field_name in ("metadata", "source_metadata",
                               "processing_stats", "settings_snapshot"):
                if job.get(field_name):
                    try:
                        job[field_name] = json.loads(job[field_name])
                    except (json.JSONDecodeError, TypeError):
                        job[field_name] = {}
                else:
                    job[field_name] = {}

            if job.get("tags"):
                try:
                    job["tags"] = json.loads(job["tags"])
                except (json.JSONDecodeError, TypeError):
                    job["tags"] = []
            else:
                job["tags"] = []

            cur.execute(
                "SELECT time, level, message FROM job_logs WHERE job_id = ? ORDER BY id",
                (job_id,),
            )
            job["logs"] = [dict(r) for r in cur.fetchall()]

            cur.execute(
                "SELECT idx, start, end, text FROM job_segments WHERE job_id = ? ORDER BY idx",
                (job_id,),
            )
            job["segments"] = [
                {
                    "index": row["idx"],
                    "start": row["start"],
                    "end": row["end"],
                    "text": row["text"],
                }
                for row in cur.fetchall()
            ]

            cur.close()
            return job

    def list_summaries(
            self,
            limit: int = 200,
            offset: int = 0,
            query: Optional[str] = None,
            status: Optional[str] = None,
            favorite: Optional[bool] = None,
            tag: Optional[str] = None,
            date_from: Optional[float] = None,
            date_to: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает краткие сведения о задачах с фильтрацией.

        Все фильтры опциональны и комбинируются через AND. Поиск query
        проверяет имя файла, текст, заметки и теги через LIKE.

        Регистронезависимость поиска гарантируется только для ASCII:
        встроенный SQLite LIKE не учитывает юникод-регистр без расширения
        ICU. Для кириллицы поиск регистрозависим.

        Args:
            limit: Максимум записей.
            offset: Смещение для пагинации.
            query: Подстрока для поиска по filename, text, notes, tags.
            status: Фильтр по статусу (done, error, cancelled, interrupted).
            favorite: True — только избранные, False — только не-избранные,
                None — без фильтра.
            tag: Фильтр по наличию тега (точное совпадение в JSON-массиве).
            date_from: Минимальная дата создания (Unix-время).
            date_to: Максимальная дата создания (Unix-время).

        Returns:
            Список словарей, отсортированный от новых к старым.
            Каждый элемент содержит поле starred (bool).
        """
        where: List[str] = []
        params: List[Any] = []

        if query:
            like = f"%{query}%"
            where.append(
                "(j.filename LIKE ? OR j.text LIKE ? OR "
                "COALESCE(j.notes, '') LIKE ? OR COALESCE(j.tags, '') LIKE ?)"
            )
            params.extend([like, like, like, like])

        if status:
            where.append("j.status = ?")
            params.append(status)

        if favorite is True:
            where.append("j.starred = 1")
        elif favorite is False:
            where.append("(j.starred = 0 OR j.starred IS NULL)")

        if tag:
            where.append("j.tags LIKE ?")
            params.append(f'%"{tag}"%')

        if date_from is not None:
            where.append("j.created_at >= ?")
            params.append(date_from)

        if date_to is not None:
            where.append("j.created_at <= ?")
            params.append(date_to)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        params.extend([limit, offset])

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f"""
                SELECT
                    j.id, j.filename, j.file_size, j.model, j.status, j.progress,
                    j.created_at, j.finished_at, j.metadata, j.tags, j.notes,
                    j.source_metadata, j.processing_stats, j.starred,
                    (SELECT COUNT(*) FROM job_segments WHERE job_id = j.id)
                        AS segments_count
                FROM jobs j
                {where_sql}
                ORDER BY j.created_at DESC
                LIMIT ? OFFSET ?
                """,
                params,
            )
            rows = cur.fetchall()
            cur.close()

        result = []
        for row in rows:
            meta = json.loads(row["metadata"] or "{}")
            source_meta = json.loads(row["source_metadata"] or "{}")
            proc_stats = json.loads(row["processing_stats"] or "{}")
            tags = json.loads(row["tags"] or "[]")

            result.append({
                "id": row["id"],
                "filename": row["filename"],
                "file_size": row["file_size"],
                "model": row["model"],
                "status": row["status"],
                "progress": row["progress"],
                "segments_count": row["segments_count"],
                "duration": meta.get("duration") or source_meta.get("duration"),
                "language": meta.get("language"),
                "created_at": row["created_at"],
                "finished_at": row["finished_at"],
                "tags": tags,
                "notes": row["notes"] or "",
                "source_metadata": source_meta,
                "processing_stats": proc_stats,
                "starred": bool(row["starred"]),
            })
        return result

    def set_starred(self, job_id: str, starred: bool) -> bool:
        """
        Устанавливает или снимает метку «избранное».

        Args:
            job_id: Идентификатор задачи.
            starred: True — добавить в избранное, False — убрать.

        Returns:
            True, если задача была найдена и обновлена.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE jobs SET starred = ? WHERE id = ?",
                (1 if starred else 0, job_id),
            )
            affected = cur.rowcount
            cur.close()
            return affected > 0

    def update_segment_text(self, job_id: str, idx: int, text: str) -> bool:
        """
        Обновляет текст сегмента и пересобирает полный текст задачи.

        Вызывается при ручной правке расшифровки в UI. После обновления
        колонка text в таблице jobs пересчитывается из всех сегментов.

        Args:
            job_id: Идентификатор задачи.
            idx: Индекс сегмента (0-based, соответствует полю idx в БД).
            text: Новый текст сегмента.

        Returns:
            True, если сегмент был найден и обновлён.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE job_segments SET text = ? WHERE job_id = ? AND idx = ?",
                (text, job_id, idx),
            )
            if cur.rowcount == 0:
                cur.close()
                return False

            cur.execute(
                "SELECT text FROM job_segments WHERE job_id = ? ORDER BY idx",
                (job_id,),
            )
            full_text = "\n".join(row["text"] for row in cur.fetchall())
            cur.execute(
                "UPDATE jobs SET text = ? WHERE id = ?",
                (full_text, job_id),
            )
            cur.close()
            return True

    def count_active(self) -> int:
        """
        Считает активные задачи (в очереди или в обработке).

        Returns:
            Количество задач с активным статусом.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("""
                SELECT COUNT(*) AS cnt FROM jobs
                WHERE status IN ('queued', 'pending', 'downloading', 'loading', 'transcribing')
            """)
            cnt = cur.fetchone()["cnt"]
            cur.close()
            return cnt

    def close(self) -> None:
        """Закрывает соединение. Вызывается при остановке сервера."""
        with self._lock:
            self._conn.close()
