"""
FastAPI-сервер приложения.

Собирает все HTTP-эндпоинты, статику и фоновые задачи в одном приложении:
  - REST API для управления задачами транскрибации;
  - SSE-стрим для live-обновлений прогресса;
  - API настроек, локалей и системных метрик;
  - фоновый поток автоочистки завершённых задач и осиротевших файлов.

Точка входа для запуска — run.py, который импортирует app и передаёт его
в uvicorn. При импорте модуль читает .env и data/settings.json, поэтому
все настройки должны быть валидны ещё до старта сервера.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import RUNTIME_SCHEMA, get_env, get_store
from .i18n import available as locales_available, load as load_locale
from .config import get_repository
from .jobs import init_job_manager, get_job_manager
from . import monitor, transcribe
from .queue import init_task_queue, get_task_queue

BASE_DIR = Path(__file__).parent.parent
env = get_env()
store = get_store()

TAGS_METADATA = [
    {
        "name": "system",
        "description": "Информация о запущенном инстансе и системных ресурсах: "
                       "версия, аптайм, CPU, RAM, GPU, диски.",
    },
    {
        "name": "jobs",
        "description": "Управление задачами транскрибации: создание, мониторинг, "
                       "отмена, удаление. Включает SSE-поток для live-обновлений.",
    },
    {
        "name": "settings",
        "description": "Runtime-настройки приложения. Значения по умолчанию "
                       "читаются из .env, переопределения сохраняются в data/settings.json.",
    },
    {
        "name": "i18n",
        "description": "Локализация интерфейса: список доступных языков и "
                       "переводы по ключам.",
    },
]

DESCRIPTION = """
**Whisper Transcriber** — веб-приложение для транскрибации аудио и видео
с использованием локальных моделей Whisper (faster-whisper + CTranslate2).

### Возможности

* 🎙️ **Локальная транскрибация** — модель работает на вашей GPU/CPU, данные не уходят в облако
* 🚀 **GPU-ускорение** — CUDA + float16, поддержка RTX 40xx/50xx
* 📊 **Live-прогресс** — SSE-стрим событий: логи, прогресс, сегменты в реальном времени
* 🌍 **Мультиязычность** — русский, английский, немецкий, французский и др.
* ⚙️ **Runtime-настройки** — смена модели, языка, batch_size без перезапуска сервера
* 🧹 **Автоочистка** — старые задачи и файлы удаляются по расписанию
* 📁 **Экспорт** — TXT, SRT (субтитры), JSON (структурированные данные)

### Быстрый старт

1. Создайте задачу через `POST /api/jobs` с файлом
2. Подпишитесь на события через `GET /api/jobs/{job_id}/stream`
3. Получите результат через `GET /api/jobs/{job_id}/result`

### Формат SSE-событий

Каждое событие — JSON-объект с полем `type`:

* `snapshot` — полный снимок состояния при подключении
* `log` — запись лога (`entry: {time, level, message}`)
* `progress` — обновление прогресса (`value`, `message`, `stage`)
* `status` — смена статуса (`status`, `message`)
* `segment` — новый распознанный сегмент (`segment: {start, end, text}`)
* `done` — задача завершена (`text`, `metadata`)
* `error` — ошибка (`message`)
* `cancelled` — задача отменена
"""

app = FastAPI(
    title="Whisper Transcriber API",
    description=DESCRIPTION,
    version=__version__,
    openapi_tags=TAGS_METADATA,
    contact={
        "name": "Whisper Transcriber",
        "url": "https://github.com/mbiuib/AITranscriber",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=env.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

STARTED_AT = time.time()


def _cleanup_loop() -> None:
    """Фоновый цикл автоочистки завершённых задач и осиротевших файлов."""
    while True:
        try:
            jm = get_job_manager()
            hours = int(store.get("server.retention_hours"))
            interval = int(store.get("server.cleanup_interval_min"))
            removed = jm.cleanup_expired(hours)
            orphans = jm.cleanup_orphan_files(env.upload_dir, max_age_hours=hours * 2)
            if removed or orphans:
                print(f"[cleanup] Удалено задач: {removed}, файлов: {orphans}")
            time.sleep(interval * 60)
        except Exception as e:
            print(f"[cleanup] ошибка: {e}")
            time.sleep(60)


@app.on_event("startup")
async def _startup() -> None:
    """
    Инициализирует репозиторий, очередь задач и фоновую очистку.

    Порядок важен: сначала помечаем прерванные задачи, потом создаём
    JobManager, потом инициализируем очередь — она вызывает
    get_job_manager() при первом же воркере.
    """
    env.upload_dir.mkdir(parents=True, exist_ok=True)
    env.models_dir.mkdir(parents=True, exist_ok=True)

    repo = get_repository()
    interrupted = repo.mark_interrupted()
    if interrupted:
        print(f"[startup] Помечено прерванных задач: {interrupted}")

    init_job_manager(repo=repo)

    jm = get_job_manager()
    max_parallel = int(store.get("server.max_parallel_jobs"))
    init_task_queue(
        get_job=lambda jid: jm.get(jid),
        run_job=transcribe.run,
        max_parallel=max_parallel,
    )
    print(f"[startup] Очередь задач: параллелизм = {max_parallel}")

    threading.Thread(target=_cleanup_loop, daemon=True).start()


@app.get(
    "/api/system",
    tags=["system"],
    summary="Информация о приложении",
    description="Возвращает версию, аптайм и пути к рабочим папкам.",
)
async def system_info() -> Dict[str, Any]:
    """
    Возвращает базовую информацию о запущенном инстансе.

    Используется UI для отображения версии и аптайма, а также для
    проверки доступности сервера.

    Returns:
        Словарь с версией, аптаймом в секундах и путями к рабочим папкам.
    """
    return {
        "version": __version__,
        "uptime_sec": time.time() - STARTED_AT,
        "models_dir": str(env.models_dir),
        "upload_dir": str(env.upload_dir),
    }


@app.get(
    "/api/system/resources",
    tags=["system"],
    summary="Снимок ресурсов системы",
    description="CPU, RAM, GPU (если доступна), диски и скользящая история "
                "последних 60 замеров для отображения графиков.",
)
async def system_resources() -> Dict[str, Any]:
    """
    Возвращает текущий снимок ресурсов системы.

    Включает CPU, RAM, GPU (если доступна), использование дисков и
    скользящую историю последних замеров для отображения графиков.

    Returns:
        Словарь из monitor.snapshot() — см. его докстринг для полной схемы.
    """
    return monitor.snapshot(env.models_dir, env.upload_dir)


@app.get("/api/dashboard", tags=["system"], summary="Агрегированная статистика")
async def dashboard() -> Dict[str, Any]:
    """
    Возвращает агрегированную статистику для дашборда.

    Объединяет данные из БД (количество задач, длительности, топ моделей)
    и текущее состояние очереди.

    Returns:
        Словарь с полями из JobRepository.dashboard_stats() плюс
        queue: {size, running, max_parallel}.
    """
    repo = get_repository()
    stats = repo.dashboard_stats()

    try:
        tq = get_task_queue()
        stats["queue"] = {
            "size": tq.size(),
            "running": tq.running_count(),
            "max_parallel": tq.get_max_parallel(),
        }
    except RuntimeError:
        stats["queue"] = {"size": 0, "running": 0, "max_parallel": 1}

    return stats


@app.get("/api/settings/schema", tags=["settings"], summary="Схема runtime-настроек")
async def settings_schema() -> Dict[str, Any]:
    """
    Возвращает схему runtime-настроек.

    UI использует схему для автоматической генерации форм: типы полей,
    допустимые значения, подсказки, порядок и группировку.

    Returns:
        Словарь с единственным ключом "schema", содержащим описание
        всех доступных настроек.
    """
    return {"schema": RUNTIME_SCHEMA}


@app.get("/api/settings", tags=["settings"], summary="Текущие значения настроек")
async def get_settings() -> Dict[str, Any]:
    """
    Возвращает текущие значения всех runtime-настроек.

    Включает как переопределения пользователя, так и значения
    по умолчанию из .env для ключей, которые не переопределены.

    Returns:
        Словарь с единственным ключом "values".
    """
    return {"values": store.all()}


@app.put("/api/settings", tags=["settings"], summary="Обновить настройки")
async def update_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обновляет одну или несколько runtime-настроек.

    Каждое значение валидируется по схеме: тип, диапазон для int и
    список допустимых значений для select. Неизвестные ключи молча
    игнорируются — это позволяет клиенту отправлять полный снапшот
    формы, даже если часть полей сервер не знает.

    Args:
        payload: Словарь либо с ключом "values", либо с самими
            значениями в корне.

    Returns:
        Словарь {"ok": True, "values": ...} с полным актуальным
        состоянием настроек после обновления.

    Raises:
        HTTPException: 400, если значение не прошло валидацию.
    """
    values = payload.get("values", payload)
    clean: Dict[str, Any] = {}
    for key, value in values.items():
        schema = RUNTIME_SCHEMA.get(key)
        if not schema:
            continue
        t = schema["type"]
        try:
            if t == "int":
                v = int(value)
                if "min" in schema and v < schema["min"]:
                    raise ValueError(f"{key} < {schema['min']}")
                if "max" in schema and v > schema["max"]:
                    raise ValueError(f"{key} > {schema['max']}")
                clean[key] = v
            elif t == "bool":
                clean[key] = bool(value)
            elif t == "select":
                if value not in schema["options"]:
                    raise ValueError(f"{key}: недопустимое значение {value}")
                clean[key] = value
            else:
                clean[key] = str(value)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"Некорректное значение {key}: {e}")
    store.set_many(clean)
    if "server.max_parallel_jobs" in clean:
        try:
            actual = get_task_queue().set_max_parallel(int(clean["server.max_parallel_jobs"]))
            print(f"[settings] Параллелизм изменён на {actual}")
        except RuntimeError:
            pass

    return {"ok": True, "values": store.all()}


@app.post("/api/settings/reset", tags=["settings"], summary="Сбросить настройки к значениям из .env")
async def reset_settings(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Сбрасывает настройки к значениям по умолчанию из .env.

    Args:
        payload: Опциональный словарь с ключом "key". Если указан —
            сбрасывается только эта настройка, иначе все сразу.

    Returns:
        Словарь {"ok": True, "values": ...} с полным состоянием
        настроек после сброса.
    """
    key = (payload or {}).get("key")
    store.reset(key)
    return {"ok": True, "values": store.all()}


@app.get("/api/locales", tags=["i18n"], summary="Список доступных локалей")
async def locales_list() -> Dict[str, Any]:
    """
    Возвращает список доступных локалей и текущий выбранный язык.

    UI использует это для отрисовки переключателя языков и подсветки
    активного варианта.

    Returns:
        Словарь {"available": [...], "current": "ru"}.
    """
    return {
        "available": locales_available(),
        "current": store.get("ui.language"),
    }


@app.get("/api/locales/{lang}", tags=["i18n"], summary="Переводы для указанного языка")
async def locale_get(lang: str) -> Dict[str, str]:
    """
    Возвращает перевод интерфейса для указанного языка.

    Args:
        lang: Код языка, обычно двухбуквенный ("ru", "en").

    Returns:
        Плоский словарь "ключ → строка".

    Raises:
        HTTPException: 404, если локаль пуста или не найдена.
    """
    data = load_locale(lang)
    if not data:
        raise HTTPException(404, f"Локаль {lang} не найдена")
    return data


@app.get("/api/jobs", tags=["jobs"], summary="Список задач с фильтрацией")
async def list_jobs(
    q: Optional[str] = Query(None, description="Поиск по имени, тексту, заметкам, тегам"),
    status: Optional[str] = Query(None, description="Фильтр по статусу"),
    favorite: Optional[bool] = Query(None, description="True — только избранные"),
    tag: Optional[str] = Query(None, description="Фильтр по тегу"),
    date_from: Optional[float] = Query(None, description="Мин. дата создания (Unix)"),
    date_to: Optional[float] = Query(None, description="Макс. дата создания (Unix)"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    Возвращает краткий список задач с фильтрацией.

    Для задач в статусе queued добавляется поле queue_position —
    порядковый номер в очереди (1-based).

    Returns:
        Словарь с полями jobs, offset, limit, active_count,
        queue_size, queue_running, queue_max_parallel.
    """
    repo = get_repository()
    jobs = repo.list_summaries(
        limit=limit, offset=offset,
        query=q, status=status, favorite=favorite, tag=tag,
        date_from=date_from, date_to=date_to,
    )

    try:
        tq = get_task_queue()
        positions = tq.positions()
        for j in jobs:
            if j["status"] == "queued":
                pos = positions.get(j["id"])
                if pos is not None:
                    j["queue_position"] = pos
        queue_size = tq.size()
        queue_running = tq.running_count()
        queue_max = tq.get_max_parallel()
    except RuntimeError:
        queue_size = 0
        queue_running = 0
        queue_max = 1

    return {
        "jobs": jobs,
        "offset": offset,
        "limit": limit,
        "active_count": repo.count_active(),
        "queue_size": queue_size,
        "queue_running": queue_running,
        "queue_max_parallel": queue_max,
    }


async def _save_and_create_job(
    file: UploadFile,
    model: str,
    language: Optional[str],
    check_duplicate: bool,
    engine: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Сохраняет файл на диск и создаёт задачу.

    Общий код для одиночной и batch-загрузки. При ошибке файл удаляется,
    задача снимается с учёта. Возвращает словарь одного из видов:

        {"status": "created", "job_id": "...", "filename": "..."}
        {"status": "duplicate", "filename": "...", "duplicate_of": {...}}
        {"status": "error", "filename": "...", "message": "..."}

    Args:
        file: Загруженный файл.
        model: Название модели Whisper.
        language: Код языка или None для авто.
        check_duplicate: Проверять ли дубликат по SHA-256.

    Returns:
        Словарь с результатом обработки одного файла.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in transcribe.ALL_SUPPORTED:
        return {
            "status": "error",
            "filename": file.filename,
            "message": f"Неподдерживаемый формат: {ext}",
        }

    max_mb = int(store.get("server.max_upload_mb"))
    max_bytes = max_mb * 1024 * 1024

    jm = get_job_manager()
    repo = get_repository()

    job = jm.create(
        filename=file.filename,
        file_path="",
        file_size=0,
        model=model,
        language=language,
        engine=engine,
    )
    job.set_loop(asyncio.get_running_loop())

    file_path = env.upload_dir / f"{job.id}{ext}"
    job.file_path = str(file_path)
    if jm._repo:
        try:
            jm._repo.update_job(job.id, file_path=str(file_path))
        except Exception:
            pass

    size = 0
    hasher = hashlib.sha256()
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    f.close()
                    file_path.unlink(missing_ok=True)
                    jm.delete(job.id)
                    return {
                        "status": "error",
                        "filename": file.filename,
                        "message": f"Файл превышает лимит {max_mb} МБ",
                    }
                f.write(chunk)
                hasher.update(chunk)
        job.file_size = size
    except Exception as e:
        file_path.unlink(missing_ok=True)
        jm.delete(job.id)
        return {
            "status": "error",
            "filename": file.filename,
            "message": f"Ошибка сохранения: {e}",
        }

    file_hash = hasher.hexdigest()
    if jm._repo:
        try:
            jm._repo.update_job(job.id, file_hash=file_hash)
        except Exception:
            pass

    if check_duplicate:
        existing = repo.find_by_hash(file_hash)
        if existing:
            file_path.unlink(missing_ok=True)
            jm.delete(job.id)
            return {
                "status": "duplicate",
                "filename": file.filename,
                "duplicate_of": existing,
            }

    job.set_status("queued", "В очереди")
    job.log("info", f"Задача {job.id} создана: {file.filename} ({size / 1024 / 1024:.1f} МБ)")
    get_task_queue().put(job.id)

    return {
        "status": "created",
        "job_id": job.id,
        "filename": file.filename,
        "file_size": size,
    }


@app.post(
    "/api/jobs",
    tags=["jobs"],
    summary="Создать задачу транскрибации",
    responses={
        200: {"description": "Задача создана"},
        400: {"description": "Неподдерживаемый формат"},
        409: {"description": "Найден дубликат"},
        413: {"description": "Файл превышает лимит"},
    },
)
async def create_job(
    file: UploadFile = File(..., description="Аудио или видеофайл"),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    engine: Optional[str] = Form(None),
    check_duplicate: bool = Form(True),
) -> Dict[str, Any]:
    """
    Создаёт одну задачу транскрибации.

    Args:
        file: Загруженный файл.
        model: Название модели Whisper.
        language: Код языка или "auto".
        check_duplicate: Если True и найден дубликат — 409.

    Returns:
        {"job_id": "...", "status": "queued"}.

    Raises:
        HTTPException: 400 — неподдерживаемый формат;
            409 — дубликат (detail содержит duplicate_of);
            413 — превышен лимит;
            500 — ошибка сохранения.
    """
    model = model or store.get("transcription.model")
    language = None if (language in (None, "auto", "")) else language

    if engine in (None, ""):
        engine = None
    elif engine not in ("whisper", "moss"):
        engine = None

    result = await _save_and_create_job(
        file, model, language, check_duplicate, engine,
    )

    if result["status"] == "created":
        return {"job_id": result["job_id"], "status": "queued"}
    if result["status"] == "duplicate":
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Найден идентичный файл",
                "duplicate_of": result["duplicate_of"],
            },
        )
    raise HTTPException(400, result.get("message", "Ошибка создания задачи"))


@app.post(
    "/api/jobs/batch",
    tags=["jobs"],
    summary="Создать несколько задач за один запрос",
    description="Принимает список файлов. Каждый обрабатывается независимо: "
                "ошибка или дубликат на одном не валит остальные. Результат "
                "содержит массив по каждому файлу и сводку.",
)
async def create_jobs_batch(
    files: List[UploadFile] = File(..., description="Список файлов"),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    engine: Optional[str] = Form(None),
    check_duplicate: bool = Form(True),
) -> Dict[str, Any]:
    """
    Создаёт несколько задач за один запрос.

    Обрабатывает файлы последовательно, чтобы не перегружать диск
    параллельными записями. Задачи попадают в очередь по мере создания.

    Args:
        files: Список загруженных файлов.
        model: Название модели для всех задач.
        language: Код языка или "auto" для всех задач.
        check_duplicate: Проверять ли дубликаты по SHA-256.

    Returns:
        Словарь {"results": [...], "summary": {created, duplicates, errors}}.

    Raises:
        HTTPException: 400 — если список пуст.
    """
    if not files:
        raise HTTPException(400, "Список файлов пуст")

    model = model or store.get("transcription.model")
    language = None if (language in (None, "auto", "")) else language

    if engine in (None, ""):
        engine = None
    elif engine not in ("whisper", "moss"):
        engine = None

    results: List[Dict[str, Any]] = []
    created = duplicates = errors = 0

    for idx, f in enumerate(files):
        try:
            r = await _save_and_create_job(
                f, model, language, check_duplicate, engine,
            )
        except Exception as e:
            r = {
                "status": "error",
                "filename": f.filename,
                "message": f"{type(e).__name__}: {e}",
            }
        r["index"] = idx
        results.append(r)

        if r["status"] == "created":
            created += 1
        elif r["status"] == "duplicate":
            duplicates += 1
        else:
            errors += 1

    return {
        "results": results,
        "summary": {
            "total": len(files),
            "created": created,
            "duplicates": duplicates,
            "errors": errors,
        },
    }


@app.patch(
    "/api/jobs/{job_id}/notes",
    tags=["jobs"],
    summary="Обновить теги и заметки задачи",
    description="Принимает JSON вида `{\"tags\": [\"встреча\"], \"notes\": \"текст\"}`. "
                "Оба поля опциональны — можно менять по отдельности.",
)
async def update_job_notes(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обновляет пользовательские теги и заметки задачи.

    Args:
        job_id: Идентификатор задачи.
        payload: Словарь с полями tags (list[str]) и/или notes (str).

    Returns:
        Словарь {"ok": True, "tags": [...], "notes": "..."}.

    Raises:
        HTTPException: 404, если задача не найдена;
                       400, если типы полей некорректны.
    """
    tags = payload.get("tags")
    notes = payload.get("notes")

    if tags is not None and not isinstance(tags, list):
        raise HTTPException(400, "tags должен быть массивом строк")
    if tags is not None:
        tags = [str(t).strip()[:32] for t in tags if str(t).strip()][:16]
    if notes is not None and not isinstance(notes, str):
        raise HTTPException(400, "notes должен быть строкой")
    if notes is not None:
        notes = notes[:10000]

    repo = get_repository()
    ok = repo.update_notes(job_id, tags=tags, notes=notes)
    if not ok:
        raise HTTPException(404, "Задача не найдена")

    snap = repo.get_job(job_id)
    return {"ok": True, "tags": snap.get("tags", []), "notes": snap.get("notes", "")}


@app.patch(
    "/api/jobs/{job_id}/star",
    tags=["jobs"],
    summary="Установить или снять метку избранного",
)
async def set_job_star(job_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Устанавливает или снимает метку «избранное».

    Args:
        job_id: Идентификатор задачи.
        payload: {"starred": true|false}.

    Returns:
        {"ok": True, "starred": <актуальное значение>}.

    Raises:
        HTTPException: 400 — некорректный payload;
                       404 — задача не найдена.
    """
    if not isinstance(payload, dict) or "starred" not in payload:
        raise HTTPException(400, "Ожидается {'starred': bool}")
    starred = bool(payload["starred"])

    repo = get_repository()
    if not repo.set_starred(job_id, starred):
        raise HTTPException(404, "Задача не найдена")
    return {"ok": True, "starred": starred}


@app.patch(
    "/api/jobs/{job_id}/segments/{idx}",
    tags=["jobs"],
    summary="Обновить текст сегмента",
    description="Заменяет текст сегмента и пересобирает полный текст задачи. "
                "Синхронизирует состояние активной задачи в памяти с БД.",
)
async def update_segment(
    job_id: str, idx: int, payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Обновляет текст сегмента и пересчитывает полный текст задачи.

    Используется для ручной правки расшифровки. Если задача активна
    (есть в памяти), обновляется и её in-memory копия, чтобы SSE-подписчики
    видели согласованное состояние.

    Args:
        job_id: Идентификатор задачи.
        idx: Индекс сегмента (0-based).
        payload: {"text": "новый текст"}.

    Returns:
        {"ok": True, "text": "новый полный текст задачи"}.

    Raises:
        HTTPException: 400 — некорректный payload;
                       404 — задача или сегмент не найдены.
    """
    if not isinstance(payload, dict) or "text" not in payload:
        raise HTTPException(400, "Ожидается {'text': str}")
    text = str(payload["text"]).strip()
    if not text:
        raise HTTPException(400, "Текст не может быть пустым")

    repo = get_repository()
    if not repo.get_job(job_id):
        raise HTTPException(404, "Задача не найдена")

    if not repo.update_segment_text(job_id, idx, text):
        raise HTTPException(404, f"Сегмент {idx} не найден")

    snap = repo.get_job(job_id)
    full_text = snap.get("text", "")

    jm = get_job_manager()
    active = jm.get(job_id)
    if active:
        for seg in active.segments:
            if seg["index"] == idx:
                seg["text"] = text
                break
        active.text = full_text

    return {"ok": True, "text": full_text}


@app.patch(
    "/api/jobs/{job_id}/speakers/{speaker}",
    tags=["jobs"],
    summary="Переименовать спикера",
    description="Меняет отображаемое имя спикера во всех сегментах задачи. "
                "Хранится в metadata.speaker_names.",
)
async def rename_speaker(
    job_id: str, speaker: str, payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Переименовывает спикера во всех сегментах задачи.

    Args:
        job_id: Идентификатор задачи.
        speaker: Исходный идентификатор (SPEAKER_00).
        payload: {"name": "Новое имя"}.

    Returns:
        {"ok": True, "speaker_names": {...}}.

    Raises:
        HTTPException: 400 — некорректный payload;
                       404 — задача не найдена.
    """
    if not isinstance(payload, dict) or "name" not in payload:
        raise HTTPException(400, "Ожидается {'name': str}")
    name = str(payload["name"]).strip()[:64]
    if not name:
        raise HTTPException(400, "Имя не может быть пустым")

    jm = get_job_manager()
    snap = jm.get_snapshot(job_id)
    if not snap:
        raise HTTPException(404, "Задача не найдена")

    meta = snap.get("metadata") or {}
    speaker_names = meta.get("speaker_names", {})
    speaker_names[speaker] = name
    meta["speaker_names"] = speaker_names

    repo = get_repository()
    repo.update_job(job_id, metadata=json.dumps(meta, ensure_ascii=False))

    # Обновляем активную задачу в памяти
    active = jm.get(job_id)
    if active:
        active.metadata = meta

    return {"ok": True, "speaker_names": speaker_names}


@app.get("/api/jobs/{job_id}", tags=["jobs"], summary="Полный снимок задачи")
async def get_job(job_id: str) -> Dict[str, Any]:
    """Возвращает полный снимок задачи из памяти или из БД."""
    snap = get_job_manager().get_snapshot(job_id)
    if not snap:
        raise HTTPException(404, "Задача не найдена")
    return snap


@app.post("/api/jobs/{job_id}/cancel", tags=["jobs"], summary="Отменить задачу")
async def cancel_job(job_id: str) -> Dict[str, bool]:
    """
    Запрашивает отмену задачи.

    Если задача ещё в очереди — она немедленно удаляется из неё и
    получает статус cancelled. Если выполняется — устанавливается флаг,
    и воркер прервёт её при первой возможности.

    Args:
        job_id: Идентификатор задачи.

    Returns:
        {"ok": True}.

    Raises:
        HTTPException: 404, если задача не найдена;
            409, если задача уже завершена.
    """
    jm = get_job_manager()
    job = jm.get(job_id)
    if not job:
        snap = jm.get_snapshot(job_id)
        if not snap:
            raise HTTPException(404, "Задача не найдена")
        raise HTTPException(409, f"Задача уже завершена ({snap['status']})")

    job.cancel()
    get_task_queue().remove(job_id)
    return {"ok": True}


@app.delete("/api/jobs/{job_id}", tags=["jobs"], summary="Удалить задачу")
async def delete_job(job_id: str) -> Dict[str, bool]:
    """
    Удаляет задачу вместе с её файлом.

    Сначала вынимает из очереди, если задача там, потом удаляет из БД.

    Args:
        job_id: Идентификатор задачи.

    Returns:
        {"ok": True}.

    Raises:
        HTTPException: 404, если задача не найдена.
    """
    get_task_queue().remove(job_id)
    ok = get_job_manager().delete(job_id, remove_file=True)
    if not ok:
        raise HTTPException(404, "Задача не найдена")
    return {"ok": True}


@app.post(
    "/api/jobs/batch/delete",
    tags=["jobs"],
    summary="Массовое удаление задач",
    description="Принимает список ID задач и удаляет их вместе с файлами. "
                "Отсутствующие ID игнорируются.",
)
async def delete_jobs_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Удаляет несколько задач за один запрос.

    Args:
        payload: {"ids": ["id1", "id2", ...]}.

    Returns:
        Словарь {"deleted": int, "not_found": [...], "files_removed": int}.

    Raises:
        HTTPException: 400 — некорректный payload или пустой список;
            413 — слишком много ID за раз (>1000).
    """
    if not isinstance(payload, dict):
        raise HTTPException(400, "Ожидается JSON-объект")

    ids = payload.get("ids")
    if not isinstance(ids, list) or not ids:
        raise HTTPException(400, "Ожидается непустой массив 'ids'")

    if len(ids) > 1000:
        raise HTTPException(413, "За раз можно удалить не более 1000 задач")

    ids = [str(x) for x in ids if x]

    tq = get_task_queue()
    for jid in ids:
        tq.remove(jid)

    result = get_job_manager().delete_many(ids, remove_files=True)
    return result


@app.get("/api/queue", tags=["jobs"], summary="Состояние очереди")
async def queue_status() -> Dict[str, Any]:
    """
    Возвращает текущее состояние очереди задач.

    Returns:
        Словарь с полями size (задач ждёт), running (выполняется),
        max_parallel (лимит) и positions (job_id → позиция).
    """
    try:
        tq = get_task_queue()
    except RuntimeError:
        return {"size": 0, "running": 0, "max_parallel": 1, "positions": {}}
    return {
        "size": tq.size(),
        "running": tq.running_count(),
        "max_parallel": tq.get_max_parallel(),
        "positions": tq.positions(),
    }


def _sse(event: Dict) -> str:
    """
    Форматирует событие в SSE-сообщение.

    SSE-формат требует префикс "data: " и двойной перевод строки в
    конце. ensure_ascii=False сохраняет юникод в читаемом виде,
    что важно для русского текста в логах и сегментах.

    Args:
        event: Словарь события.

    Returns:
        Готовая строка для отправки в StreamingResponse.
    """
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get(
    "/api/jobs/{job_id}/stream",
    tags=["jobs"],
    summary="SSE-поток событий задачи",
    description="Первым сообщением идёт snapshot с полным состоянием. "
                "Для завершённых задач поток закрывается сразу. "
                "Для активных — транслируются live-события до done/error/cancelled.",
)
async def stream_job(job_id: str) -> StreamingResponse:
    """Открывает SSE-поток событий по задаче."""
    jm = get_job_manager()
    job = jm.get(job_id)

    if not job:
        snap = jm.get_snapshot(job_id)
        if not snap:
            raise HTTPException(404, "Задача не найдена")

        async def gen_finished():
            yield _sse({"type": "snapshot", "data": snap})
            final_type = "done" if snap["status"] == "done" else (
                "error" if snap["status"] in ("error", "interrupted") else "cancelled"
            )
            yield _sse({
                "type": final_type,
                "text": snap.get("text", ""),
                "metadata": snap.get("metadata", {}),
                "message": snap.get("error"),
            })

        return StreamingResponse(
            gen_finished(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    queue: asyncio.Queue = asyncio.Queue()
    job._subscribers.append(queue)
    snapshot = job.snapshot()

    async def gen():
        try:
            yield _sse({"type": "snapshot", "data": snapshot})
            while True:
                event = await queue.get()
                yield _sse(event)
                if event.get("type") in ("done", "error", "cancelled"):
                    break
        finally:
            if queue in job._subscribers:
                job._subscribers.remove(queue)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/jobs/{job_id}/result", tags=["jobs"], summary="Финальный результат")
async def get_result(job_id: str) -> Dict[str, Any]:
    """Возвращает финальный результат задачи."""
    snap = get_job_manager().get_snapshot(job_id)
    if not snap:
        raise HTTPException(404, "Задача не найдена")
    if snap["status"] != "done":
        raise HTTPException(409, f"Задача ещё не готова ({snap['status']})")
    return {
        "text": snap.get("text", ""),
        "segments": snap.get("segments", []),
        "metadata": snap.get("metadata", {}),
    }


@app.on_event("startup")
async def _startup() -> None:
    """
    Инициализирует репозиторий, очередь задач и фоновую очистку.

    Порядок важен: сначала помечаем прерванные задачи, потом создаём
    JobManager, потом инициализируем очередь — она вызывает
    get_job_manager() при первом же воркере.
    """
    env.upload_dir.mkdir(parents=True, exist_ok=True)
    env.models_dir.mkdir(parents=True, exist_ok=True)

    repo = get_repository()
    interrupted = repo.mark_interrupted()
    if interrupted:
        print(f"[startup] Помечено прерванных задач: {interrupted}")

    init_job_manager(repo=repo)

    jm = get_job_manager()
    max_parallel = int(store.get("server.max_parallel_jobs"))
    init_task_queue(
        get_job=lambda jid: jm.get(jid),
        run_job=transcribe.run,
        max_parallel=max_parallel,
    )

    # Логируем активные настройки при старте
    print("[startup] Активные настройки транскрибации:")
    for key, value in store.as_dict_for_worker().items():
        print(f"    {key} = {value!r}")
    print(f"    settings.json: {env.settings_file} "
          f"({'exists' if env.settings_file.exists() else 'empty'})")
    print(f"[startup] Очередь задач: параллелизм = {max_parallel}")

    threading.Thread(target=_cleanup_loop, daemon=True).start()


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
