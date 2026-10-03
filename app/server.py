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
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import RUNTIME_SCHEMA, get_env, get_store
from .i18n import available as locales_available, load as load_locale
from .config import get_repository
from .jobs import init_job_manager, get_job_manager
from . import monitor, transcribe

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
    Инициализирует репозиторий, помечает прерванные задачи и запускает
    фоновую очистку.
    """
    env.upload_dir.mkdir(parents=True, exist_ok=True)
    env.models_dir.mkdir(parents=True, exist_ok=True)

    repo = get_repository()
    interrupted = repo.mark_interrupted()
    if interrupted:
        print(f"[startup] Помечено прерванных задач: {interrupted}")

    init_job_manager(repo=repo)
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


@app.get("/api/jobs", tags=["jobs"], summary="Список всех задач")
async def list_jobs() -> Dict[str, Any]:
    """Возвращает краткий список всех задач, отсортированный от новых к старым."""
    return {"jobs": get_job_manager().list_summaries(limit=200)}


@app.post(
    "/api/jobs",
    tags=["jobs"],
    summary="Создать задачу транскрибации",
    description="Принимает файл (multipart/form-data) и параметры обработки. "
                "Возвращает ID задачи сразу; воркер запускается в фоне. "
                "Следите за прогрессом через `/api/jobs/{id}/stream`.",
    responses={
        200: {
            "description": "Задача создана",
            "content": {
                "application/json": {
                    "example": {"job_id": "a1b2c3d4e5f6", "status": "pending"}
                }
            },
        },
        400: {"description": "Неподдерживаемый формат файла"},
        413: {"description": "Файл превышает лимит из настроек"},
    },
)
async def create_job(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
) -> Dict[str, str]:
    """
    Создаёт новую задачу транскрибации и запускает воркер.

    Файл сохраняется на диск стримингом с проверкой размера на каждом
    чанке — это позволяет отклонить слишком большой файл до того, как
    он займёт всю оперативку или диск. Параметры model и language
    опциональны: если не переданы, используются значения из текущих
    runtime-настроек.

    Воркер запускается в daemon-потоке сразу, ещё до возврата ответа
    клиенту. Это значит, что SSE-подписка может подключиться уже к
    работающей задаче — snapshot через /stream вернёт актуальное
    состояние на момент подключения.

    Args:
        file: Загруженный файл (multipart/form-data).
        model: Название модели Whisper. None → значение из настроек.
        language: Код языка или "auto". None или "auto" → автоопределение.

    Returns:
        Словарь {"job_id": "...", "status": "pending"}.

    Raises:
        HTTPException: 400, если формат не поддерживается;
            413, если файл превышает лимит из настроек;
            500, если не удалось сохранить файл на диск.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in transcribe.ALL_SUPPORTED:
        raise HTTPException(400, f"Неподдерживаемый формат: {ext}")

    max_mb = int(store.get("server.max_upload_mb"))
    max_bytes = max_mb * 1024 * 1024

    jm = get_job_manager()
    job = jm.create(
        filename=file.filename,
        file_path="",
        file_size=0,
        model=model or store.get("transcription.model"),
        language=None if (language in (None, "auto", "")) else language,
    )
    job.set_loop(asyncio.get_running_loop())

    file_path = env.upload_dir / f"{job.id}{ext}"
    job.file_path = str(file_path)
    if jm._repo:
        jm._repo.update_job(job.id, file_path=str(file_path))

    size = 0
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    f.close()
                    file_path.unlink(missing_ok=True)
                    jm.delete(job.id)
                    raise HTTPException(413, f"Файл превышает лимит {max_mb} МБ")
                f.write(chunk)
        job.file_size = size
    except HTTPException:
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        jm.delete(job.id)
        raise HTTPException(500, f"Ошибка сохранения: {e}")

    job.log("info", f"Задача {job.id} создана: {file.filename} ({size / 1024 / 1024:.1f} МБ)")

    threading.Thread(target=transcribe.run, args=(job,), daemon=True).start()

    return {"job_id": job.id, "status": "pending"}


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


@app.get("/api/jobs/{job_id}", tags=["jobs"], summary="Полный снимок задачи")
async def get_job(job_id: str) -> Dict[str, Any]:
    """Возвращает полный снимок задачи из памяти или из БД."""
    snap = get_job_manager().get_snapshot(job_id)
    if not snap:
        raise HTTPException(404, "Задача не найдена")
    return snap


@app.post("/api/jobs/{job_id}/cancel", tags=["jobs"], summary="Отменить задачу")
async def cancel_job(job_id: str) -> Dict[str, bool]:
    """Запрашивает отмену задачи."""
    job = get_job_manager().get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    job.cancel()
    return {"ok": True}


@app.delete("/api/jobs/{job_id}", tags=["jobs"], summary="Удалить задачу")
async def delete_job(job_id: str) -> Dict[str, bool]:
    """Удаляет задачу вместе с её файлом."""
    ok = get_job_manager().delete(job_id, remove_file=True)
    if not ok:
        raise HTTPException(404, "Задача не найдена")
    return {"ok": True}


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


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
