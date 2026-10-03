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
from .jobs import job_manager
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
    """
    Фоновый цикл автоочистки завершённых задач и осиротевших файлов.

    Запускается в отдельном daemon-потоке при старте приложения. Период
    между итерациями берётся из настройки server.cleanup_interval_min
    и перечитывается на каждом цикле — то есть изменение в UI вступает
    в силу без перезапуска сервера.

    Осиротевшие файлы удаляются с двойным запасом по времени
    (retention_hours * 2): если задача была удалена, а её файл каким-то
    образом остался, он не будет лежать вечно, но и не будет удалён
    слишком рано, если что-то пошло не так.

    При ошибке в одной итерации (например, недоступен диск) цикл
    не завершается — sleep на 60 секунд и продолжение. Так фоновая
    задача не падает молча при временных сбоях.
    """
    while True:
        try:
            hours = int(store.get("server.retention_hours"))
            interval = int(store.get("server.cleanup_interval_min"))
            removed = job_manager.cleanup_expired(hours)
            orphans = job_manager.cleanup_orphan_files(env.upload_dir, max_age_hours=hours * 2)
            if removed or orphans:
                print(f"[cleanup] Удалено задач: {removed}, файлов: {orphans}")
            time.sleep(interval * 60)
        except Exception as e:
            print(f"[cleanup] ошибка: {e}")
            time.sleep(60)


@app.on_event("startup")
async def _startup() -> None:
    """
    Инициализация при старте приложения.

    Создаёт рабочие папки (если их нет) и запускает фоновый поток
    автоочистки. Daemon-флаг гарантирует, что поток завершится вместе
    с основным процессом, а не заблокирует выключение сервера.
    """
    env.upload_dir.mkdir(parents=True, exist_ok=True)
    env.models_dir.mkdir(parents=True, exist_ok=True)
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
    """
    Возвращает краткий список всех задач.

    Отсортирован от новых к старым. Каждая запись содержит только
    метаданные без логов и сегментов — это позволяет UI быстро
    отрисовать список истории, не выкачивая мегабайты текста.

    Returns:
        Словарь {"jobs": [...]} с summary() каждой задачи.
    """
    return {"jobs": [j.summary() for j in job_manager.list()]}


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

    job = job_manager.create(
        filename=file.filename,
        file_path="",
        file_size=0,
        model=model or store.get("transcription.model"),
        language=None if (language in (None, "auto", "")) else language,
    )
    job.set_loop(asyncio.get_running_loop())

    file_path = env.upload_dir / f"{job.id}{ext}"
    job.file_path = str(file_path)

    size = 0
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    f.close()
                    file_path.unlink(missing_ok=True)
                    job_manager.delete(job.id)
                    raise HTTPException(413, f"Файл превышает лимит {max_mb} МБ")
                f.write(chunk)
        job.file_size = size
    except HTTPException:
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        job_manager.delete(job.id)
        raise HTTPException(500, f"Ошибка сохранения: {e}")

    job.log("info", f"Задача {job.id} создана: {file.filename} ({size / 1024 / 1024:.1f} МБ)")

    threading.Thread(target=transcribe.run, args=(job,), daemon=True).start()

    return {"job_id": job.id, "status": "pending"}


@app.get("/api/jobs/{job_id}", tags=["jobs"], summary="Полный снимок задачи")
async def get_job(job_id: str) -> Dict[str, Any]:
    """
    Возвращает полный снимок состояния задачи.

    В отличие от /api/jobs (списка), здесь включены логи, сегменты
    и полный текст. Используется для восстановления UI без SSE,
    например после F5 страницы.

    Args:
        job_id: Идентификатор задачи.

    Returns:
        Полный snapshot задачи.

    Raises:
        HTTPException: 404, если задача не найдена.
    """
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    return job.snapshot()


@app.post("/api/jobs/{job_id}/cancel", tags=["jobs"], summary="Отменить задачу")
async def cancel_job(job_id: str) -> Dict[str, bool]:
    """
    Запрашивает отмену задачи.

    Отмена не мгновенная: воркер проверяет флаг между шагами и
    завершает работу при первой возможности. Ответ приходит сразу
    после установки флага, а фактическое завершение придёт через SSE
    событием type="cancelled".

    Args:
        job_id: Идентификатор задачи.

    Returns:
        Словарь {"ok": True}.

    Raises:
        HTTPException: 404, если задача не найдена.
    """
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    job.cancel()
    return {"ok": True}


@app.delete("/api/jobs/{job_id}", tags=["jobs"], summary="Удалить задачу")
async def delete_job(job_id: str) -> Dict[str, bool]:
    """
    Удаляет задачу вместе с её файлом.

    Если задача ещё выполняется, удаление не остановит воркер —
    он продолжит работать, но при попытке обратиться к задаче
    через API получит 404.

    Args:
        job_id: Идентификатор задачи.

    Returns:
        Словарь {"ok": True}.

    Raises:
        HTTPException: 404, если задача не найдена.
    """
    ok = job_manager.delete(job_id, remove_file=True)
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
    description="Первым сообщением всегда идёт `snapshot` с полным состоянием. "
                "Затем — live-события: `log`, `progress`, `status`, `segment`. "
                "Поток закрывается после `done`, `error` или `cancelled`.",
    response_class=StreamingResponse,
)
async def stream_job(job_id: str) -> StreamingResponse:
    """
    Открывает SSE-поток событий по задаче.

    Первым сообщением всегда идёт snapshot с полным текущим состоянием
    — это позволяет клиенту мгновенно восстановить UI после
    переподключения, не дожидаясь новых событий. Затем идёт поток
    live-событий: log, progress, status, segment.

    Если задача уже завершена к моменту подключения, поток закрывается
    сразу после отправки snapshot и одного финального события.

    Поток корректно отписывается от очереди в finally, даже если
    клиент оборвал соединение — иначе подписчики копились бы в памяти.

    Args:
        job_id: Идентификатор задачи.

    Returns:
        StreamingResponse с media_type="text/event-stream".

    Raises:
        HTTPException: 404, если задача не найдена.
    """
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")

    queue: asyncio.Queue = asyncio.Queue()
    job._subscribers.append(queue)
    snapshot = job.snapshot()

    async def gen():
        try:
            yield _sse({"type": "snapshot", "data": snapshot})
            if job.status in ("done", "error", "cancelled"):
                yield _sse({
                    "type": job.status if job.status != "done" else "done",
                    "text": job.text,
                    "metadata": job.metadata,
                    "message": job.error,
                })
                return
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


@app.get("/api/jobs/{job_id}/result", tags=["jobs"], summary="Финальный результат задачи")
async def get_result(job_id: str) -> Dict[str, Any]:
    """
    Возвращает финальный результат задачи в структурированном виде.

    Используется как альтернатива SSE для сценариев, где поток неудобен:
    скрипты, curl, экспорт через API.

    Args:
        job_id: Идентификатор задачи.

    Returns:
        Словарь {"text": ..., "segments": [...], "metadata": {...}}.

    Raises:
        HTTPException: 404, если задача не найдена;
            409, если задача ещё не завершена.
    """
    job = job_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    if job.status != "done":
        raise HTTPException(409, f"Задача ещё не готова ({job.status})")
    return {"text": job.text, "segments": job.segments, "metadata": job.metadata}


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
