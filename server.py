"""
Whisper Transcriber — Web Application
Все настройки читаются из .env (см. .env.example)
"""

import os
import sys
from pathlib import Path

# ===========================================================================
# 1. Конфигурация — ДО импорта ML-библиотек
# ===========================================================================

from config import get_settings, apply_env

settings = get_settings()
apply_env(settings)

# Создаём папки
settings.upload_dir_resolved.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# 2. 🔧 Фикс путей к CUDA DLL (Windows) — до ctranslate2
# ===========================================================================

def _register_cuda_dlls():
    if sys.platform != "win32":
        return
    try:
        import site
        candidates = [Path(sys.prefix) / "Lib" / "site-packages"]
        try:
            candidates.extend(Path(p) for p in site.getsitepackages())
        except Exception:
            pass

        subdirs = [
            ("nvidia", "cublas", "bin"),
            ("nvidia", "cuda_runtime", "bin"),
            ("nvidia", "cudnn", "bin"),
            ("nvidia", "cuda_nvrtc", "bin"),
            ("nvidia", "cufft", "bin"),
            ("nvidia", "curand", "bin"),
            ("nvidia", "cusolver", "bin"),
            ("nvidia", "cusparse", "bin"),
            ("nvidia", "nvjitlink", "bin"),
        ]
        added = set()
        for sp in candidates:
            if not sp.exists():
                continue
            for parts in subdirs:
                d = sp.joinpath(*parts)
                if d.exists() and str(d) not in added:
                    try:
                        os.add_dll_directory(str(d))
                        added.add(str(d))
                    except OSError:
                        pass
        try:
            import ctranslate2
            ct2 = Path(ctranslate2.__file__).parent
            if ct2.exists() and str(ct2) not in added:
                os.add_dll_directory(str(ct2))
                added.add(str(ct2))
        except Exception:
            pass
        if added:
            os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:
        print(f"[WARN] CUDA DLL registration failed: {e}", file=sys.stderr)

_register_cuda_dlls()

# ===========================================================================
# 3. Остальные импорты
# ===========================================================================

import asyncio
import json
import time
import uuid
import threading
import traceback
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import timedelta

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ===========================================================================
# 4. Константы
# ===========================================================================

SUPPORTED_AUDIO = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".wma", ".aac", ".opus"}
SUPPORTED_VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".ts"}
ALL_SUPPORTED = SUPPORTED_AUDIO | SUPPORTED_VIDEO

MODEL_REPOS = {
    "tiny":           "Systran/faster-whisper-tiny",
    "base":           "Systran/faster-whisper-base",
    "small":          "Systran/faster-whisper-small",
    "medium":         "Systran/faster-whisper-medium",
    "large-v2":       "Systran/faster-whisper-large-v2",
    "large-v3":       "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}

LANGUAGES = {
    "auto": "Автоопределение",
    "ru": "Русский",
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "uk": "Українська",
}

# ===========================================================================
# 5. Job — состояние задачи
# ===========================================================================

@dataclass
class Job:
    id: str
    filename: str
    file_path: str
    model: str
    language: Optional[str]
    batch_size: int = 8
    vad_filter: bool = True
    initial_prompt: str = ""
    hotwords: str = ""

    status: str = "pending"
    stage: str = "Ожидание"
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

    def set_loop(self, loop):
        self._loop = loop

    def emit(self, event: Dict):
        if self._loop is None:
            return
        for q in list(self._subscribers):
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:
                pass

    def log(self, level: str, message: str):
        entry = {"time": time.time(), "level": level, "message": message}
        self.logs.append(entry)
        self.emit({"type": "log", "entry": entry})

    def set_progress(self, value: int, message: str = "", stage: str = None):
        self.progress = max(0, min(100, value))
        if message:
            self.message = message
        if stage:
            self.stage = stage
        self.emit({
            "type": "progress",
            "value": self.progress,
            "message": self.message,
            "stage": self.stage,
        })

    def set_status(self, status: str, message: str = ""):
        self.status = status
        if message:
            self.message = message
        self.emit({"type": "status", "status": self.status, "message": self.message})

    def add_segment(self, start: float, end: float, text: str):
        seg = {"index": len(self.segments), "start": start, "end": end, "text": text}
        self.segments.append(seg)
        self.emit({"type": "segment", "segment": seg})

    def snapshot(self) -> Dict:
        return {
            "id": self.id,
            "filename": self.filename,
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

    def cancel(self):
        self._cancelled = True


jobs: Dict[str, Job] = {}
jobs_lock = threading.Lock()


# ===========================================================================
# 6. Скачивание моделей
# ===========================================================================

def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def ensure_model_downloaded(job: Job, model_size: str) -> str:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    if model_size not in MODEL_REPOS:
        raise ValueError(f"Неизвестная модель: {model_size}")

    repo_id = MODEL_REPOS[model_size]
    org, repo = repo_id.split("/", 1)
    local_dir = settings.models_dir / org / repo
    local_dir.mkdir(parents=True, exist_ok=True)

    model_file = local_dir / "model.bin"
    required = ["model.bin", "config.json", "tokenizer.json"]

    if all((local_dir / f).exists() for f in required):
        size_gb = model_file.stat().st_size / (1024 ** 3)
        job.log("success", f"Модель найдена в кэше: {local_dir.name} ({size_gb:.2f} ГБ)")
        return str(local_dir)

    job.log("info", f"Скачивание {repo_id} → {local_dir}")

    try:
        from tqdm.auto import tqdm as _tqdm_base
    except ImportError:
        _tqdm_base = object

    class _ProgressTqdm(_tqdm_base):
        _last_pct = -1

        def update(self, n=1):
            try:
                super().update(n)
            except Exception:
                pass
            if getattr(self, "total", None):
                pct = int(self.n / self.total * 100)
                if pct != self._last_pct:
                    self._last_pct = pct
                    job.set_progress(
                        2 + int(pct * 0.28),
                        f"Скачивание: {_fmt_bytes(self.n)} / {_fmt_bytes(self.total)}",
                        "downloading",
                    )

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=settings.hf_token or None,
            allow_patterns=["*.bin", "*.json", "*.txt"],
            ignore_patterns=["*.md", ".gitattributes", "*.py"],
            max_workers=settings.max_workers,
            tqdm_class=_ProgressTqdm,
        )
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", 0)
        if code in (401, 403):
            raise RuntimeError(f"HF Hub отклонил токен (код {code}). Проверьте HF_TOKEN в .env")
        raise
    except Exception as e:
        raise RuntimeError(f"Ошибка скачивания модели: {e}") from e

    if not model_file.exists():
        raise RuntimeError(f"Не найден {model_file} после скачивания.")

    size_gb = model_file.stat().st_size / (1024 ** 3)
    job.log("success", f"Модель скачана: {size_gb:.2f} ГБ")
    return str(local_dir)


# ===========================================================================
# 7. Воркер транскрибации
# ===========================================================================

def _format_ts(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _extract_audio(job: Job, video_path: str) -> str:
    from moviepy import VideoFileClip

    temp_dir = tempfile.mkdtemp(prefix="whisper_")
    audio_path = os.path.join(temp_dir, "extracted_audio.wav")

    job.log("info", "Извлечение аудиодорожки...")
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        raise ValueError("В видеофайле отсутствует аудиодорожка.")

    clip.audio.write_audiofile(
        audio_path,
        codec="pcm_s16le",
        fps=16000,
        nbytes=2,
        ffmpeg_params=["-ac", "1"],
        logger=None,
    )
    clip.close()
    job.log("success", "Аудио извлечено: 16 кГц, моно, PCM")
    return audio_path


def run_transcription(job: Job):
    temp_audio = None
    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline

        audio_path = job.file_path
        ext = Path(job.file_path).suffix.lower()

        if ext in SUPPORTED_VIDEO:
            job.set_status("downloading", "Извлечение аудио...")
            temp_audio = _extract_audio(job, job.file_path)
            audio_path = temp_audio

        if job._cancelled:
            raise InterruptedError()

        job.set_status("downloading", "Подготовка модели...")
        job.set_progress(2, "Проверка кэша модели...", "downloading")
        job.log("info", f"Модель: {job.model}")

        model_path = ensure_model_downloaded(job, job.model)

        if job._cancelled:
            raise InterruptedError()

        job.set_status("loading", "Загрузка модели в GPU...")
        job.set_progress(32, "Загрузка в GPU...", "loading")
        job.log("info", f"Загрузка модели в {settings.default_device.upper()}...")

        t_load = time.time()
        model = WhisperModel(
            model_path,
            device=settings.default_device,
            compute_type=settings.default_compute_type,
        )
        batched = BatchedInferencePipeline(model=model)
        job.log("success", f"Модель загружена за {time.time() - t_load:.1f} с")

        if job._cancelled:
            raise InterruptedError()

        job.set_status("transcribing", "Транскрибация...")
        job.set_progress(35, "Транскрибация...", "transcribing")
        job.log("info", f"Параметры: batch_size={job.batch_size}, vad={job.vad_filter}")

        kwargs = {
            "batch_size": job.batch_size,
            "beam_size": 5,
            "vad_filter": job.vad_filter,
            "vad_parameters": dict(min_silence_duration_ms=500, speech_pad_ms=200),
            "word_timestamps": True,
        }
        if job.language:
            kwargs["language"] = job.language
        if job.initial_prompt:
            kwargs["initial_prompt"] = job.initial_prompt
        if job.hotwords:
            kwargs["hotwords"] = job.hotwords

        t_start = time.time()
        segments, info = batched.transcribe(audio_path, **kwargs)

        total_duration = getattr(info, "duration", 0) or 0
        job.log("info", f"Язык: {info.language} (уверенность {info.language_probability:.0%})")
        job.log("info", f"Длительность аудио: {_format_ts(total_duration)}")

        text_parts = []
        for i, seg in enumerate(segments):
            if job._cancelled:
                raise InterruptedError()

            text = seg.text.strip()
            if not text:
                continue

            job.add_segment(seg.start, seg.end, text)
            text_parts.append(text)

            if total_duration > 0 and seg.end > 0:
                pct = 35 + int((seg.end / total_duration) * 64)
            else:
                pct = min(35 + i, 99)
            pct = min(pct, 99)

            job.set_progress(
                pct,
                f"[{_format_ts(seg.end)} / {_format_ts(total_duration)}] {text[:60]}...",
                "transcribing",
            )

        job.text = "\n".join(text_parts)
        job.metadata = {
            "language": getattr(info, "language", "?"),
            "language_probability": getattr(info, "language_probability", 0),
            "duration": total_duration,
            "segments_count": len(job.segments),
            "model": job.model,
            "processing_time": time.time() - t_start,
        }
        job.set_progress(100, "Готово!", "done")
        job.set_status("done", "Транскрибация завершена")
        job.finished_at = time.time()
        job.log("success", f"Готово за {time.time() - t_start:.1f} с. Символов: {len(job.text):,}")
        job.emit({"type": "done", "text": job.text, "metadata": job.metadata})

    except InterruptedError:
        job.set_status("cancelled", "Отменено пользователем")
        job.log("warn", "Транскрибация отменена")
        job.emit({"type": "cancelled"})

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        job.error = err
        job.set_status("error", err)
        job.log("error", err)
        job.log("error", traceback.format_exc())
        job.emit({"type": "error", "message": err})

    finally:
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
                parent = os.path.dirname(temp_audio)
                if os.path.isdir(parent) and parent.startswith(tempfile.gettempdir()):
                    try:
                        os.rmdir(parent)
                    except OSError:
                        pass
            except OSError:
                pass


# ===========================================================================
# 8. FastAPI
# ===========================================================================

app = FastAPI(title="Whisper Transcriber", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# 9. API
# ===========================================================================

@app.get("/api/system")
async def system_info():
    info = {"cuda": False, "gpu": None, "vram": None, "torch": False}
    try:
        import torch
        info["torch"] = True
        if torch.cuda.is_available():
            info["cuda"] = True
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1
            )
    except ImportError:
        pass
    return info


@app.get("/api/models")
async def list_models():
    return {
        "models": list(MODEL_REPOS.keys()),
        "default": settings.default_model,
        "languages": LANGUAGES,
        "default_language": settings.default_language,
        "default_batch_size": settings.default_batch_size,
        "default_vad": settings.default_vad,
    }


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    model: str = Form(None),
    language: str = Form(None),
    batch_size: int = Form(None),
    vad_filter: bool = Form(None),
    initial_prompt: str = Form(""),
    hotwords: str = Form(""),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALL_SUPPORTED:
        raise HTTPException(400, f"Неподдерживаемый формат: {ext}")

    # Применяем значения по умолчанию из .env, если не переданы
    model = model or settings.default_model
    language = language or settings.default_language
    batch_size = batch_size if batch_size is not None else settings.default_batch_size
    vad_filter = vad_filter if vad_filter is not None else settings.default_vad

    if model not in MODEL_REPOS:
        raise HTTPException(400, f"Неизвестная модель: {model}")

    job_id = uuid.uuid4().hex[:12]
    file_path = settings.upload_dir_resolved / f"{job_id}{ext}"

    # Стриминговая запись + проверка размера
    size = 0
    max_bytes = settings.max_upload_bytes
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    f.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        413,
                        f"Файл превышает лимит {settings.max_upload_mb} МБ",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Ошибка сохранения файла: {e}")

    job = Job(
        id=job_id,
        filename=file.filename,
        file_path=str(file_path),
        model=model,
        language=None if language == "auto" else language,
        batch_size=batch_size,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )

    with jobs_lock:
        jobs[job_id] = job

    loop = asyncio.get_running_loop()
    job.set_loop(loop)

    threading.Thread(target=run_transcription, args=(job,), daemon=True).start()

    job.log("info", f"Задача {job_id} создана: {file.filename} ({size / 1024 / 1024:.1f} МБ)")

    return {"job_id": job_id, "status": "pending"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    return job.snapshot()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    job.cancel()
    return {"ok": True}


def _sse(event: Dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    job = jobs.get(job_id)
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
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/jobs/{job_id}/result")
async def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Задача не найдена")
    if job.status != "done":
        raise HTTPException(409, f"Задача ещё не готова (статус: {job.status})")
    return {
        "text": job.text,
        "segments": job.segments,
        "metadata": job.metadata,
    }


# ===========================================================================
# 10. Статика
# ===========================================================================

app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
