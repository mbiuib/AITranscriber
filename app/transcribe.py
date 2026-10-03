"""
Воркер транскрибации.

Содержит всю логику обработки одного файла: извлечение аудио из видео,
скачивание модели с HuggingFace Hub, запуск faster-whisper на GPU/CPU,
потоковая выдача сегментов в Job по мере обработки.

Модуль вызывается из server.py в daemon-потоке при создании задачи.
Все побочные эффекты (прогресс, логи, сегменты) публикуются через методы
Job, а не возвращаются из run() — это позволяет UI получать обновления
в реальном времени через SSE.

Поддерживаемые форматы и список моделей заданы на уровне модуля, чтобы
server.py мог валидировать вход до запуска воркера.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

from .config import get_env, get_store
from .jobs import Job
from . import media_info

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


def _fmt_bytes(n: float) -> str:
    """
    Форматирует размер в человекочитаемую строку.

    Args:
        n: Размер в байтах.

    Returns:
        Строка вида "3.1 GB", "512.3 MB".
    """
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_ts(seconds: float) -> str:
    """
    Форматирует время в формат ЧЧ:ММ:СС.

    Args:
        seconds: Время в секундах.

    Returns:
        Строка вида "01:23:45".
    """
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _ensure_model(job: Job, model: str) -> str:
    """
    Гарантирует наличие модели на диске и возвращает путь к ней.

    Если файлы модели уже есть в локальной папке, скачивание пропускается.
    Иначе выполняется snapshot_download с HuggingFace Hub с прогрессом,
    который транслируется в job через кастомный tqdm-класс.

    Структура хранения — LM Studio-подобная:
        <models_dir>/<org>/<repo>/

    Прогресс скачивания отображается в диапазоне 2–30% общего прогресса
    задачи. Это оставляет место для этапов извлечения аудио (0–2%),
    загрузки модели в VRAM (32–35%) и самой транскрибации (35–100%).

    Args:
        job: Задача, в которую публикуются логи и прогресс.
        model: Название модели из MODEL_REPOS (например, "large-v3").

    Returns:
        Абсолютный путь к папке с файлами модели.

    Raises:
        ValueError: Если указана неизвестная модель.
        RuntimeError: Если HF Hub отклонил токен или скачивание не удалось.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    env = get_env()
    if model not in MODEL_REPOS:
        raise ValueError(f"Неизвестная модель: {model}")

    repo_id = MODEL_REPOS[model]
    org, repo = repo_id.split("/", 1)
    local_dir = env.models_dir / org / repo
    local_dir.mkdir(parents=True, exist_ok=True)

    model_file = local_dir / "model.bin"
    required = ["model.bin", "config.json", "tokenizer.json"]

    if all((local_dir / f).exists() for f in required):
        size_gb = model_file.stat().st_size / (1024 ** 3)
        job.log("success", f"Модель в кэше: {local_dir.name} ({size_gb:.2f} ГБ)")
        return str(local_dir)

    job.log("info", f"Скачивание {repo_id} → {local_dir}")

    try:
        from tqdm.auto import tqdm as _tqdm_base
    except ImportError:
        _tqdm_base = object

    class _Tqdm(_tqdm_base):
        """
        Обёртка над tqdm, транслирующая прогресс скачивания в Job.

        huggingface_hub создаёт tqdm-бары для каждого файла; мы
        перехватываем update() и публикуем процент через job.set_progress.
        Дедупликация по _last нужна, чтобы не спамить событиями SSE —
        tqdm вызывает update() очень часто.
        """
        _last = -1

        def update(self, n=1):
            try:
                super().update(n)
            except Exception:
                pass
            if getattr(self, "total", None):
                pct = int(self.n / self.total * 100)
                if pct != self._last:
                    self._last = pct
                    job.set_progress(
                        2 + int(pct * 0.28),
                        f"Скачивание: {_fmt_bytes(self.n)} / {_fmt_bytes(self.total)}",
                        "downloading",
                    )

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=env.hf_token or None,
            allow_patterns=["*.bin", "*.json", "*.txt"],
            ignore_patterns=["*.md", ".gitattributes", "*.py"],
            max_workers=env.max_workers,
            tqdm_class=_Tqdm,
        )
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", 0)
        if code in (401, 403):
            raise RuntimeError(f"HF Hub отклонил токен (код {code}). Проверьте HF_TOKEN в .env")
        raise
    except Exception as e:
        raise RuntimeError(f"Ошибка скачивания модели: {e}") from e

    size_gb = model_file.stat().st_size / (1024 ** 3)
    job.log("success", f"Модель скачана: {size_gb:.2f} ГБ")
    return str(local_dir)


def _extract_audio(job: Job, video_path: str) -> str:
    """
    Извлекает аудиодорожку из видео во временный WAV-файл.

    Параметры (16 кГц, моно, PCM s16le) выбраны под требования Whisper:
    модель обучена именно на таких данных, и подготовка входа заранее
    через FFmpeg быстрее, чем пересэмплирование внутри модели.

    Временный файл создаётся в системной temp-папке и удаляется
    вызывающей стороной в блоке finally. Возвращаемый путь ведёт на
    файл audio.wav внутри уникальной подпапки с префиксом "whisper_".

    Args:
        job: Задача, в которую публикуются логи.
        video_path: Путь к исходному видеофайлу.

    Returns:
        Путь к извлечённому WAV-файлу.

    Raises:
        ValueError: Если в видео отсутствует аудиодорожка.
    """
    from moviepy import VideoFileClip
    temp_dir = tempfile.mkdtemp(prefix="whisper_")
    audio_path = os.path.join(temp_dir, "audio.wav")
    job.log("info", "Извлечение аудиодорожки…")
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        raise ValueError("В видео нет аудиодорожки.")
    clip.audio.write_audiofile(
        audio_path, codec="pcm_s16le", fps=16000, nbytes=2,
        ffmpeg_params=["-ac", "1"], logger=None,
    )
    clip.close()
    job.log("success", "Аудио извлечено: 16 кГц, моно")
    return audio_path


def run(job: Job) -> None:
    """
    Выполняет полный цикл транскрибации задачи.

    Собирает метаданные исходного файла и статистику обработки: время
    каждого этапа, пик VRAM, RTF. Всё это сохраняется в БД по завершении
    (или при ошибке/отмене) и доступно в детальном просмотре задачи.

    Args:
        job: Задача со всеми параметрами обработки.
    """
    temp_audio = None
    stats: Dict[str, Any] = {
        "started_at": time.time(),
        "stages": {},
        "peak_vram_mb": 0,
    }
    stage_start = time.time()

    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline

        settings = get_store().as_dict_for_worker()
        model_name = job.model or settings["model"]
        language = job.language or (None if settings["language"] == "auto" else settings["language"])

        # Сохраняем снимок настроек, использованных для этой задачи
        settings_snapshot = {
            "model": model_name,
            "language": language or "auto",
            "batch_size": settings["batch_size"],
            "beam_size": settings["beam_size"],
            "vad": settings["vad"],
            "condition_on_previous": settings["condition_on_previous"],
            "device": settings["device"],
            "compute_type": settings["compute_type"],
            "initial_prompt": settings["initial_prompt"],
            "hotwords": settings["hotwords"],
        }
        if job._repo:
            try:
                job._repo.update_job(
                    job.id,
                    settings_snapshot=json.dumps(settings_snapshot, ensure_ascii=False),
                )
            except Exception:
                pass

        # Метаданные файла
        job.log("info", "Чтение метаданных файла…")
        src_meta = media_info.probe(job.file_path)
        if src_meta:
            if src_meta.get("duration"):
                job.log("info", f"Длительность источника: {_fmt_ts(src_meta['duration'])}")
            if src_meta.get("format_name"):
                job.log("info", f"Формат: {src_meta['format_name']}")
            if job._repo:
                try:
                    job._repo.update_job(
                        job.id,
                        source_metadata=json.dumps(src_meta, ensure_ascii=False),
                    )
                except Exception:
                    pass
        stage_start = time.time()

        audio_path = job.file_path
        if Path(job.file_path).suffix.lower() in SUPPORTED_VIDEO:
            job.set_status("downloading", "Извлечение аудио…")
            temp_audio = _extract_audio(job, job.file_path)
            audio_path = temp_audio
        stats["stages"]["extract_audio"] = round(time.time() - stage_start, 2)

        if job.is_cancelled():
            raise InterruptedError()

        stage_start = time.time()
        job.set_status("downloading", "Подготовка модели…")
        job.set_progress(2, "Проверка кэша…", "downloading")
        job.log("info", f"Модель: {model_name}")
        model_path = _ensure_model(job, model_name)
        stats["stages"]["download_model"] = round(time.time() - stage_start, 2)

        if job.is_cancelled():
            raise InterruptedError()

        stage_start = time.time()
        job.set_status("loading", "Загрузка модели…")
        job.set_progress(32, "Загрузка в GPU…", "loading")
        device = settings["device"]
        compute = settings["compute_type"]
        job.log("info", f"Устройство: {device}, точность: {compute}")

        t0 = time.time()
        model = WhisperModel(model_path, device=device, compute_type=compute)
        batched = BatchedInferencePipeline(model=model)
        job.log("success", f"Модель загружена за {time.time() - t0:.1f} с")
        stats["stages"]["load_model"] = round(time.time() - stage_start, 2)

        if job.is_cancelled():
            raise InterruptedError()

        job.set_status("transcribing", "Транскрибация…")
        job.set_progress(35, "Транскрибация…", "transcribing")

        kwargs = {
            "batch_size": int(settings["batch_size"]),
            "beam_size": int(settings["beam_size"]),
            "vad_filter": bool(settings["vad"]),
            "vad_parameters": dict(min_silence_duration_ms=500, speech_pad_ms=200),
            "word_timestamps": True,
            "condition_on_previous_text": bool(settings["condition_on_previous"]),
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        }
        if language:
            kwargs["language"] = language
        if settings["initial_prompt"]:
            kwargs["initial_prompt"] = settings["initial_prompt"]
        if settings["hotwords"]:
            kwargs["hotwords"] = settings["hotwords"]

        job.log("info", f"Параметры: batch={kwargs['batch_size']}, beam={kwargs['beam_size']}, "
                        f"vad={kwargs['vad_filter']}, lang={language or 'auto'}")

        t_start = time.time()
        segments, info = batched.transcribe(audio_path, **kwargs)

        total_duration = getattr(info, "duration", 0) or 0
        job.log("info", f"Язык: {info.language} ({info.language_probability:.0%})")
        job.log("info", f"Длительность: {_fmt_ts(total_duration)}")

        text_parts = []
        peak_vram = 0
        for i, seg in enumerate(segments):
            if job.is_cancelled():
                raise InterruptedError()
            text = seg.text.strip()
            if not text:
                continue
            job.add_segment(seg.start, seg.end, text)
            text_parts.append(text)

            if i % 20 == 0:
                peak_vram = max(peak_vram, _get_gpu_used_mb())

            if total_duration > 0 and seg.end > 0:
                pct = 35 + int((seg.end / total_duration) * 64)
            else:
                pct = min(35 + i, 99)
            job.set_progress(
                min(pct, 99),
                f"[{_fmt_ts(seg.end)} / {_fmt_ts(total_duration)}] {text[:60]}…",
                "transcribing",
            )

        proc_time = time.time() - t_start
        stats["stages"]["transcribe"] = round(proc_time, 2)
        stats["peak_vram_mb"] = peak_vram
        stats["total_seconds"] = round(time.time() - stats["started_at"], 2)
        stats["rtf"] = round(proc_time / max(total_duration, 0.01), 3)

        job.text = "\n".join(text_parts)
        job.metadata = {
            "language": getattr(info, "language", "?"),
            "language_probability": getattr(info, "language_probability", 0),
            "duration": total_duration,
            "segments_count": len(job.segments),
            "model": model_name,
            "processing_time": proc_time,
            "speed_factor": round(total_duration / max(proc_time, 0.01), 2),
        }
        if job._repo:
            try:
                job._repo.update_job(
                    job.id,
                    processing_stats=json.dumps(stats, ensure_ascii=False),
                )
            except Exception:
                pass

        job.set_progress(100, "Готово!", "done")
        job.set_status("done", "Транскрибация завершена")
        job.finished_at = time.time()
        job.log("success", f"Готово за {proc_time:.1f} с. Символов: {len(job.text):,}")
        job.persist_final()
        job.emit({"type": "done", "text": job.text, "metadata": job.metadata})

    except InterruptedError:
        stats["total_seconds"] = round(time.time() - stats["started_at"], 2)
        if job._repo:
            try:
                job._repo.update_job(
                    job.id,
                    processing_stats=json.dumps(stats, ensure_ascii=False),
                )
            except Exception:
                pass
        job.finished_at = time.time()
        job.set_status("cancelled", "Отменено")
        job.log("warn", "Транскрибация отменена")
        job.persist_final()
        job.emit({"type": "cancelled"})
    except Exception as e:
        stats["total_seconds"] = round(time.time() - stats["started_at"], 2)
        if job._repo:
            try:
                job._repo.update_job(
                    job.id,
                    processing_stats=json.dumps(stats, ensure_ascii=False),
                )
            except Exception:
                pass
        err = f"{type(e).__name__}: {e}"
        job.error = err
        job.finished_at = time.time()
        job.set_status("error", err)
        job.log("error", err)
        job.log("error", traceback.format_exc())
        job.persist_final()
        job.emit({"type": "error", "message": err})
    finally:
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
                parent = os.path.dirname(temp_audio)
                if parent.startswith(tempfile.gettempdir()):
                    try:
                        os.rmdir(parent)
                    except OSError:
                        pass
            except OSError:
                pass


def _get_gpu_used_mb() -> int:
    """
    Возвращает текущее использование VRAM в мегабайтах.

    Returns:
        Использовано МБ или 0, если GPU недоступна.
    """
    try:
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return round(mem.used / (1024 ** 2))
    except Exception:
        return 0
