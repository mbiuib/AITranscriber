"""
Воркер транскрибации на MOSS-Transcribe-Diarize.

Альтернатива связке Whisper + Pyannote. Модель делает транскрибацию
и диаризацию за один проход, выдавая размеченный текст в формате
[start][Sxx]text[end].

Устанавливается отдельно из GitHub репозитория OpenMOSS:
    git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git
    cd MOSS-Transcribe-Diarize && pip install -e .

Модель скачивается с HuggingFace Hub при первом запуске (~1.8 ГБ)
в <models_dir>/<org>/<repo>/. Требует trust_remote_code=True — код
модели исполняется локально.

Длинные аудио обрабатываются чанками: MOSS держит весь вход в памяти
за один проход, и KV-кэш растёт линейно с длительностью. Для 16 ГБ
VRAM безопасный размер чанка — 5 минут. Чанки перекрываются на 2
секунды, дубликаты на границах вырезаются после склейки.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import get_env, get_store
from .jobs import Job
from . import media_info

SUPPORTED_AUDIO = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".wma", ".aac", ".opus"}
SUPPORTED_VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".ts"}
ALL_SUPPORTED = SUPPORTED_AUDIO | SUPPORTED_VIDEO

MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
MODEL_REVISION = "main"

# Размер чанка в секундах. 300 (5 минут) — безопасно для 16 ГБ VRAM.
# Для больших GPU можно увеличить до 600 (10 минут).
CHUNK_DURATION_SEC = 300.0
CHUNK_OVERLAP_SEC = 2.0

# Ограничение на длину выходного текста за один проход.
# 4096 токенов ≈ 5 минут транскрибации, достаточно для одного чанка.
MAX_NEW_TOKENS = 4096

_model = None
_processor = None
_model_lock = None


def is_available() -> bool:
    """
    Проверяет, установлен ли пакет moss_transcribe_diarize.

    Returns:
        True, если пакет импортируется без ошибок.
    """
    try:
        from moss_transcribe_diarize import parse_transcript  # noqa: F401
        return True
    except ImportError:
        return False


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


def _normalize_speaker(raw: str) -> str:
    """
    Преобразует метку спикера MOSS в формат SPEAKER_XX.

    MOSS выдаёт метки вида [S01], [S02]. Внутренний формат проекта —
    SPEAKER_00, SPEAKER_01, что ожидает UI и format_speaker.

    Args:
        raw: Метка из вывода модели, например "S01".

    Returns:
        Нормализованная метка "SPEAKER_00".
    """
    m = re.search(r"S(\d+)", raw)
    if m:
        idx = int(m.group(1)) - 1
        return f"SPEAKER_{idx:02d}"
    return "SPEAKER_00"


def _find_model_weights(local_dir: Path) -> Optional[Path]:
    """
    Ищет файл весов модели в папке.

    Поддерживает три варианта, которые встречаются в HF-репозиториях:
    единый model.safetensors, шардированные model-XXXXX-of-XXXXX.safetensors
    и старый формат pytorch_model.bin.

    Args:
        local_dir: Папка с моделью.

    Returns:
        Путь к первому найденному файлу весов или None.
    """
    patterns = [
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
    ]
    for pattern in patterns:
        for path in sorted(local_dir.glob(pattern)):
            if path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
                return path
    return None


def _total_weights_size(local_dir: Path) -> int:
    """
    Считает суммарный размер всех файлов весов в папке.

    Для шардированной модели учитывает все шарды. Возвращает байты.

    Args:
        local_dir: Папка с моделью.

    Returns:
        Суммарный размер в байтах или 0.
    """
    total = 0
    for pattern in ("*.safetensors", "*.bin"):
        for path in local_dir.glob(pattern):
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


def _get_audio_duration(audio_path: str) -> float:
    """
    Возвращает длительность аудио в секундах через ffprobe.

    Args:
        audio_path: Путь к аудиофайлу.

    Returns:
        Длительность в секундах или 0 при ошибке.
    """
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return float(out)
    except Exception:
        return 0.0


def _extract_audio(job: Job, video_path: str) -> str:
    """
    Извлекает аудиодорожку из видео во временный WAV-файл 16 кГц моно.

    Args:
        job: Задача для логирования.
        video_path: Путь к исходному видеофайлу.

    Returns:
        Путь к извлечённому WAV-файлу.

    Raises:
        ValueError: Если в видео нет аудиодорожки.
    """
    from moviepy import VideoFileClip
    temp_dir = tempfile.mkdtemp(prefix="moss_")
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


def _split_audio(
    audio_path: str,
    chunk_sec: float = CHUNK_DURATION_SEC,
    overlap_sec: float = CHUNK_OVERLAP_SEC,
) -> List[Tuple[str, float]]:
    """
    Разбивает аудио на чанки через ffmpeg.

    Возвращает список кортежей (путь_к_чанку, смещение_в_секундах).
    Чанки создаются во временной папке, удаляются вызывающей стороной.

    Args:
        audio_path: Путь к исходному аудио.
        chunk_sec: Длительность одного чанка в секундах.
        overlap_sec: Перехлёст между чанками в секундах.

    Returns:
        Список (путь, смещение) для каждого чанка.
    """
    duration = _get_audio_duration(audio_path)
    if duration <= 0 or duration <= chunk_sec:
        return [(audio_path, 0.0)]

    temp_dir = tempfile.mkdtemp(prefix="moss_chunks_")
    chunks: List[Tuple[str, float]] = []
    step = chunk_sec - overlap_sec
    offset = 0.0
    idx = 0

    while offset < duration:
        chunk_path = os.path.join(temp_dir, f"chunk_{idx:03d}.wav")
        actual_len = min(chunk_sec + overlap_sec, duration - offset)

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "quiet",
                    "-ss", str(offset),
                    "-t", str(actual_len),
                    "-i", audio_path,
                    "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le",
                    chunk_path,
                ],
                check=True,
                timeout=180,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg не смог нарезать чанк {idx}: {e}") from e

        chunks.append((chunk_path, offset))
        offset += step
        idx += 1

    return chunks


def _dedupe_overlap(
    segments: List[Dict[str, Any]],
    text_similarity: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    Убирает дубликаты сегментов, появившиеся из-за перехлёста чанков.

    Если два соседних сегмента имеют близкие временные метки и похожий
    текст (по Jaccard-схожести слов), оставляется один — более длинный.

    Args:
        segments: Список сегментов с полями start, end, text, speaker.
        text_similarity: Порог схожести текста (0-1).

    Returns:
        Список без дубликатов.
    """
    if len(segments) <= 1:
        return segments

    sorted_segs = sorted(segments, key=lambda s: (s["start"], s["end"]))
    result: List[Dict[str, Any]] = [sorted_segs[0]]

    for seg in sorted_segs[1:]:
        prev = result[-1]

        time_overlap = (
            seg["start"] < prev["end"] - 0.5
            and seg["end"] > prev["start"] + 0.5
        )

        if time_overlap:
            a = set(prev["text"].lower().split())
            b = set(seg["text"].lower().split())
            if a and b:
                jaccard = len(a & b) / len(a | b)
                if jaccard >= text_similarity:
                    if len(seg["text"]) > len(prev["text"]):
                        result[-1] = seg
                    continue
            else:
                if len(seg["text"]) <= len(prev["text"]):
                    continue

        result.append(seg)

    return result


def _cleanup_chunks(chunks: List[Tuple[str, float]], original: str) -> None:
    """
    Удаляет временные файлы чанков.

    Не трогает исходный файл audio_path, даже если он в списке.

    Args:
        chunks: Список (путь, смещение).
        original: Путь к исходному аудио, который нужно сохранить.
    """
    dirs_to_clean: set = set()
    for path, _ in chunks:
        if path == original:
            continue
        if os.path.exists(path):
            try:
                os.remove(path)
                dirs_to_clean.add(os.path.dirname(path))
            except OSError:
                pass

    for d in dirs_to_clean:
        if d.startswith(tempfile.gettempdir()):
            try:
                os.rmdir(d)
            except OSError:
                pass


def _ensure_model_downloaded(job: Job) -> str:
    """
    Скачивает модель MOSS в локальную папку, если её там ещё нет.

    Использует snapshot_download с кастомным tqdm-классом, который
    транслирует прогресс скачивания в job.set_progress. Файлы кладутся
    в <models_dir>/<org>/<repo>/.

    Поддерживает как единый файл весов, так и шардированные модели
    (несколько *.safetensors с индексом).

    Args:
        job: Задача для логирования и публикации прогресса.

    Returns:
        Путь к локальной папке с моделью.

    Raises:
        RuntimeError: Если скачивание не удалось или файлы весов
            не найдены после завершения.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    env = get_env()
    org, repo = MODEL_ID.split("/", 1)
    local_dir = env.models_dir / org / repo
    local_dir.mkdir(parents=True, exist_ok=True)

    weights = _find_model_weights(local_dir)
    config = local_dir / "config.json"
    jinja = local_dir / "chat_template.jinja"

    if weights is not None and config.exists() and jinja.exists():
        size_gb = _total_weights_size(local_dir) / (1024 ** 3)
        job.log("success",
                f"Модель MOSS в кэше: {local_dir.name} ({size_gb:.2f} ГБ)")
        return str(local_dir)

    job.log("info", f"Скачивание MOSS {MODEL_ID} → {local_dir}")

    try:
        from tqdm.auto import tqdm as _tqdm_base
    except ImportError:
        _tqdm_base = object

    class _Tqdm(_tqdm_base):
        _last = -1

        def update(self, n=1):
            try:
                super().update(n)
            except Exception:
                pass
            if getattr(self, "total", None) and self.total > 0:
                pct = int(self.n / self.total * 100)
                if pct != self._last:
                    self._last = pct
                    job.set_progress(
                        10 + int(pct * 0.20),
                        f"Скачивание MOSS: {self.n / 1e9:.2f} / "
                        f"{self.total / 1e9:.2f} ГБ",
                        "downloading",
                    )

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=str(local_dir),
            token=env.hf_token or None,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.bin",
                "*.txt",
                "*.py",
                "*.model",
                "*.tiktoken",
                "*.md",
                "*.jinja",
            ],
            ignore_patterns=[
                ".gitattributes",
                "*.git*",
                "*.h5",
                "*.msgpack",
                "*.onnx",
            ],
            max_workers=4,
            tqdm_class=_Tqdm,
        )
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", 0)
        if code in (401, 403):
            raise RuntimeError(
                f"HF Hub отклонил токен (код {code}). Проверьте HF_TOKEN в .env"
            )
        raise
    except Exception as e:
        raise RuntimeError(f"Ошибка скачивания MOSS: {e}") from e

    weights = _find_model_weights(local_dir)
    if weights is None:
        files = [p.name for p in local_dir.iterdir()] if local_dir.exists() else []
        raise RuntimeError(
            f"После скачивания не найдены файлы весов в {local_dir}.\n"
            f"Содержимое папки: {files}\n"
            f"Проверьте имя файла с весами в репозитории "
            f"https://huggingface.co/{MODEL_ID}/tree/main"
        )

    size_gb = _total_weights_size(local_dir) / (1024 ** 3)
    job.log("success", f"Модель MOSS скачана: {size_gb:.2f} ГБ")
    return str(local_dir)


def _load_model(job: Job):
    """
    Загружает модель и процессор MOSS-Transcribe-Diarize.

    Сначала проверяет наличие файлов в <models_dir>/<org>/<repo>/ —
    если их там нет, скачивает через snapshot_download с трансляцией
    прогресса. Затем загружает модель в память с локального пути.

    Подгружает chat_template.jinja вручную, если AutoProcessor
    его не подхватил автоматически.

    Args:
        job: Задача для логирования.

    Returns:
        Кортеж (model, processor).

    Raises:
        RuntimeError: Если модель не установлена или не загрузилась.
    """
    global _model, _processor, _model_lock

    if _model_lock is None:
        import threading
        _model_lock = threading.RLock()

    with _model_lock:
        if _model is not None and _processor is not None:
            return _model, _processor

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                f"Не установлены зависимости MOSS: {e}\n"
                f"Выполните:\n"
                f"  git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git\n"
                f"  cd MOSS-Transcribe-Diarize\n"
                f"  pip install -e ."
            )

        env = get_env()
        device = "cuda" if env.default_device == "cuda" and torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        job.set_status("downloading", "Проверка модели MOSS…")
        job.set_progress(10, "Проверка кэша MOSS…", "downloading")
        local_path = _ensure_model_downloaded(job)

        # Освобождаем VRAM перед загрузкой MOSS
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        job.set_status("loading", "Загрузка MOSS в память…")
        job.set_progress(32, "Загрузка в GPU…", "loading")
        job.log("info", f"Загрузка MOSS из {local_path} ({device}, {dtype})…")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                trust_remote_code=True,
                dtype="auto",
                token=env.hf_token or None,
                local_files_only=True,
            ).to(dtype=dtype).to(device).eval()

            processor = AutoProcessor.from_pretrained(
                local_path,
                trust_remote_code=True,
                token=env.hf_token or None,
                local_files_only=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Не удалось загрузить {MODEL_ID}: {type(e).__name__}: {e}"
            ) from e

        # Страховка: если chat_template не подхватился из файла
        if getattr(processor, "chat_template", None) is None:
            jinja_path = Path(local_path) / "chat_template.jinja"
            if jinja_path.exists():
                processor.chat_template = jinja_path.read_text(encoding="utf-8")
                job.log("info", "chat_template загружен вручную из файла")

        _model = model
        _processor = processor
        job.log("success", "Модель MOSS загружена")
        return _model, _processor


def unload_model() -> None:
    """Выгружает модель и процессор из памяти, освобождает VRAM."""
    global _model, _processor
    if _model_lock is None:
        return
    with _model_lock:
        _model = None
        _processor = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def run(job: Job) -> None:
    """
    Выполняет транскрибацию через MOSS-Transcribe-Diarize.

    Длинные аудио разбиваются на чанки по CHUNK_DURATION_SEC с перехлёстом
    CHUNK_OVERLAP_SEC, каждый обрабатывается отдельно, результаты
    склеиваются. Дубликаты на границах чанков вырезаются функцией
    _dedupe_overlap.

    Args:
        job: Задача со всеми параметрами обработки.
    """
    temp_audio = None
    chunks: List[Tuple[str, float]] = []
    stats: Dict[str, Any] = {
        "started_at": time.time(),
        "stages": {},
        "engine": "moss",
        "chunks": 0,
    }
    stage_start = time.time()

    try:
        settings = get_store().as_dict_for_worker()

        # Метаданные файла
        job.log("info", "Чтение метаданных файла…")
        src_meta = media_info.probe(job.file_path)
        if src_meta:
            if src_meta.get("duration"):
                job.log("info", f"Длительность источника: {_fmt_ts(src_meta['duration'])}")
            if job._repo:
                try:
                    job._repo.update_job(
                        job.id,
                        source_metadata=json.dumps(src_meta, ensure_ascii=False),
                    )
                except Exception:
                    pass
        stage_start = time.time()

        # Извлечение аудио
        audio_path = job.file_path
        if Path(job.file_path).suffix.lower() in SUPPORTED_VIDEO:
            job.set_status("downloading", "Извлечение аудио…")
            temp_audio = _extract_audio(job, job.file_path)
            audio_path = temp_audio
        stats["stages"]["extract_audio"] = round(time.time() - stage_start, 2)

        if job.is_cancelled():
            raise InterruptedError()

        # Загрузка модели
        stage_start = time.time()
        job.set_status("loading", "Загрузка модели MOSS…")
        model, processor = _load_model(job)
        stats["stages"]["load_model"] = round(time.time() - stage_start, 2)

        if job.is_cancelled():
            raise InterruptedError()

        # Импорт функций пакета MOSS
        try:
            from moss_transcribe_diarize.inference_utils import (
                build_transcription_messages,
                generate_transcription,
            )
            from moss_transcribe_diarize import parse_transcript
        except ImportError as e:
            raise RuntimeError(
                f"Пакет moss_transcribe_diarize не установлен: {e}\n"
                f"Выполните: pip install -e . из папки MOSS-Transcribe-Diarize"
            ) from e

        import torch

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Разбиение на чанки
        total_dur = _get_audio_duration(audio_path)
        if total_dur > CHUNK_DURATION_SEC:
            job.log("info",
                f"Аудио {_fmt_ts(total_dur)} — разбиваю на чанки "
                f"по {int(CHUNK_DURATION_SEC)} с")
            chunks = _split_audio(audio_path)
        else:
            chunks = [(audio_path, 0.0)]

        stats["chunks"] = len(chunks)
        job.log("info", f"Чанков для обработки: {len(chunks)}")

        # Транскрибация по чанкам
        job.set_status("transcribing", "Транскрибация (MOSS)…")
        all_segments: List[Dict[str, Any]] = []
        speakers_set: set = set()
        proc_time_total = 0.0

        for ci, (chunk_path, offset) in enumerate(chunks):
            if job.is_cancelled():
                raise InterruptedError()

            pct = 35 + int((ci / len(chunks)) * 60)
            job.set_progress(
                pct,
                f"Чанк {ci + 1}/{len(chunks)}…",
                "transcribing",
            )

            messages = build_transcription_messages(chunk_path)

            # Подсказки hotwords
            if settings.get("hotwords"):
                hint = f" 热词提示：{settings['hotwords']}"
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        content = msg.get("content")
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    part["text"] = part.get("text", "") + hint
                                    break
                        break

            t_chunk = time.time()
            result = generate_transcription(
                model, processor, messages,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                device=device, dtype=dtype,
            )
            proc_time_total += time.time() - t_chunk

            chunk_segments = list(parse_transcript(result["text"]))
            job.log("info",
                f"Чанк {ci + 1}/{len(chunks)}: "
                f"{len(chunk_segments)} сегментов")

            for seg in chunk_segments:
                text = (seg.text or "").strip()
                if not text:
                    continue
                speaker = _normalize_speaker(getattr(seg, "speaker", "S01"))
                speakers_set.add(speaker)
                all_segments.append({
                    "start": round(float(seg.start) + offset, 3),
                    "end": round(float(seg.end) + offset, 3),
                    "text": text,
                    "speaker": speaker,
                })

            # Освобождаем кэш между чанками
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        stats["stages"]["transcribe"] = round(proc_time_total, 2)

        if job.is_cancelled():
            raise InterruptedError()

        # Убираем дубликаты на границах чанков
        before_dedupe = len(all_segments)
        all_segments = _dedupe_overlap(all_segments)
        if len(all_segments) != before_dedupe:
            job.log("info",
                f"Удалено дубликатов на границах чанков: "
                f"{before_dedupe - len(all_segments)}")

        for i, s in enumerate(all_segments):
            s["index"] = i

        segments = all_segments

        if not segments:
            raise RuntimeError("MOSS вернул пустой результат")

        # Сохраняем сегменты
        if job._repo:
            try:
                job._repo.replace_segments(job.id, segments)
            except Exception as e:
                job.log("warn", f"Не удалось заменить сегменты: {e}")

        job.segments = segments
        job.text = "\n".join(s["text"] for s in segments)

        speakers = sorted(speakers_set)
        speaker_map = {sp: f"Спикер {i + 1}" for i, sp in enumerate(speakers)}

        job.log("success",
                f"MOSS: {len(segments)} сегментов, "
                f"спикеров: {len(speakers)} ({', '.join(speakers)})")

        # Метаданные
        proc_time = proc_time_total
        stats["total_seconds"] = round(time.time() - stats["started_at"], 2)
        if total_dur > 0:
            stats["rtf"] = round(proc_time / total_dur, 3)

        job.metadata = {
            "language": settings.get("language", "auto"),
            "duration": total_dur,
            "segments_count": len(segments),
            "model": MODEL_ID,
            "processing_time": proc_time,
            "speed_factor": round(total_dur / max(proc_time, 0.01), 2),
            "diarization": True,
            "speakers": speakers,
            "speaker_names": speaker_map,
            "engine": "moss",
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

        job.emit({
            "type": "segments_replaced",
            "segments": job.segments,
        })
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
        # Удаляем временные чанки
        try:
            _cleanup_chunks(chunks, temp_audio or job.file_path)
        except Exception:
            pass

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
