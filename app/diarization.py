"""
Диаризация спикеров через pyannote.audio.

Определяет, кто говорил в каждый момент времени, и сопоставляет это
с сегментами Whisper. Использует модель pyannote/speaker-diarization-3.1
(или community-1 в версии 4.0+).

Требует HF-токен с принятыми условиями использования моделей:
    https://huggingface.co/pyannote/speaker-diarization-3.1
    https://huggingface.co/pyannote/segmentation-3.0

Пайплайн загружается один раз и кэшируется в памяти. При первом запуске
скачивает ~50 МБ весов. После завершения работы модели выгружаются из
VRAM, чтобы не мешать следующей задаче.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from .config import get_env

_pipeline = None
_pipeline_lock = threading.RLock()
_diarization_available: Optional[bool] = None
_last_error: Optional[str] = None

MODEL_NAME = "pyannote/speaker-diarization-3.1"


def is_available() -> bool:
    """
    Проверяет, доступна ли диаризация (установлен pyannote).

    Returns:
        True, если пакет pyannote.audio импортируется без ошибок.
    """
    global _diarization_available
    if _diarization_available is None:
        try:
            from pyannote.audio import Pipeline  # noqa: F401
            _diarization_available = True
        except ImportError:
            _diarization_available = False
    return _diarization_available


def last_error() -> Optional[str]:
    """
    Возвращает текст последней ошибки инициализации пайплайна.

    Returns:
        Строка с ошибкой или None, если ошибок не было.
    """
    return _last_error


def _get_pipeline():
    """
    Загружает и кэширует pyannote Pipeline.

    Устойчива к разнице версий: в pyannote.audio >= 4.0 параметр
    авторизации называется token, в 3.x — use_auth_token. Сначала
    пробуем новый вариант, при TypeError откатываемся на старый.

    Returns:
        Pipeline или None, если не удалось загрузить.

    Raises:
        RuntimeError: Если HF-токен не задан или модель недоступна.
    """
    global _pipeline, _last_error

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        if not is_available():
            _last_error = "pyannote.audio не установлен"
            raise RuntimeError(_last_error)

        env = get_env()
        if not env.hf_token:
            _last_error = (
                "Не задан HF_TOKEN. Диаризация требует токен с принятыми "
                "условиями моделей pyannote."
            )
            raise RuntimeError(_last_error)

        try:
            from pyannote.audio import Pipeline
            import torch

            pipeline = None
            errors = []

            try:
                pipeline = Pipeline.from_pretrained(
                    MODEL_NAME,
                    token=env.hf_token,
                )
            except TypeError as e:
                errors.append(f"token=... → {e}")
                try:
                    pipeline = Pipeline.from_pretrained(
                        MODEL_NAME,
                        use_auth_token=env.hf_token,
                    )
                except TypeError as e2:
                    errors.append(f"use_auth_token=... → {e2}")
                    try:
                        pipeline = Pipeline.from_pretrained(MODEL_NAME)
                    except Exception as e3:
                        errors.append(f"no-auth → {e3}")
                        raise RuntimeError("; ".join(errors))

            if pipeline is None:
                raise RuntimeError("Pipeline вернул None")

            if env.default_device == "cuda" and torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))

            _pipeline = pipeline
            _last_error = None
            return _pipeline

        except RuntimeError:
            raise
        except Exception as e:
            _last_error = (
                f"Не удалось загрузить {MODEL_NAME}: "
                f"{type(e).__name__}: {e}\n"
                f"Проверьте, что приняты условия использования:\n"
                f"  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                f"  https://huggingface.co/pyannote/segmentation-3.0"
            )
            raise RuntimeError(_last_error)


def unload_pipeline() -> None:
    """
    Выгружает пайплайн из памяти и освобождает VRAM.

    Вызывается после завершения диаризации, чтобы освободить GPU
    для следующей задачи.
    """
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            _pipeline = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


def run(
    audio_path: str,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    log=None,
) -> List[Dict[str, Any]]:
    """
    Запускает диаризацию и возвращает таймлайн спикеров.

    Совместима с pyannote.audio 3.x и 4.x. В 3.x пайплайн возвращает
    Annotation напрямую, в 4.x — DiarizeOutput с полем speaker_diarization.
    Обе структуры приводятся к одному списку turns.

    По возможности используется exclusive_speaker_diarization из 4.x:
    он гарантирует, что каждая секунда отнесена ровно к одному спикеру.

    Args:
        audio_path: Путь к аудиофайлу (WAV, 16 кГц моно).
        min_speakers: Минимум спикеров (None — автодетект).
        max_speakers: Максимум спикеров (None — автодетект).
        log: Callback для логирования: log(level, message).

    Returns:
        Список словарей {"start": float, "end": float, "speaker": str},
        отсортированный по start.

    Raises:
        RuntimeError: Если пайплайн не загружен или диаризация упала.
    """
    def _log(level: str, msg: str) -> None:
        if log:
            try:
                log(level, msg)
            except Exception:
                pass

    pipeline = _get_pipeline()

    kwargs: Dict[str, Any] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = int(min_speakers)
    if max_speakers is not None:
        kwargs["max_speakers"] = int(max_speakers)

    _log("info", f"Диаризация: модель {MODEL_NAME}")
    if kwargs:
        _log("info", f"Ограничения: {kwargs}")

    try:
        import torchaudio
        import torch

        _log("info", "Загрузка аудио через torchaudio…")
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000,
            )
            waveform = resampler(waveform)
            sample_rate = 16000

        audio_input = {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

        result = pipeline(audio_input, **kwargs)

    except Exception as e:
        raise RuntimeError(f"Ошибка диаризации: {type(e).__name__}: {e}") from e

    annotation = _extract_annotation(result, _log)
    if annotation is None:
        raise RuntimeError("Не удалось извлечь таймлайн из результата пайплайна")

    turns: List[Dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })

    unique = sorted({t["speaker"] for t in turns})
    _log("success",
         f"Диаризация завершена: {len(turns)} реплик, "
         f"спикеров: {len(unique)} ({', '.join(unique)})")

    return turns


def _extract_annotation(result: Any, log) -> Any:
    """
    Извлекает Annotation из результата пайплайна любой версии.

    В pyannote.audio 3.x результат — это Annotation напрямую. В 4.x —
    DiarizeOutput с полями speaker_diarization и
    exclusive_speaker_diarization. Предпочтение отдаётся exclusive:
    он гарантирует непересекающиеся интервалы.

    Args:
        result: Объект, возвращённый пайплайном.
        log: Callback для логирования.

    Returns:
        Annotation или None, если структура неизвестна.
    """
    if hasattr(result, "itertracks"):
        return result

    if hasattr(result, "exclusive_speaker_diarization"):
        excl = result.exclusive_speaker_diarization
        if excl is not None and hasattr(excl, "itertracks"):
            try:
                log("info", "Используется exclusive_speaker_diarization")
            except Exception:
                pass
            return excl

    if hasattr(result, "speaker_diarization"):
        sd = result.speaker_diarization
        if sd is not None and hasattr(sd, "itertracks"):
            return sd

    return None


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """
    Считает длину пересечения двух интервалов в секундах.

    Args:
        a_start: Начало первого интервала.
        a_end: Конец первого интервала.
        b_start: Начало второго интервала.
        b_end: Конец второго интервала.

    Returns:
        Длина пересечения или 0, если интервалы не пересекаются.
    """
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def split_segments_by_speaker(
    segments: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    min_segment_duration: float = 1.0,
    merge_gap: float = 1.0,
    no_split_below: float = 3.0,
) -> List[Dict[str, Any]]:
    """
    Разбивает сегменты Whisper по границам реплик спикеров.

    Короткие сегменты (< no_split_below) не разбиваются — им назначается
    один спикер по мажоритарному голосованию слов. Остальные разбиваются
    по word-level перекрытию с репликами pyannote.

    Args:
        segments: Сегменты Whisper с полями start, end, text, words.
        turns: Таймлайн спикеров из run().
        min_segment_duration: Минимальная длина итогового сегмента.
        merge_gap: Пауза в секундах, после которой начинается новый
            сегмент даже для одного спикера.
        no_split_below: Сегменты короче этой длительности не разбиваются.

    Returns:
        Список сегментов с полями index, start, end, text, speaker.
    """
    if not turns:
        return [
            {
                "index": i,
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
                "speaker": "SPEAKER_00",
            }
            for i, s in enumerate(segments)
        ]

    if not segments:
        return []

    sorted_turns = sorted(turns, key=lambda t: t["start"])

    words: List[Dict[str, Any]] = []
    for seg in segments:
        seg_words_raw = seg.get("words") or []
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        seg_duration = seg_end - seg_start

        if not seg_words_raw:
            words.append({
                "start": seg_start,
                "end": seg_end,
                "word": seg["text"],
                "speaker": _speaker_for_word(seg_start, seg_end, sorted_turns),
            })
            continue

        bag: List[Dict[str, Any]] = []
        for w in seg_words_raw:
            bag.append({
                "start": float(w["start"]),
                "end": float(w["end"]),
                "word": str(w["word"]),
            })

        if seg_duration < no_split_below:
            _assign_majority_speaker(bag, sorted_turns)
        else:
            for w in bag:
                w["speaker"] = _speaker_for_word(
                    w["start"], w["end"], sorted_turns,
                )

        words.extend(bag)

    if not words:
        return []

    runs = _group_words_into_runs(words, merge_gap)
    runs = _merge_adjacent_runs(runs, merge_gap)
    runs = _absorb_short_runs(runs, min_segment_duration)

    result: List[Dict[str, Any]] = []
    for r in runs:
        text = "".join(w["word"] for w in r["words"]).strip()
        if not text:
            continue
        result.append({
            "index": len(result),
            "start": round(r["start"], 3),
            "end": round(r["end"], 3),
            "speaker": r["speaker"],
            "text": text,
        })

    return result


def _assign_majority_speaker(
    words: List[Dict[str, Any]], sorted_turns: List[Dict[str, Any]],
) -> None:
    """
    Назначает всем словам одного спикера по мажоритарному голосованию.

    Каждое слово голосует своей длительностью за спикера, определённого
    по перекрытию. Побеждает спикер с максимальным суммарным временем.
    Модифицирует список на месте.

    Args:
        words: Слова с полями start, end.
        sorted_turns: Отсортированный таймлайн реплик.
    """
    weights: Dict[str, float] = {}
    for w in words:
        sp = _speaker_for_word(w["start"], w["end"], sorted_turns)
        weights[sp] = weights.get(sp, 0.0) + (w["end"] - w["start"])

    if weights:
        winner = max(weights.items(), key=lambda x: x[1])[0]
    else:
        winner = "UNKNOWN"

    for w in words:
        w["speaker"] = winner


def _speaker_for_word(
    start: float, end: float, sorted_turns: List[Dict[str, Any]],
) -> str:
    """
    Определяет спикера для слова по максимальному перекрытию.

    Слово привязывается к той реплике, с которой у него больше всего
    общих миллисекунд. Если перекрытий нет вообще (слово в паузе),
    берётся ближайшая по центру реплика.

    Args:
        start: Начало слова.
        end: Конец слова.
        sorted_turns: Отсортированный таймлайн реплик.

    Returns:
        Идентификатор спикера.
    """
    if not sorted_turns:
        return "UNKNOWN"

    best_speaker = None
    best_overlap = 0.0

    for t in sorted_turns:
        if t["end"] <= start:
            continue
        if t["start"] >= end:
            break
        ov = _overlap(start, end, t["start"], t["end"])
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = t["speaker"]

    if best_speaker is not None:
        return best_speaker

    center = (start + end) / 2
    nearest = min(
        sorted_turns,
        key=lambda t: abs((t["start"] + t["end"]) / 2 - center),
    )
    return nearest["speaker"]


def _group_words_into_runs(
    words: List[Dict[str, Any]], merge_gap: float,
) -> List[Dict[str, Any]]:
    """
    Группирует слова в runs по спикеру и паузам.

    Новый run начинается при смене спикера или при паузе длиннее
    merge_gap.

    Args:
        words: Слова с уже назначенным спикером.
        merge_gap: Порог паузы между словами.

    Returns:
        Список runs с полями start, end, speaker, words.
    """
    if not words:
        return []

    runs: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for w in words:
        if current is None:
            current = {
                "start": w["start"],
                "end": w["end"],
                "speaker": w["speaker"],
                "words": [w],
            }
            continue

        same_speaker = current["speaker"] == w["speaker"]
        gap = w["start"] - current["end"]

        if same_speaker and gap <= merge_gap:
            current["end"] = w["end"]
            current["words"].append(w)
        else:
            runs.append(current)
            current = {
                "start": w["start"],
                "end": w["end"],
                "speaker": w["speaker"],
                "words": [w],
            }

    if current:
        runs.append(current)

    return runs


def _merge_adjacent_runs(
    runs: List[Dict[str, Any]], max_gap: float,
) -> List[Dict[str, Any]]:
    """
    Объединяет соседние runs одного спикера.

    Проходит по списку и склеивает пары runs, у которых совпадает
    speaker и пауза между ними меньше max_gap.

    Args:
        runs: Черновые runs после группировки.
        max_gap: Максимальная пауза для склейки.

    Returns:
        Список runs после склейки.
    """
    if len(runs) <= 1:
        return runs

    merged: List[Dict[str, Any]] = [runs[0]]
    for run in runs[1:]:
        prev = merged[-1]
        if prev["speaker"] == run["speaker"]:
            gap = run["start"] - prev["end"]
            if gap <= max_gap:
                prev["end"] = run["end"]
                prev["words"].extend(run["words"])
                continue
        merged.append(run)
    return merged


def _absorb_short_runs(
    runs: List[Dict[str, Any]], min_duration: float,
) -> List[Dict[str, Any]]:
    """
    Приклеивает короткие runs к соседям того же спикера.

    Run короче min_duration пытается склеиться с предыдущим run'ом
    того же спикера, если такой есть.

    Args:
        runs: Runs после объединения соседних.
        min_duration: Минимально допустимая длина run.

    Returns:
        Список runs после поглощения коротких.
    """
    if len(runs) <= 1:
        return runs

    result: List[Dict[str, Any]] = []
    for run in runs:
        duration = run["end"] - run["start"]

        if duration >= min_duration or not result:
            result.append(run)
            continue

        prev = result[-1]
        if prev["speaker"] == run["speaker"]:
            prev["end"] = run["end"]
            prev["words"].extend(run["words"])
        else:
            result.append(run)

    return result


def format_speaker(speaker: str, mapping: Optional[Dict[str, str]] = None) -> str:
    """
    Форматирует имя спикера для отображения.

    Args:
        speaker: Исходный идентификатор ("SPEAKER_00").
        mapping: Опциональный словарь переопределений имён.

    Returns:
        Человекочитаемое имя ("Спикер 1" или пользовательское).
    """
    if mapping and speaker in mapping:
        return mapping[speaker]
    if speaker.startswith("SPEAKER_"):
        try:
            idx = int(speaker.split("_")[1]) + 1
            return f"Спикер {idx}"
        except (ValueError, IndexError):
            pass
    if speaker == "UNKNOWN":
        return "Неизвестный"
    return speaker