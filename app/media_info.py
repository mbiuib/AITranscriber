"""
Извлечение метаданных медиафайла.

Использует ffprobe (входит в FFmpeg) для чтения контейнера и потоков,
а также считает SHA-256 для дедупликации. Все ошибки проглатываются —
отсутствие метаданных не должно ломать транскрибацию.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict


def probe(path: str) -> Dict[str, Any]:
    """
    Читает метаданные медиафайла через ffprobe.

    Возвращает нормализованный словарь: длительность, битрейт, формат,
    кодеки, разрешение (для видео), частоту дискретизации (для аудио).
    Все поля опциональны — если ffprobe их не вернул, они отсутствуют.

    Args:
        path: Путь к медиафайлу.

    Returns:
        Словарь с метаданными или пустой словарь при любой ошибке.
    """
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                path,
            ],
            timeout=10,
            stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
    except Exception:
        return {}

    result: Dict[str, Any] = {}
    fmt = data.get("format", {})
    result["format_name"] = fmt.get("format_long_name") or fmt.get("format_name")
    if "duration" in fmt:
        try:
            result["duration"] = float(fmt["duration"])
        except (ValueError, TypeError):
            pass
    if "bit_rate" in fmt:
        try:
            result["bitrate_kbps"] = round(int(fmt["bit_rate"]) / 1000)
        except (ValueError, TypeError):
            pass
    if "size" in fmt:
        try:
            result["size_bytes"] = int(fmt["size"])
        except (ValueError, TypeError):
            pass

    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")

        if codec_type == "audio" and "audio" not in result:
            result["audio"] = {
                "codec": stream.get("codec_name"),
                "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
                "channels": stream.get("channels"),
            }

        elif codec_type == "video" and "video" not in result:
            result["video"] = {
                "codec": stream.get("codec_name"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "fps": _parse_fps(stream.get("r_frame_rate")),
            }

    return result


def _parse_fps(rate: str | None) -> float | None:
    """
    Преобразует строку вида "30/1" в число 30.0.

    Args:
        rate: Строка частоты кадров от ffprobe.

    Returns:
        Частота в fps или None, если строка некорректна.
    """
    if not rate or "/" not in rate:
        return None
    try:
        num, den = rate.split("/", 1)
        return round(int(num) / int(den), 2) if int(den) else None
    except (ValueError, ZeroDivisionError):
        return None


def file_hash(path: str, chunk_size: int = 1024 * 1024) -> str | None:
    """
    Считает SHA-256 хеш файла.

    Читает файл порциями по chunk_size, чтобы не грузить большие видео
    в память целиком.

    Args:
        path: Путь к файлу.
        chunk_size: Размер читаемого блока в байтах.

    Returns:
        Hex-строка хеша или None при ошибке чтения.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None
