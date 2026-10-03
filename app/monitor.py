"""
Мониторинг ресурсов: CPU, RAM, GPU, диски.

Собирает снимок текущей загрузки системы и ведёт скользящую историю
последних замеров для отображения графиков в UI. Использует psutil для
CPU/RAM и pynvml (или nvidia-smi как fallback) для GPU.

Обе библиотеки опциональны: если psutil недоступен, функции CPU/RAM
возвращают нули, если pynvml недоступен — пробуется nvidia-smi, а если
и его нет, поле gpu в снимке будет None. Это позволяет модулю работать
в любом окружении, включая CPU-only и системы без NVIDIA-драйверов.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any, Deque, Dict, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False


HISTORY_LEN = 60
_history: Dict[str, Deque] = {
    "cpu": deque(maxlen=HISTORY_LEN),
    "ram": deque(maxlen=HISTORY_LEN),
    "gpu_util": deque(maxlen=HISTORY_LEN),
    "gpu_mem": deque(maxlen=HISTORY_LEN),
}
_history_lock = Lock()


def _gpu_info_nvml() -> Optional[Dict[str, Any]]:
    """
    Собирает информацию о GPU через NVML (nvidia-ml-py).

    Основной путь получения GPU-метрик. Даёт больше данных, чем
    nvidia-smi: температуру и текущее потребление энергии. Опциональные
    поля (temperature, power) могут быть None, если конкретная модель
    карты их не поддерживает.

    Returns:
        Словарь с метриками GPU или None, если NVML недоступен или
        вызов любой из функций nvml завершился ошибкой.
    """
    if not HAS_NVML:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        except Exception:
            power = None
            power_limit = None
        return {
            "name": name,
            "utilization": util.gpu,
            "memory_used": mem.used,
            "memory_total": mem.total,
            "memory_percent": round(mem.used / mem.total * 100, 1),
            "memory_used_gb": round(mem.used / (1024 ** 3), 2),
            "memory_total_gb": round(mem.total / (1024 ** 3), 2),
            "temperature": temp,
            "power_w": round(power, 1) if power else None,
            "power_limit_w": round(power_limit, 1) if power_limit else None,
        }
    except Exception:
        return None


def _gpu_info_smi() -> Optional[Dict[str, Any]]:
    """
    Собирает информацию о GPU через nvidia-smi.

    Резервный путь на случай, если NVML недоступен (например, при
    нестандартной установке драйвера). Даёт меньше данных — без
    энергопотребления — но работает там, где nvidia-smi в PATH.

    Returns:
        Словарь с метриками GPU или None, если nvidia-smi отсутствует,
        завершился с ошибкой или превысил таймаут в 2 секунды.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=2, stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        name, util, mem_used, mem_total, temp = parts
        mem_used_b = int(mem_used) * 1024 ** 2
        mem_total_b = int(mem_total) * 1024 ** 2
        return {
            "name": name,
            "utilization": int(util),
            "memory_used": mem_used_b,
            "memory_total": mem_total_b,
            "memory_percent": round(mem_used_b / mem_total_b * 100, 1),
            "memory_used_gb": round(mem_used_b / 1024 ** 3, 2),
            "memory_total_gb": round(mem_total_b / 1024 ** 3, 2),
            "temperature": int(temp),
            "power_w": None,
            "power_limit_w": None,
        }
    except Exception:
        return None


def snapshot(models_dir: Path, upload_dir: Path) -> Dict[str, Any]:
    """
    Возвращает срез текущего состояния ресурсов.

    Собирает метрики CPU, RAM, GPU и дисков, попутно дописывая текущие
    значения CPU/RAM/GPU в скользящую историю (deque длиной HISTORY_LEN).
    Каждое поле структуры имеет предсказуемый набор ключей даже если
    соответствующая библиотека недоступна — UI всегда может рассчитывать
    на наличие всех полей.

    Args:
        models_dir: Папка с моделями — для расчёта свободного места
            на диске под модели.
        upload_dir: Папка с загруженными файлами — для расчёта места
            под пользовательские данные.

    Returns:
        Словарь с полями:
            timestamp (float): Unix-время снимка.
            cpu (dict): percent, count, count_physical, freq_mhz.
            ram (dict): total_gb, used_gb, available_gb, percent.
            gpu (dict | None): метрики GPU или None, если GPU не найдена.
            disks (list[dict]): информация по каждой смонтированной точке.
            history (dict): история последних HISTORY_LEN значений
                для cpu, ram, gpu_util, gpu_mem в виде списков.
    """
    result: Dict[str, Any] = {"timestamp": time.time()}

    if HAS_PSUTIL:
        result["cpu"] = {
            "percent": psutil.cpu_percent(interval=None),
            "count": psutil.cpu_count(logical=True),
            "count_physical": psutil.cpu_count(logical=False),
            "freq_mhz": int(psutil.cpu_freq().current) if psutil.cpu_freq() else None,
        }
        mem = psutil.virtual_memory()
        result["ram"] = {
            "total_gb": round(mem.total / 1024 ** 3, 2),
            "used_gb": round(mem.used / 1024 ** 3, 2),
            "available_gb": round(mem.available / 1024 ** 3, 2),
            "percent": mem.percent,
        }
    else:
        result["cpu"] = {"percent": 0, "count": 0, "count_physical": 0}
        result["ram"] = {"total_gb": 0, "used_gb": 0, "available_gb": 0, "percent": 0}

    gpu = _gpu_info_nvml() or _gpu_info_smi()
    result["gpu"] = gpu

    disks = []
    for label, path in [("models", models_dir), ("data", upload_dir)]:
        try:
            usage = shutil.disk_usage(path)
            disks.append({
                "label": label,
                "path": str(path),
                "total_gb": round(usage.total / 1024 ** 3, 1),
                "used_gb": round(usage.used / 1024 ** 3, 1),
                "free_gb": round(usage.free / 1024 ** 3, 1),
                "percent": round(usage.used / usage.total * 100, 1),
            })
        except Exception:
            pass
    result["disks"] = disks

    with _history_lock:
        if HAS_PSUTIL:
            _history["cpu"].append(result["cpu"]["percent"])
            _history["ram"].append(result["ram"]["percent"])
        if gpu:
            _history["gpu_util"].append(gpu["utilization"])
            _history["gpu_mem"].append(gpu["memory_percent"])
        result["history"] = {k: list(v) for k, v in _history.items()}

    return result
