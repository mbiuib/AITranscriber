"""
Точка входа приложения.

Выполняет три шага в строгом порядке:
    1. Прописывает переменные окружения HuggingFace из .env.
    2. Регистрирует пути к CUDA DLL из пакетов nvidia-* (Windows).
    3. Запускает uvicorn с FastAPI-приложением.

Порядок критичен: ML-библиотеки (huggingface_hub, faster_whisper,
ctranslate2) читают переменные окружения и ищут DLL в момент импорта.
Если этот модуль не выполнит шаги 1 и 2 до первого импорта этих
библиотек, они стартуют в деградированном режиме — без HF-токена
и без CUDA.
"""
import os
import sys
from pathlib import Path

from app.config import apply_env_vars, get_env

apply_env_vars()


def _register_cuda_dlls() -> None:
    """
    Регистрирует директории с CUDA DLL в загрузчике Windows.

    CTranslate2 (движок faster-whisper) при импорте ищет cublas64_12.dll
    и cudnn*.dll по стандартным путям Windows. Пакеты nvidia-* кладут
    эти библиотеки в site-packages/nvidia/*/bin/, но не прописывают
    путь автоматически. Без регистрации импорт ctranslate2 падает
    с ошибкой "Library cublas64_12.dll is not found".

    Решение — обойти все известные подпапки nvidia-* в site-packages
    и добавить существующие в список DLL-директорий через
    os.add_dll_directory(). Дополнительно пути дублируются в PATH:
    некоторые зависимости ищут библиотеки именно там, а не через
    add_dll_directory.

    Функция ничего не делает на не-Windows платформах: на Linux и macOS
    загрузчик сам находит библиотеки через RPATH и DYLD_LIBRARY_PATH.
    Любые ошибки логируются в stderr и не прерывают запуск приложения —
    в худшем случае модель просто загрузится на CPU.
    """
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
        if added:
            os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
    except Exception as e:
        print(f"[WARN] CUDA DLL registration: {e}", file=sys.stderr)


_register_cuda_dlls()


if __name__ == "__main__":
    import uvicorn
    from app.server import app
    env = get_env()
    uvicorn.run(app, host=env.host, port=env.port, log_level=env.log_level)
