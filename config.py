"""
Конфигурация приложения — читается из .env.

Использование:
    from config import get_settings, apply_env
    settings = get_settings()   # объект с типизированными полями
    apply_env(settings)          # прописать HF-переменные в os.environ
                                 # ВЫЗВАТЬ ДО импорта ML-библиотек!
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- HuggingFace ----------
    hf_token: str = ""
    hf_transfer: bool = True
    hf_download_timeout: int = 30
    hf_symlinks_warning: bool = False
    hf_telemetry: bool = False

    # ---------- Пути ----------
    models_dir: Path = Path(r"G:\LLMs\models")
    upload_dir: Path = Path("uploads")

    # ---------- Сервер ----------
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    cors_origins: str = "*"

    # ---------- Whisper ----------
    default_model: str = "large-v3"
    default_language: str = "auto"
    default_batch_size: int = 8
    default_vad: bool = True
    default_compute_type: str = "float16"
    default_device: str = "cuda"

    # ---------- Лимиты ----------
    max_upload_mb: int = 4096
    max_workers: int = 4

    # ---------- Производные ----------

    @property
    def upload_dir_resolved(self) -> Path:
        """Абсолютный путь к папке загрузок."""
        if self.upload_dir.is_absolute():
            return self.upload_dir
        return BASE_DIR / self.upload_dir

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Singleton: конфиг читается один раз."""
    return Settings()


def apply_env(settings: Settings) -> None:
    """
    Прописывает переменные окружения для HuggingFace Hub.
    Должно вызываться ДО первого импорта huggingface_hub / faster_whisper / ctranslate2.
    """
    import os

    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.hf_token

    if not settings.hf_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    if not settings.hf_telemetry:
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    if settings.hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(settings.hf_download_timeout)
