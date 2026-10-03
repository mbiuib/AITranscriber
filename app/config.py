"""
Конфигурация приложения.

Значения по умолчанию читаются из .env через pydantic-settings.
Runtime-переопределения хранятся в data/settings.json поверх значений .env.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent


RUNTIME_SCHEMA: Dict[str, Dict[str, Any]] = {
    "transcription.model": {
        "type": "select",
        "options": ["tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"],
        "label": "Модель", "group": "transcription", "order": 1,
    },
    "transcription.language": {
        "type": "select",
        "options": ["auto", "ru", "en", "de", "fr", "es", "zh", "ja", "ko", "uk"],
        "label": "Язык", "group": "transcription", "order": 2,
    },
    "transcription.batch_size": {
        "type": "int", "min": 1, "max": 32,
        "label": "Batch size", "group": "transcription", "order": 3,
        "hint": "Больше — быстрее, но требует больше VRAM",
    },
    "transcription.beam_size": {
        "type": "int", "min": 1, "max": 10,
        "label": "Beam size", "group": "transcription", "order": 4,
    },
    "transcription.vad": {
        "type": "bool",
        "label": "VAD-фильтрация", "group": "transcription", "order": 5,
    },
    "transcription.condition_on_previous": {
        "type": "bool",
        "label": "Учитывать предыдущий текст", "group": "transcription", "order": 6,
        "hint": "Помогает при смешанной речи (русский + английские термины)",
    },
    "transcription.device": {
        "type": "select", "options": ["cuda", "cpu"],
        "label": "Устройство", "group": "advanced", "order": 1,
    },
    "transcription.compute_type": {
        "type": "select", "options": ["float16", "int8_float16", "int8", "float32"],
        "label": "Точность", "group": "advanced", "order": 2,
    },
    "transcription.initial_prompt": {
        "type": "text",
        "label": "Начальный промпт", "group": "advanced", "order": 3,
        "hint": "Задаёт контекст. Помогает распознавать специфичные термины.",
    },
    "transcription.hotwords": {
        "type": "text",
        "label": "Слова-подсказки", "group": "advanced", "order": 4,
        "hint": "Имена, термины через запятую",
    },
    "server.max_upload_mb": {
        "type": "int", "min": 1, "max": 102400,
        "label": "Максимальный размер файла (МБ)", "group": "server", "order": 1,
    },
    "server.retention_hours": {
        "type": "int", "min": 1, "max": 8760,
        "label": "Хранить результаты (часов)", "group": "server", "order": 2,
        "hint": "Файлы и результаты старше указанного времени удаляются автоматически",
    },
    "server.cleanup_interval_min": {
        "type": "int", "min": 1, "max": 1440,
        "label": "Период очистки (минут)", "group": "server", "order": 3,
    },
    "ui.language": {
        "type": "select", "options": ["ru", "en"],
        "label": "Язык интерфейса", "group": "ui", "order": 1,
    },
    "ui.theme": {
        "type": "select", "options": ["dark", "light"],
        "label": "Тема", "group": "ui", "order": 2,
    },
}


class EnvSettings(BaseSettings):
    """
    Настройки из .env, доступные только для чтения при старте приложения.

    Служат значениями по умолчанию для SettingsStore, а также задают
    инфраструктурные параметры (порты, пути, токены), которые не имеет
    смысла менять в рантайме.

    Attributes:
        hf_token: Токен HuggingFace для скачивания моделей.
        hf_transfer: Использовать hf_transfer для ускорения загрузки.
        hf_download_timeout: Таймаут HTTP-запросов к HF Hub в секундах.
        hf_symlinks_warning: Показывать предупреждение о симлинках.
        hf_telemetry: Отправлять телеметрию в HF Hub.
        models_dir: Корневая папка для хранения моделей.
        data_dir: Папка для данных приложения (uploads, settings.json).
        host: Адрес прослушивания сервера.
        port: Порт сервера.
        log_level: Уровень логирования uvicorn.
        cors_origins: Разрешённые origins через запятую.
        default_model: Модель Whisper по умолчанию.
        default_language: Язык транскрибации по умолчанию.
        default_batch_size: Размер батча по умолчанию.
        default_vad: Включена ли VAD-фильтрация по умолчанию.
        default_compute_type: Точность вычислений по умолчанию.
        default_device: Устройство (cuda/cpu) по умолчанию.
        default_beam_size: Beam size по умолчанию.
        default_condition_on_previous: Учитывать ли предыдущий текст.
        default_initial_prompt: Начальный промпт по умолчанию.
        default_hotwords: Слова-подсказки по умолчанию.
        max_upload_mb: Максимальный размер загружаемого файла в МБ.
        max_workers: Параллельных HTTP-загрузок при скачивании моделей.
        retention_hours: Сколько часов хранить завершённые задачи.
        cleanup_interval_min: Период фоновой очистки в минутах.
        default_ui_language: Язык интерфейса по умолчанию.
        default_ui_theme: Тема интерфейса по умолчанию.
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    hf_token: str = ""
    hf_transfer: bool = True
    hf_download_timeout: int = 30
    hf_symlinks_warning: bool = False
    hf_telemetry: bool = False

    models_dir: Path = Path(r"G:\LLMs\models")
    data_dir: Path = Path("data")

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    cors_origins: str = "*"

    default_model: str = "large-v3"
    default_language: str = "ru"
    default_batch_size: int = 8
    default_vad: bool = True
    default_compute_type: str = "float16"
    default_device: str = "cuda"
    default_beam_size: int = 5
    default_condition_on_previous: bool = True
    default_initial_prompt: str = (
        "Это совещание на русском языке. Используются технические термины "
        "на английском: Kubernetes, Docker, REST API, CI/CD, JavaScript, Python."
    )
    default_hotwords: str = ""

    max_upload_mb: int = 4096
    max_workers: int = 4
    retention_hours: int = 24
    cleanup_interval_min: int = 30

    default_ui_language: str = "ru"
    default_ui_theme: str = "dark"

    @property
    def data_dir_resolved(self) -> Path:
        """
        data_dir как абсолютный путь.

        Относительные пути интерпретируются относительно корня проекта
        (папки, содержащей .env и app/).

        Returns:
            Абсолютный путь к папке с данными приложения.
        """
        return self.data_dir if self.data_dir.is_absolute() else BASE_DIR / self.data_dir

    @property
    def upload_dir(self) -> Path:
        """
        Папка для загруженных пользователем файлов.

        Returns:
            Путь к подпапке uploads/ внутри data_dir_resolved.
        """
        return self.data_dir_resolved / "uploads"

    @property
    def settings_file(self) -> Path:
        """
        JSON-файл с runtime-переопределениями настроек.

        Returns:
            Путь к файлу settings.json внутри data_dir_resolved.
        """
        return self.data_dir_resolved / "settings.json"

    @property
    def cors_origins_list(self) -> List[str]:
        """
        cors_origins, разобранный в список строк.

        Returns:
            Список origins без пробелов. Пустые элементы отброшены.
        """
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        """
        max_upload_mb в байтах.

        Returns:
            Максимальный размер файла в байтах.
        """
        return self.max_upload_mb * 1024 * 1024

    @property
    def db_path(self) -> Path:
        """
        Путь к файлу SQLite с историей задач.

        Returns:
            Путь к data/jobs.db внутри data_dir_resolved.
        """
        return self.data_dir_resolved / "jobs.db"


class SettingsStore:
    """
    Runtime-настройки поверх значений из .env.

    Все ключи соответствуют RUNTIME_SCHEMA. Если ключ не переопределён
    пользователем, возвращается значение из EnvSettings. Переопределения
    сохраняются в JSON и переживают перезапуск приложения.

    Класс потокобезопасен: все операции чтения и записи защищены RLock.

    Attributes:
        _env: Источник значений по умолчанию.
        _overrides: Текущие переопределения, сохранённые пользователем.
    """

    def __init__(self, env: EnvSettings):
        """
        Инициализирует хранилище и подгружает сохранённые переопределения.

        Args:
            env: Настройки из .env, используемые как значения по умолчанию.
        """
        self._env = env
        self._lock = threading.RLock()
        self._overrides: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """
        Читает сохранённые переопределения с диска.

        Если файл отсутствует или повреждён, хранилище остаётся пустым —
        это не считается ошибкой, просто используются значения из .env.
        """
        f = self._env.settings_file
        if f.exists():
            try:
                self._overrides = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                self._overrides = {}

    def _save(self) -> None:
        """
        Записывает текущие переопределения на диск.

        Создаёт родительскую папку, если её нет. Формат — UTF-8 JSON
        с отступом в 2 пробела для удобства ручного редактирования.
        """
        f = self._env.settings_file
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(
            json.dumps(self._overrides, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _default_for(self, key: str) -> Any:
        """
        Возвращает значение по умолчанию из .env для указанного ключа.

        Args:
            key: Ключ настройки из RUNTIME_SCHEMA.

        Returns:
            Значение по умолчанию или None, если ключ неизвестен.
        """
        env = self._env
        mapping = {
            "transcription.model": env.default_model,
            "transcription.language": env.default_language,
            "transcription.batch_size": env.default_batch_size,
            "transcription.beam_size": env.default_beam_size,
            "transcription.vad": env.default_vad,
            "transcription.condition_on_previous": env.default_condition_on_previous,
            "transcription.device": env.default_device,
            "transcription.compute_type": env.default_compute_type,
            "transcription.initial_prompt": env.default_initial_prompt,
            "transcription.hotwords": env.default_hotwords,
            "server.max_upload_mb": env.max_upload_mb,
            "server.retention_hours": env.retention_hours,
            "server.cleanup_interval_min": env.cleanup_interval_min,
            "ui.language": env.default_ui_language,
            "ui.theme": env.default_ui_theme,
        }
        return mapping.get(key)

    def get(self, key: str) -> Any:
        """
        Возвращает значение настройки.

        Сначала ищет переопределение в runtime-хранилище, затем —
        значение по умолчанию из .env.

        Args:
            key: Ключ настройки из RUNTIME_SCHEMA.

        Returns:
            Значение настройки любого типа (str, int, bool) или None.
        """
        with self._lock:
            if key in self._overrides:
                return self._overrides[key]
            return self._default_for(key)

    def set(self, key: str, value: Any) -> None:
        """
        Устанавливает одну настройку и сохраняет изменения на диск.

        Args:
            key: Ключ настройки из RUNTIME_SCHEMA.
            value: Новое значение. Тип не валидируется — предполагается,
                что вызывающая сторона уже проверила его по схеме.
        """
        with self._lock:
            self._overrides[key] = value
            self._save()

    def set_many(self, values: Dict[str, Any]) -> None:
        """
        Устанавливает несколько настроек одной транзакцией.

        Все изменения записываются на диск одной операцией — это быстрее
        и безопаснее, чем вызывать set() в цикле.

        Args:
            values: Словарь ключей и значений для переопределения.
        """
        with self._lock:
            self._overrides.update(values)
            self._save()

    def reset(self, key: Optional[str] = None) -> None:
        """
        Сбрасывает настройки к значениям из .env.

        Args:
            key: Если указан — сбрасывается только эта настройка.
                Если None — сбрасываются все переопределения.
        """
        with self._lock:
            if key:
                self._overrides.pop(key, None)
            else:
                self._overrides.clear()
            self._save()

    def all(self) -> Dict[str, Any]:
        """
        Возвращает снимок всех настроек.

        Включает как переопределения пользователя, так и значения
        по умолчанию из .env для ключей, которые не переопределены.

        Returns:
            Словарь, где ключи — из RUNTIME_SCHEMA, значения — актуальные
            настройки на момент вызова.
        """
        with self._lock:
            return {k: self.get(k) for k in RUNTIME_SCHEMA.keys()}

    def as_dict_for_worker(self) -> Dict[str, Any]:
        """
        Возвращает настройки транскрибации в формате для воркера.

        Ключи не совпадают с RUNTIME_SCHEMA — воркер ожидает «плоский»
        словарь без префиксов групп.

        Returns:
            Словарь параметров транскрибации: model, language, batch_size,
            beam_size, vad, condition_on_previous, device, compute_type,
            initial_prompt, hotwords.
        """
        return {
            "model": self.get("transcription.model"),
            "language": self.get("transcription.language"),
            "batch_size": self.get("transcription.batch_size"),
            "beam_size": self.get("transcription.beam_size"),
            "vad": self.get("transcription.vad"),
            "condition_on_previous": self.get("transcription.condition_on_previous"),
            "device": self.get("transcription.device"),
            "compute_type": self.get("transcription.compute_type"),
            "initial_prompt": self.get("transcription.initial_prompt"),
            "hotwords": self.get("transcription.hotwords"),
        }


_env_singleton: Optional[EnvSettings] = None
_store_singleton: Optional[SettingsStore] = None


def get_env() -> EnvSettings:
    """
    Возвращает singleton EnvSettings.

    При первом вызове читает .env и создаёт экземпляр, при последующих —
    возвращает уже созданный объект.

    Returns:
        Единственный на процесс экземпляр EnvSettings.
    """
    global _env_singleton
    if _env_singleton is None:
        _env_singleton = EnvSettings()
    return _env_singleton


def get_store() -> SettingsStore:
    """
    Возвращает singleton SettingsStore.

    При первом вызове создаёт хранилище поверх get_env(), при
    последующих — возвращает уже созданный объект.

    Returns:
        Единственный на процесс экземпляр SettingsStore.
    """
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = SettingsStore(get_env())
    return _store_singleton


def apply_env_vars() -> None:
    """
    Прописывает переменные окружения HuggingFace из настроек.

    Должна быть вызвана до первого импорта huggingface_hub,
    faster_whisper или ctranslate2 — иначе значения не подхватятся,
    и HF Hub будет работать в анонимном режиме с throttling.
    """
    env = get_env()
    if env.hf_token:
        os.environ["HF_TOKEN"] = env.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = env.hf_token
    if not env.hf_symlinks_warning:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    if not env.hf_telemetry:
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if env.hf_transfer:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(env.hf_download_timeout)


_repo_singleton = None

def get_repository():
    """
    Возвращает singleton JobRepository.

    Импорт JobRepository внутри функции — чтобы избежать циклических
    зависимостей между config.py и storage.py.

    Returns:
        Единственный на процесс экземпляр JobRepository.
    """
    global _repo_singleton
    if _repo_singleton is None:
        from .storage import JobRepository
        _repo_singleton = JobRepository(get_env().db_path)
    return _repo_singleton
