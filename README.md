# 🎙️ Whisper Transcriber

Веб-приложение для транскрибации аудио и видео с использованием локальных
моделей Whisper. Работает на вашей GPU, данные не уходят в облако.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/fastapi-0.115+-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/CUDA-12.x-76b900" alt="CUDA">
  <img src="https://img.shields.io/badge/license-MIT-yellow" alt="License">
</p>

---

## ✨ Возможности

- 🎙️ **Локальная транскрибация** — модель работает на вашей машине, файлы не покидают её
- 🚀 **GPU-ускорение** — CUDA + float16, оптимизация под RTX 40xx/50xx
- 📊 **Live-прогресс** — SSE-стрим событий: логи, прогресс, сегменты в реальном времени
- 🌍 **Мультиязычность** — русский, английский, немецкий, французский, испанский и др.
- ⚙️ **Runtime-настройки** — смена модели, языка, batch_size без перезапуска сервера
- 🧹 **Автоочистка** — старые задачи и осиротевшие файлы удаляются автоматически
- 📁 **Экспорт** — TXT, SRT (субтитры), JSON (структурированные данные)
- 📱 **Адаптивный UI** — комфортная работа на смартфонах и планшетах
- 🎨 **Тёмная и светлая темы**, переключение языка интерфейса
- 📚 **OpenAPI / Swagger** — полная документация API из коробки

---

## 📋 Требования

### Аппаратные

- **GPU** (рекомендуется): NVIDIA с поддержкой CUDA, минимум 6 ГБ VRAM
  - Для модели `large-v3` в float16 нужно ~4 ГБ VRAM
  - Для `large-v3` с `batch_size=16` — ~10 ГБ
- **CPU** (альтернатива): работает, но в 5–10 раз медленнее
- **RAM**: минимум 8 ГБ, рекомендуется 16+
- **Диск**: 10 ГБ под модели + место под загруженные файлы

### Программные

- **Python** 3.10+ (проверено на 3.12)
- **FFmpeg** в системном PATH — нужен для извлечения аудио из видео
- **NVIDIA драйвер** 535+ и CUDA Runtime 12.x (если используете GPU)

---

## 🚀 Установка

### 1. Клонирование

```bash
git clone https://github.com/your-repo/whisper-transcriber.git
cd whisper-transcriber
```

### 2. Виртуальное окружение

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Зависимости

```bash
pip install -r requirements.txt
```

Если нужен GPU-режим — убедитесь, что установлен PyTorch с CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4. FFmpeg

**Windows:** скачайте с https://www.gyan.dev/ffmpeg/builds/, распакуйте,
добавьте `bin/` в системный `PATH`.

**Linux:**

```bash
sudo apt install ffmpeg
```

**macOS:**

```bash
brew install ffmpeg
```

Проверка: `ffmpeg -version` должен вывести версию, а не «команда не найдена».

### 5. Конфигурация

Скопируйте шаблон и откройте его:

```bash
cp .env.example .env      # Linux / macOS
copy .env.example .env    # Windows
```

Заполните минимум два параметра:

```env
# Токен HuggingFace (получить: https://huggingface.co/settings/tokens)
# Тип токена: Read. Начинается с "hf_..."
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Куда складывать модели
# Структура: <MODELS_DIR>/<издатель>/<репозиторий>/
MODELS_DIR=G:\LLMs\models
```

Остальные параметры имеют разумные значения по умолчанию и меняются
в веб-интерфейсе.

### 6. Запуск

```bash
python run.py
```

Откройте в браузере:

- **http://127.0.0.1:8000** — веб-интерфейс
- **http://127.0.0.1:8000/docs** — Swagger API
- **http://127.0.0.1:8000/redoc** — ReDoc API

---

## 📖 Использование

### Веб-интерфейс

1. **Новая задача** — перетащите аудио или видеофайл в зону загрузки.
   Поддерживаются MP4, MKV, AVI, MOV, WebM, M4A, MP3, WAV, FLAC, OGG и др.
2. Выберите **модель** (от `tiny` до `large-v3`), **язык** (или `auto`)
   и **batch size**.
3. Нажмите **«Начать транскрибацию»**. Логи и прогресс появятся внизу
   в раздвижной консоли.
4. Результат отображается справа с таймкодами. Кнопки экспорта:
   `.txt`, `.srt`, `.json`, а также копирование в буфер.

### История

Все задачи сохраняются в памяти процесса. Если перезагрузить страницу во
время активной задачи, она **восстановится** — интерфейс подхватит состояние
с сервера и продолжит показывать прогресс.

Параметр `RETENTION_HOURS` в настройках задаёт срок хранения завершённых
задач. По умолчанию — 24 часа.

### Настройки

Меняются прямо в UI, сохраняются в `data/settings.json` и переживают
перезапуск сервера. В `.env` задаются только значения по умолчанию.

---

## 🔌 API

Полная документация — в **Swagger** по адресу `/docs`. Ниже — краткая
шпаргалка по основным эндпоинтам.

### Создание задачи

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F "file=@meeting.m4a" \
  -F "model=large-v3" \
  -F "language=ru"
```

Ответ:

```json
{"job_id": "a1b2c3d4e5f6", "status": "pending"}
```

### Подписка на прогресс (SSE)

```bash
curl -N http://127.0.0.1:8000/api/jobs/a1b2c3d4e5f6/stream
```

Поток событий:

```
data: {"type": "snapshot", "data": {...}}

data: {"type": "log", "entry": {"level": "info", "message": "..."}}

data: {"type": "progress", "value": 42, "message": "...", "stage": "transcribing"}

data: {"type": "segment", "segment": {"start": 1.23, "end": 4.56, "text": "..."}}

data: {"type": "done", "text": "...", "metadata": {...}}
```

### Результат

```bash
curl http://127.0.0.1:8000/api/jobs/a1b2c3d4e5f6/result
```

### Python-клиент

```python
import requests
import json

BASE = "http://127.0.0.1:8000"

with open("meeting.m4a", "rb") as f:
    r = requests.post(
        f"{BASE}/api/jobs",
        files={"file": f},
        data={"model": "large-v3", "language": "ru"},
    )
    job_id = r.json()["job_id"]

with requests.get(f"{BASE}/api/jobs/{job_id}/stream", stream=True) as r:
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        event = json.loads(line[6:])
        if event["type"] == "progress":
            print(f"{event['value']}% — {event.get('message', '')}")
        elif event["type"] == "done":
            print("\n" + event["text"])
            break
```

---

## ⚙️ Конфигурация

Все параметры задаются в `.env`. Ниже — основные.

### HuggingFace

| Переменная | По умолчанию | Описание |
|---|---|---|
| `HF_TOKEN` | `""` | Токен доступа (тип Read) для скачивания моделей |
| `HF_TRANSFER` | `true` | Ускорение загрузки через `hf_transfer` |
| `HF_DOWNLOAD_TIMEOUT` | `30` | Таймаут HTTP-запросов в секундах |

### Пути

| Переменная | По умолчанию | Описание |
|---|---|---|
| `MODELS_DIR` | `G:\LLMs\models` | Корневая папка для хранения моделей |
| `DATA_DIR` | `data` | Папка для загруженных файлов и настроек |

### Сервер

| Переменная | По умолчанию | Описание |
|---|---|---|
| `HOST` | `127.0.0.1` | Адрес прослушивания |
| `PORT` | `8000` | Порт |
| `LOG_LEVEL` | `info` | Уровень логирования uvicorn |
| `CORS_ORIGINS` | `*` | Разрешённые origins через запятую |

### Whisper

| Переменная | По умолчанию | Описание |
|---|---|---|
| `DEFAULT_MODEL` | `large-v3` | Модель по умолчанию |
| `DEFAULT_LANGUAGE` | `ru` | Язык по умолчанию |
| `DEFAULT_BATCH_SIZE` | `8` | Размер батча |
| `DEFAULT_VAD` | `true` | VAD-фильтрация |
| `DEFAULT_COMPUTE_TYPE` | `float16` | Точность вычислений |
| `DEFAULT_DEVICE` | `cuda` | Устройство (`cuda` / `cpu`) |

### Лимиты и очистка

| Переменная | По умолчанию | Описание |
|---|---|---|
| `MAX_UPLOAD_MB` | `4096` | Максимальный размер файла в МБ |
| `RETENTION_HOURS` | `24` | Сколько часов хранить завершённые задачи |
| `CLEANUP_INTERVAL_MIN` | `30` | Период фоновой очистки в минутах |

---

## 🧠 Модели

Все модели скачиваются автоматически при первом использовании в формате
**CTranslate2** (faster-whisper).

| Модель | Размер | VRAM (float16) | Скорость | Качество |
|---|---|---|---|---|
| `tiny` | 75 МБ | ~1 ГБ | ×32 | низкое |
| `base` | 145 МБ | ~1 ГБ | ×16 | базовое |
| `small` | 484 МБ | ~2 ГБ | ×6 | хорошее |
| `medium` | 1.5 ГБ | ~5 ГБ | ×2 | очень хорошее |
| `large-v2` | 3.1 ГБ | ~4 ГБ | ×1 | отличное |
| `large-v3` | 3.1 ГБ | ~4 ГБ | ×1 | лучшее |
| `large-v3-turbo` | 1.6 ГБ | ~2 ГБ | ×8 | почти как large-v3 |

**Совет:** для русского языка начните с `large-v3`. Если GPU не тянет —
попробуйте `large-v3-turbo`: скорость почти как у `medium`, качество близко
к `large-v3`.

Модели хранятся в `<MODELS_DIR>/<издатель>/<репозиторий>/`. Структура
LM Studio-подобная — если у вас уже есть модели оттуда, они подхватятся
автоматически.

---

## 🐛 Решение проблем

### `Library cublas64_12.dll is not found or cannot be loaded`

Windows не находит CUDA-библиотеки, потому что пакеты `nvidia-*` кладут их
в `site-packages/nvidia/*/bin/`, но не прописывают в PATH. Приложение
регистрирует эти пути автоматически через `os.add_dll_directory` в `run.py`
до импорта `ctranslate2`. Если ошибка всё равно появляется:

1. Проверьте, что установлены пакеты:

   ```bash
   pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
   ```

2. Убедитесь, что запускаете через `python run.py`, а не `python app/server.py`
   напрямую — иначе регистрация путей не выполнится.

3. Для RTX 50xx (Blackwell, sm_120) обязательно используйте
   `compute_type="float16"` — `int8` и `auto` могут не работать.

### `CUDA out of memory`

Уменьшите `batch_size` в настройках UI (например, с 8 до 4 или 2). Либо
выберите модель поменьше (`large-v3-turbo` вместо `large-v3`).

### Модель долго скачивается или виснет

Проверьте:

- HF-токен в `.env` заполнен и валиден
- Установлен `hf_transfer`: `pip install hf_transfer`
- Есть доступ к `huggingface.co` (иногда блокируется корпоративным firewall)

### Транскрибация зависает на 2%

Скорее всего, идёт скачивание модели — прогресс-бар должен расти от 2% до
30%. Если он стоит на месте больше минуты — смотрите логи в консоли
внизу экрана: там будет видно, на каком именно файле застряло.

### Английские слова в русской речи распознаются плохо

Уже включены по умолчанию:

- `condition_on_previous_text=True`
- `temperature=[0.0, 0.2, ..., 1.0]`
- Дефолтный `initial_prompt` с примерами технических терминов

Дополнительно:

- Добавьте специфичные термины в `hotwords` (настройки → advanced)
- Отредактируйте `initial_prompt` под ваш домен
- Используйте модель `large-v3` — она лучше справляется со смешанной речью

### Файлы накапливаются на диске

Проверьте `RETENTION_HOURS` в настройках. Фоновый поток очистки удаляет
завершённые задачи и осиротевшие файлы раз в `CLEANUP_INTERVAL_MIN` минут.

---

## 🏗️ Архитектура

```
whisper-transcriber/
├── run.py                  # Точка входа: env vars + CUDA DLL + uvicorn
├── requirements.txt
├── .env.example
├── README.md
│
├── app/
│   ├── __init__.py         # версия пакета
│   ├── config.py           # .env + runtime-настройки
│   ├── jobs.py             # Job, JobManager, очистка
│   ├── monitor.py          # CPU/RAM/GPU/диски
│   ├── transcribe.py       # воркер: скачивание + инференс
│   ├── i18n.py             # загрузчик переводов
│   └── server.py           # FastAPI: эндпоинты, SSE, статика
│
├── locales/
│   ├── ru.json
│   └── en.json
│
├── static/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── fonts/
│       └── material-symbols-rounded.woff2
│
└── data/                   # создаётся автоматически
    ├── uploads/            # загруженные файлы (автоочистка)
    └── settings.json       # runtime-настройки
```

### Поток данных

```
Клиент  →  POST /api/jobs (multipart)
          ↓
       [Файл сохраняется в data/uploads/]
          ↓
       [Job создаётся в памяти]
          ↓
       [Воркер запускается в daemon-потоке]
          ↓
Клиент  ←  GET /api/jobs/{id}/stream (SSE)
          ←  snapshot + log + progress + segment + done
          ↓
       [Результат доступен в Job.text и Job.metadata]
          ↓
Клиент  ←  GET /api/jobs/{id}/result (JSON)
```

Задачи живут **только в памяти** процесса — они не переживают перезапуск
сервера. Для постоянного хранения нужно добавить SQLite (см. раздел
«Идеи для развития»).

---

## 🛠️ Разработка

### Структура зависимостей

```
FastAPI  →  server.py  →  transcribe.py  →  faster-whisper
                              ↓                  ↓
                          jobs.py           ctranslate2
                              ↑                  ↓
                          monitor.py       CUDA / cuDNN
```

### Запуск в dev-режиме с автоперезагрузкой

```bash
uvicorn app.server:app --reload --host 127.0.0.1 --port 8000
```

Но помните: `run.py` выполняет критичную инициализацию (env vars,
CUDA DLL) до импорта приложения. Прямой запуск через `uvicorn` может
привести к проблемам с CUDA — используйте `python run.py`.

### Обновление локалей

Файлы `locales/*.json` — плоские словари `ключ → строка`. После правки
перезапустите сервер. Если добавили новый язык — он автоматически
появится в переключателе.

---

## 🗺️ Идеи для развития

- [ ] **Диаризация спикеров** через `pyannote.audio` — «кто говорил»
- [ ] **Суммаризация** через локальную LLM (Qwen 2.5, Llama 3.2)
- [ ] **Перевод** субтитров на другие языки (NLLB, MarianMT)
- [x] **Очередь задач** с FIFO вместо параллельного запуска
- [x] **SQLite** для сохранения истории между перезапусками
- [ ] **Live-транскрибация** с микрофона через MediaRecorder
- [ ] **Docker + docker-compose** для одной команды развёртывания
- [ ] **Аутентификация** через JWT или API-ключи
- [ ] **WebSocket** вместо SSE для двунаправленного канала
- [x] **Метрики Prometheus** — количество задач, средний RTF

---

## 📄 Лицензия

MIT — используйте, модифицируйте и распространяйте свободно.

---

## 🙏 Благодарности

- [OpenAI Whisper](https://github.com/openai/whisper) — оригинальная модель
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — оптимизация через CTranslate2
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) — быстрый движок инференса
- [FastAPI](https://fastapi.tiangolo.com/) — веб-фреймворк
- [Material Symbols](https://fonts.google.com/icons) — иконки от Google
