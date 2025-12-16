# Unified ANPR + Snow Service (RU)

Сервис объединяет события двух камер: номерной (Hikvision ANPR) и снеговой (RTSP). Поток такой: снеговая камера кладёт кадры в память, при приходе ANPR вебхука ищется предыдущее снеговое событие в окне, вызывается Gemini для оценки заполненности кузова и отправляется единый multipart на внешний сервис.

## Архитектура и потоки
- `api.py` (FastAPI): эндпоинты `GET /health`, `POST /anpr`, `POST /api/v1/anpr/hikvision`. Загружает модели ANPR, по старту может включить снежный воркер.
- `snow_worker.py`: фон, читает RTSP, детектит грузовики (YOLO), при движении в зоне сохраняет кадр в памяти и кладёт в очередь мерджера (без диска, без Gemini).
- `combined_merger.py`: хранит снеговые события в памяти (TTL + окно), при ANPR событии берёт ближайшее предыдущее снеговое, только тогда вызывает Gemini (если есть ключ) и шлёт единый multipart на `UPSTREAM_URL`.
- `modules/anpr.py`: детектор номера (YOLO) + OCR (PaddleOCR) + нормализация KZ шаблонов.

### Логика снег → номер
1) Воркер видит грузовик в центральной зоне, движение слева направо → кодирует кадр в JPEG, кладёт в очередь с `event_time`/`bbox` (без Gemini).
2) При ANPR вебхуке мерджер ищет предыдущее снеговое в окне `MERGE_WINDOW_SECONDS` (snow раньше, plate позже). Просроченные (`MERGE_TTL_SECONDS`) удаляются.
3) Если найдено: вызывает Gemini по `snowSnapshot` (обрезает по bbox), заполняет `snow_volume_percentage/confidence`, прикладывает `snowSnapshot.jpg`, `matched_snow=true`.
4) Если не найдено или нет ключа Gemini: ставит нули, `matched_snow` остаётся по факту наличия матча; снимок снега прикладывается только при матче.
5) События без валидного номера/уверенности пропускаются.

### Логика ANPR вебхука
- Путь 1 (multipart Hikvision): парсит `anpr.xml` (номер/время/уверенность), кадр `detectionPicture.jpg` прогоняется через свою модель. Если нет валидного номера/уверенности — пропуск. Если модель/камера не вернули номер, подставляется хардкод `747AO`, иначе отправляется реальный.
- Путь 2 (fallback JPEG в body): только модель ANPR. Аналогично: при отсутствии валидного номера/уверенности — пропуск; хардкод ставится только если модель не дала номер.
- Все события/пропуски пишутся в `hik_raws/detections.log`.

## Переменные окружения (`.env` пример)
```
UPSTREAM_URL=https://snowops-anpr-service.onrender.com/api/v1/anpr/events
PLATE_CAMERA_ID=camera-001

# merge timing
MERGE_WINDOW_SECONDS=30
MERGE_TTL_SECONDS=60

# snow worker
ENABLE_SNOW_WORKER=true
SNOW_VIDEO_SOURCE_URL=rtsp://user:pass@host:port/Streaming/Channels/101
SNOW_CAMERA_ID=camera-snow
SNOW_YOLO_MODEL_PATH=yolov8n.pt
SNOW_CENTER_ZONE_START_X=0.15
SNOW_CENTER_ZONE_END_X=0.85
SNOW_CENTER_ZONE_START_Y=0.0
SNOW_CENTER_ZONE_END_Y=1.0
SNOW_CENTER_LINE_X=0.5
SNOW_MIN_DIRECTION_DELTA=5
SNOW_STATIONARY_TIMEOUT_SECONDS=10.0
SNOW_SHOW_WINDOW=false

# merge timing (для фильтрации старых событий)
MERGE_MAX_EVENT_AGE_SECONDS=15.0

# Gemini (нужен только при мердже со снегом)
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
```

## Запуск
```bash
python -m venv .venv
.\.venv\Scripts\activate   # или source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

uvicorn api:app --host 0.0.0.0 --port 8000 --env-file .env
```
Если `ENABLE_SNOW_WORKER=true`, воркер стартует вместе с приложением. Остановка — Ctrl+C.

## Эндпоинты
- `GET /health` → `{"status": "ok"}`
- `POST /anpr` → `multipart/form-data` с полем `file` (JPEG/PNG), отдаёт JSON с номером.
- `POST /api/v1/anpr/hikvision` → вебхук Hikvision (multipart с `anpr.xml` + изображениями или raw JPEG fallback). Триггерит ANPR и мердж со снегом.

## Что уходит во внешний сервис (multipart на `UPSTREAM_URL`)
- Поле `event` (строка JSON). Ключи:
  - `camera_id` (`PLATE_CAMERA_ID`)
  - `event_time` (из XML или сейчас, RFC3339)
  - `plate`, `confidence`
  - `camera_plate`, `camera_confidence` (из XML, если были)
  - `model_plate`, `model_det_conf`, `model_ocr_conf`
  - `direction`, `lane`, `vehicle` (заглушка `{}`)
  - `timestamp` (время обработки)
  - `original_plate_test` (оригинальный номер до возможного хардкода, для отладки)
  - `matched_snow` (true/false)
  - При матче со снегом: `snow_volume_percentage`, `snow_volume_confidence`, `snow_gemini_raw`
  - При отсутствии матча: `snow_volume_percentage=0`, `snow_volume_confidence=0`, `matched_snow=false`
- Поле `photos` (несколько файлов):
  - `detectionPicture.jpg` — кадр ANPR (всегда при multipart Hikvision, при fallback — выдранный JPEG)
  - `featurePicture.jpg` — если пришла
  - `licensePlatePicture.jpg` — если пришла
  - `snowSnapshot.jpg` — если было совпавшее снеговое событие

## Данные и логи
- Логи вебхуков ANPR: `hik_raws/detections.log`
- Снимки снега на диск не пишутся (всё в памяти).

## Важные нюансы
- Очередь снега чистится по TTL при добавлении/мердже; при полном простое старые элементы останутся в памяти до следующего события.
- При отсутствии `GEMINI_API_KEY` снеговая часть ставится в нули, но при матче `matched_snow` остаётся true и `snowSnapshot.jpg` уходит.
- В `.env` нельзя хранить реальные ключи/RTSP в репозитории.
