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

### Логика ANPR вебхука с Plate Recognizer
1. **Получение кадров**: При получении webhook от Hikvision камеры:
   - Парсится `anpr.xml` (номер/время/уверенность от камеры)
   - Кадр `detectionPicture.jpg` обрабатывается через локальную модель ANPR (YOLO + PaddleOCR)
   - Кадр с bbox сохраняется в кольцевой буфер (`FrameBuffer`) для выбора лучшего кадра
   - Также используются кадры `featurePicture.jpg` и `licensePlatePicture.jpg` (если пришли)

2. **Выбор лучшего кадра по резкости**:
   - Из буфера (последние 1.2 секунды, ~10 FPS) выбирается кадр с максимальным blur_score (Variance of Laplacian)
   - Сравниваются также кадры из `licensePlatePicture.jpg` и `featurePicture.jpg`
   - Выбирается ROI номера с максимальной резкостью

3. **Распознавание номера (приоритет)**:
   - **Plate Recognizer (cloud)**: Если `PLR_ENABLED=1` и есть токен, отправляется лучший ROI в Plate Recognizer API
     - Если `score >= PLR_MIN_SCORE` (по умолчанию 0.75) → используется результат PLR
     - Если `blur_score < BLUR_THRESHOLD` и `PLR_TRY_PREPROCESSED=1` → дополнительно отправляется предобработанный ROI
   - **Fallback на Gemini**: Если PLR не использован или score низкий → используется Gemini OCR (из `analyze_with_gemini`)
   - **Fallback на локальную модель**: Если Gemini не дал результат → используется результат локальной модели ANPR

4. **Отправка события**: Формируется JSON с полями `plr_plate`, `plr_score`, `plr_used`, `blur_score`, `buffer_frames` и отправляется на upstream.

- Путь 1 (multipart Hikvision): описан выше
- Путь 2 (fallback JPEG в body): только модель ANPR, аналогично
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

# Plate Recognizer (cloud ALPR) - повышение точности распознавания
# Документация: https://guides.platerecognizer.com/docs/snapshot/api-reference/
PLR_ENABLED=1                    # Включить Plate Recognizer (1/0)
PLR_TOKEN=YOUR_PLATE_RECOGNIZER_TOKEN  # Токен от Plate Recognizer (получить на platerecognizer.com)
PLR_BASE_URL=https://api.platerecognizer.com/v1/plate-reader/  # Cloud API endpoint
PLR_REGIONS=kz                   # Коды стран/регионов (kz=Kazakhstan, можно kz,ru)
PLR_TIMEOUT_SECONDS=8            # Таймаут запроса к PLR API
PLR_MIN_SCORE=0.75               # Минимальный score для использования результата PLR (0.0-1.0)
PLR_TRY_PREPROCESSED=0           # Пробовать предобработанный ROI если blur_score низкий (1/0)

# Выбор лучшего кадра по резкости
BLUR_THRESHOLD=60                # Порог резкости (Variance of Laplacian), ниже - кадр считается "мыльным"
FRAME_BUFFER_SECONDS=1.2          # Длительность буфера кадров (секунды)
FRAME_BUFFER_FPS=10              # Частота сохранения кадров в буфер (кадров/сек)
FRAME_BUFFER_MAX_W=1280          # Максимальная ширина кадра в буфере (для экономии памяти)
FRAME_BUFFER_MAX_H=720           # Максимальная высота кадра в буфере

# Отладка (сохранение кадров)
DEBUG_SAVE=0                     # Сохранять debug кадры (1/0)
DEBUG_SAVE_DIR=debug_plr         # Директория для debug кадров
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
  - `plate`, `confidence` - **финальный номер и уверенность** (от PLR, Gemini или локальной модели)
  - `plate_source` - источник номера: `"plr"`, `"gemini"` или `"model"`
  - `camera_plate`, `camera_confidence` (из XML, если были)
  - `model_plate`, `model_det_conf`, `model_ocr_conf` (от локальной модели ANPR)
  - **Plate Recognizer поля** (если `PLR_ENABLED=1`):
    - `plr_plate` - номер от Plate Recognizer
    - `plr_score` - уверенность PLR (0.0-1.0)
    - `plr_used` - использован ли PLR (true/false)
    - `plr_latency_ms` - время ответа PLR API (мс)
    - `plr_candidates` - альтернативные варианты номера от PLR
  - **Выбор лучшего кадра**:
    - `blur_score` - резкость выбранного кадра (Variance of Laplacian)
    - `buffer_frames` - количество кадров в буфере на момент события
  - `direction`, `lane`, `vehicle` (заглушка `{}`)
  - `timestamp` (время обработки)
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

## Тестирование Plate Recognizer

### Локальный тест
```bash
# Тест на реальном изображении из проекта
python scripts/test_plr.py tests/plate_frames/b7e52cbc-3302-4cea-bf92-409c5f4e1315.jpg

# Или используйте кроп номера
python scripts/test_plr.py debug_raw_crop.jpg

# Или оригинальное изображение
python scripts/test_plr.py tests/output/b7e52cbc-3302-4cea-bf92-409c5f4e1315_orig.jpg

# Ожидаемый вывод:
# Plate: 035AL115
# Score: 0.92
# Candidates: 3
# Latency: 234.5 ms
```

### Проверка работы в production
1. Включите `DEBUG_SAVE=1` для сохранения debug кадров
2. Проверьте логи: должны быть строки `[PLR] ✅ Used:` или `[PLR] ⚠️ Fallback`
3. Проверьте поля `plr_used`, `plr_score`, `blur_score` в `hik_raws/detections.log`

## Важные нюансы
- Очередь снега чистится по TTL при добавлении/мердже; при полном простое старые элементы останутся в памяти до следующего события.
- При отсутствии `GEMINI_API_KEY` снеговая часть ставится в нули, но при матче `matched_snow` остаётся true и `snowSnapshot.jpg` уходит.
- **Plate Recognizer**: Если `PLR_ENABLED=0` или нет токена, используется только fallback на Gemini/локальную модель.
- **Выбор лучшего кадра**: Буфер заполняется кадрами из webhook (detectionPicture.jpg). Для непрерывного потока нужен отдельный RTSP worker (не реализован, используется только webhook).
- **Память**: Буфер ограничен по размеру (FRAME_BUFFER_SECONDS × FRAME_BUFFER_FPS кадров, каждый до 1280×720).
- В `.env` нельзя хранить реальные ключи/RTSP в репозитории.
