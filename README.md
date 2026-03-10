# Unified ANPR + Snow Service (RU)

Сервис объединяет события двух камер: номерной (Hikvision ANPR) и снеговой (RTSP). Снеговая камера даёт фото кузова при пересечении линии, ANPR камера шлёт свой вебхук; сервис матчит события по времени, опционально вызывает Gemini, и отправляет единый multipart на внешний бэкенд.

## Архитектура
- `api.py`: FastAPI, эндпоинты `GET /`, `GET /health`, `POST /api/v1/anpr/hikvision` (основной ISAPI вебхук), `/anpr` (заглушка). При старте может поднять снеговой воркер.
- `snow_worker.py`: читает RTSP, детектит грузовики (YOLO), фиксирует пересечение линии (`LINE_*`), пушит событие с кадром в мерджер.
- `combined_merger.py`: матчит снег ↔ ANPR (любой порядок прихода) в окне `MERGE_WINDOW_SECONDS`, TTL `MERGE_TTL_SECONDS`, макс. возраст `MERGE_MAX_EVENT_AGE_SECONDS`. При матче вызывает Gemini (если есть ключ) и шлёт multipart на `UPSTREAM_URL`.
- Локальный ANPR/OCR удалён: номер берём от камеры или из Gemini.

## Переменные окружения (`app.env` пример)
```
UPSTREAM_URL=https://snowops-anpr-service.onrender.com/api/v1/anpr/events
PLATE_CAMERA_ID=camera-001

# merge timing
MERGE_WINDOW_SECONDS=20
MERGE_TTL_SECONDS=50
MERGE_MAX_EVENT_AGE_SECONDS=30.0
MERGE_REQUIRE_SNOW_MATCH=true
LOCAL_TZ_OFFSET_HOURS=5  # для event_time

# snow worker
ENABLE_SNOW_WORKER=true
# Очереди снег/номер (линия → snow_queue, камера → plate_queue, пары по FIFO)
ENABLE_LINE_PAIRING=true
# PAIRING_TIME_WINDOW_SECONDS=0  — только FIFO; >0 — пара допустима, если разница времени ≤ N сек
# PAIRING_SNOW_QUEUE_MAXLEN=30   PAIRING_PLATE_QUEUE_MAXLEN=30
SNOW_VIDEO_SOURCE_URL=rtsp://user:pass@host:port/Streaming/Channels/101
SNOW_CAMERA_ID=camera-snow
SNOW_YOLO_MODEL_PATH=yolov8n.pt
# Линия пересечения (0..1)
LINE_X1=0.120
LINE_Y1=0.300
LINE_X2=0.580
LINE_Y2=0.700
LINE_DIRECTION=forward  # any|forward|backward
SNOW_SHOW_WINDOW=true

# Gemini (только для распознавания номера; снег — локальный YOLO)
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
# Путь к обученной модели снега (truck+snow). После обучения укажи best.pt — см. TRAIN_AND_DEPLOY.md
# SNOW_YOLO_WEIGHTS=models/training_runs/snow_detection/weights/best.pt
```

## Поток событий

**Режим по умолчанию (без ENABLE_LINE_PAIRING):**  
1) Снеговая камера: воркер буферизует кадры, при пересечении линии отправляет фото в мерджер.  
2) ANPR камера шлёт multipart на `/api/v1/anpr/hikvision`. Сервис сразу вызывает `capture_snow_photo` (поиск кадра по ANPR-фото в буфере) и кладёт одно событие в очередь.  
3) Воркеры обрабатывают очередь: Gemini — номер, YOLO — снег, отправка на `UPSTREAM_URL`.

**Режим очередей (ENABLE_LINE_PAIRING=true):** параллельные потоки без блокировки камеры.  
1) При старте поднимается снеговой воркер; на потоке рисуется линия. При пересечении линии грузовиком кадр снега кладётся в **очередь снега** (кэш в памяти).  
2) Номерная камера шлёт событие с фото — оно кладётся в **очередь номеров**. Ответ камере 200 OK отдаётся сразу (без захвата снега).  
3) Пары образуются по порядку (FIFO): первый снег + первое номерное событие. Опционально можно включить окно по времени (`PAIRING_TIME_WINDOW_SECONDS`).  
4) Образованные пары попадают в общую очередь обработки; несколько воркеров параллельно делают Gemini + YOLO и отправку.  
Требуется `ENABLE_SNOW_WORKER=true` и `ENABLE_LINE_PAIRING=true`. В этом режиме снеговой воркер при пересечении линии пушит только в очередь сопоставления (не в мерджер).

## Что отправляем на внешний сервис (multipart)
- Поле `event` (JSON): `camera_id`, `event_time`, `plate`, `confidence`, `direction`, `lane`, `vehicle`, `snow_detected`, `detection_confidence`, `snow_volume_percentage/confidence` (для совместимости), `matched_snow`, `timestamp`, `event_id`, `camera_plate/confidence`, `plate_source`, `xml_event_type`.
- Поле `photos`: `detectionPicture.jpg`, `featurePicture.jpg`/`licensePlatePicture.jpg` (если были), `snowSnapshot.jpg` (если есть кадр снега).

## Настройка линии (угол и координаты)
Утилита `line_calibrator.py` помогает подобрать линию вручную (RTSP или статичное фото):
```bash
python line_calibrator.py --source rtsp://user:pass@host:port/Streaming/Channels/101 --output line_config.json
python line_calibrator.py --image sample.jpg --output line_config.json
```
Управление: ЛКМ ставит две точки; `D` — смена направления; `R` — сброс; `S` — сохранить; `Q`/`Esc` — выход. Координаты можно перенести в `app.env` (`LINE_*`).

## Запуск
```bash
python -m venv .venv
.\.venv\Scripts\activate   # или source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# старт на 8000 или нужном порту
uvicorn api:app --host 0.0.0.0 --port 8000 --env-file app.env
```
Если `ENABLE_SNOW_WORKER=true`, воркер стартует вместе с приложением. Остановка — Ctrl+C.

Обучение модели снега (YOLO, классы truck + snow): см. **TRAIN_AND_DEPLOY.md** и скрипт `train_yolo.py`.

## Тест сопоставления снег/номер (очереди)
Положите тестовые фото в две папки:
- `test_data/plate/` — кадры номерной камеры: **1.jpg**, **2.jpg**, **3.jpg** (порядок по номеру).
- `test_data/snow/` — кадры снеговой камеры: **1.jpg**, **2.jpg**, **3.jpg** (тот же порядок).

Запуск (из корня проекта, с паузой 2 сек между «событиями»):
```bash
python scripts/run_pairing_test.py
```
Опции: `--plate-dir`, `--snow-dir`, `--delay 2`. Скрипт пушит кадры в очереди, проверяет образование пар и выводит итог (ожидается пара 1↔1, 2↔2, 3↔3).

## Полноценный тест (имитация камеры + весь процесс)
1. **Только очереди:** `python scripts/run_pairing_test.py`
2. **(Опционально)** Mock upstream, чтобы не слать на боевой сервер: `python scripts/mock_upstream.py --port 9999`, в `app.env` задать `UPSTREAM_URL=http://127.0.0.1:9999/events`
3. **Запуск API:** `uvicorn api:app --host 0.0.0.0 --port 8000 --env-file app.env`
4. **Отправка трёх событий** (anpr_1..3.xml + plate 1..3.jpg): `python scripts/full_test.py` (опции: `--count 3`, `--delay 2`, `--base-url http://127.0.0.1:8000`)

Подробно: `test_data/README_FULL_TEST.txt`.

## Эндпоинты
- `GET /` и `GET /health` → `{"status": "ok"}`
- `POST /api/v1/anpr/hikvision` → ISAPI вебхук Hikvision (multipart с `anpr.xml` + изображениями или raw JPEG fallback). Триггерит матчинга и отправку upstream.
- `POST /anpr` → заглушка (оставлена для обратной совместимости).
