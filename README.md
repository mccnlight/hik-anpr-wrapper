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

# Gemini (для анализа снега/номера)
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
```

## Поток событий
1) Снеговая камера: воркер буферизует кадры, при пересечении линии грузовиком отправляет фото + метаданные в мерджер.
2) ANPR камера (Hikvision) шлёт multipart на `/api/v1/anpr/hikvision` с `anpr.xml` и фотографиями. Мы сразу пытаемся захватить кадр снега из буфера.
3) `EventMerger` матчит снег ↔ ANPR по времени (любой порядок), ждёт «поздний снег» до `WAIT_FOR_SNOW_SECONDS` (если задан, иначе окно). При матче вызывает Gemini (если есть ключ), собирает итоговый `event` и отправляет multipart на `UPSTREAM_URL`.
4) `event_time` всегда текущее локальное время (`LOCAL_TZ_OFFSET_HOURS`).
5) Если `MERGE_REQUIRE_SNOW_MATCH=true` — без снега не отправляем; иначе шлём с `matched_snow=false`.

## Что отправляем на внешний сервис (multipart)
- Поле `event` (JSON): `camera_id`, `event_time` (локальное ISO), `plate` (или `UNKNOWN`), `confidence`, `direction`, `lane`, `vehicle`, `snow_volume_percentage/confidence`, `matched_snow`, `timestamp`, `event_id`, `camera_plate/confidence`, `plate_source`, `xml_event_type`.
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

## Эндпоинты
- `GET /` и `GET /health` → `{"status": "ok"}`
- `POST /api/v1/anpr/hikvision` → ISAPI вебхук Hikvision (multipart с `anpr.xml` + изображениями или raw JPEG fallback). Триггерит матчинга и отправку upstream.
- `POST /anpr` → заглушка (оставлена для обратной совместимости).
