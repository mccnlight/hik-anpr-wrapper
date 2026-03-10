ПОЛНОЦЕННЫЙ ТЕСТ (имитация камеры + весь процесс)

================================================================================
КОМАНДЫ ПО ПОРЯДКУ — КОПИРУЙ И ВСТАВЛЯЙ В КАЖДЫЙ ТЕРМИНАЛ
================================================================================

--- ТЕРМИНАЛ 1 (mock upstream, опционально) ---
cd c:\Snow\hik-anpr-wrapper
.\.venv\Scripts\Activate.ps1
python scripts/mock_upstream.py --port 9999

--- ТЕРМИНАЛ 2 (API, обязателен) ---
cd c:\Snow\hik-anpr-wrapper
.\.venv\Scripts\Activate.ps1
uvicorn api:app --host 0.0.0.0 --port 8000 --env-file app.env

--- ТЕРМИНАЛ 3 (тест, после того как API запустился) ---
cd c:\Snow\hik-anpr-wrapper
.\.venv\Scripts\Activate.ps1
python scripts/full_test.py --count 4 --pairing --snow-delay 0.3

================================================================================

Подготовка
----------
- В test_data/plate/ лежат 1.jpg … 4.jpg (фото номерной камеры).
- В test_data/snow/ лежат 1.jpg … 4.jpg (кадры снеговой камеры).
- В test_data/ лежат anpr_1.xml … anpr_4.xml (номера для событий).
- В app.env для теста с pairing должно быть: ENABLE_LINE_PAIRING=true.


ТРИ ТЕРМИНАЛА (подробно)
========================

Терминал 1 — Mock upstream (опционально, чтобы не слать на боевой сервер)
--------------------------------------------------------------------------
  cd c:\Snow\hik-anpr-wrapper
  .\.venv\Scripts\Activate.ps1
  python scripts/mock_upstream.py --port 9999

Если используешь mock: в app.env задай UPSTREAM_URL=http://127.0.0.1:9999/events


Терминал 2 — API
----------------
  cd c:\Snow\hik-anpr-wrapper
  .\.venv\Scripts\Activate.ps1
  uvicorn api:app --host 0.0.0.0 --port 8000 --env-file app.env

Дождись "Application startup complete" и "Uvicorn running on http://0.0.0.0:8000".


Терминал 3 — Отправка тестовых событий
--------------------------------------
  cd c:\Snow\hik-anpr-wrapper
  .\.venv\Scripts\Activate.ps1

Только номера (без снега):
  python scripts/full_test.py --count 4

Реалистичный тест (снег → 0.3 с → номер, номера из XML):
  python scripts/full_test.py --count 4 --pairing --snow-delay 0.3

С паузой между событиями и своим URL:
  python scripts/full_test.py --count 4 --pairing --delay 2 --base-url http://127.0.0.1:8000


Ожидание
--------
- Ответы 200 от API. В логах API — push-snow, затем hikvision, воркеры (YOLO по снегу, отправка upstream).
- Если запущен mock_upstream — в терминале 1 появятся входящие запросы.


Дополнительно
-------------
- run_pairing_test.py — проверка только логики очередей (snow/plate), без API.
- Без --pairing full_test.py шлёт только anpr.xml + plate; с --pairing сначала снег на /api/v1/test/push-snow, потом номерное событие.
