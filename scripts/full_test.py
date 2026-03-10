#!/usr/bin/env python3
"""
Полноценный тест: имитация прихода событий с камеры (multipart anpr.xml + detectionPicture.jpg).
Номера берутся из XML (anpr_1.xml … anpr_4.xml). При --pairing сначала шлётся снеговой кадр,
через доли секунды — номерное событие (реалистичный порядок для ENABLE_LINE_PAIRING).

Использование:
  1) Запусти API с ENABLE_LINE_PAIRING=true в app.env:
       uvicorn api:app --host 0.0.0.0 --port 8000 --env-file app.env
  2) (Опционально) Mock upstream: python scripts/mock_upstream.py --port 9999
  3) Тест только номеров:
       python scripts/full_test.py --count 4
  4) Реалистичный тест (снег + номер почти одновременно):
       python scripts/full_test.py --count 4 --pairing --snow-delay 0.3
"""
import argparse
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PLATE_DIR = PROJECT_ROOT / "test_data" / "plate"
TEST_SNOW_DIR = PROJECT_ROOT / "test_data" / "snow"
TEST_ANPR_DIR = PROJECT_ROOT / "test_data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Отправка тестовых событий ANPR на API (имитация камеры)")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Базовый URL API",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Количество событий (1–4 по файлам в test_data/plate и anpr_1..4.xml)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Пауза между запросами (сек)",
    )
    parser.add_argument(
        "--plate-dir",
        type=Path,
        default=TEST_PLATE_DIR,
        help="Папка с фото номерной камеры (1.jpg, 2.jpg, ...)",
    )
    parser.add_argument(
        "--snow-dir",
        type=Path,
        default=TEST_SNOW_DIR,
        help="Папка со снеговыми кадрами (1.jpg, 2.jpg, ...) для --pairing",
    )
    parser.add_argument(
        "--pairing",
        action="store_true",
        help="Реалистичный тест: сначала push-snow, через --snow-delay отправка номера (plate+anpr)",
    )
    parser.add_argument(
        "--snow-delay",
        type=float,
        default=0.3,
        help="Задержка (сек) между отправкой снега и номерного события при --pairing",
    )
    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}/api/v1/anpr/hikvision"
    push_snow_url = f"{args.base_url.rstrip('/')}/api/v1/test/push-snow"
    count = max(1, min(args.count, 4))

    for i in range(1, count + 1):
        plate_path = args.plate_dir / f"{i}.jpg"
        anpr_path = TEST_ANPR_DIR / f"anpr_{i}.xml"
        if not plate_path.is_file():
            print(f"[FULL_TEST] Нет файла {plate_path}")
            sys.exit(1)
        if not anpr_path.is_file():
            print(f"[FULL_TEST] Нет файла {anpr_path}")
            sys.exit(1)
        if args.pairing:
            snow_path = args.snow_dir / f"{i}.jpg"
            if not snow_path.is_file():
                print(f"[FULL_TEST] При --pairing нужен файл {snow_path}")
                sys.exit(1)

    print(f"[FULL_TEST] URL: {url}")
    print(f"[FULL_TEST] Событий: {count}, пауза между событиями: {args.delay} с")
    if args.pairing:
        print(f"[FULL_TEST] Режим pairing: snow -> sleep {args.snow_delay} с -> plate (номера из XML)")
    print()

    for i in range(1, count + 1):
        plate_path = args.plate_dir / f"{i}.jpg"
        anpr_path = TEST_ANPR_DIR / f"anpr_{i}.xml"

        if args.pairing:
            snow_path = args.snow_dir / f"{i}.jpg"
            with open(snow_path, "rb") as f:
                snow_bytes = f.read()
            files_snow = [("file", (f"{i}.jpg", snow_bytes, "image/jpeg"))]
            print(f"[FULL_TEST] Событие #{i}: отправка снега {snow_path.name} -> push-snow...")
            try:
                r = httpx.post(push_snow_url, files=files_snow, timeout=10.0)
                if r.status_code != 200:
                    print(f"[FULL_TEST]   push-snow: {r.status_code} {r.text}")
                    sys.exit(1)
                print(f"[FULL_TEST]   push-snow: OK")
            except Exception as e:
                print(f"[FULL_TEST]   Ошибка push-snow: {e}")
                sys.exit(1)
            time.sleep(args.snow_delay)

        with open(plate_path, "rb") as f:
            plate_bytes = f.read()
        with open(anpr_path, "rb") as f:
            anpr_bytes = f.read()

        files = [
            ("anpr.xml", ("anpr.xml", anpr_bytes, "application/xml")),
            ("detectionPicture.jpg", ("detectionPicture.jpg", plate_bytes, "image/jpeg")),
        ]
        print(f"[FULL_TEST] Отправка события #{i} (plate={plate_path.name}, anpr=anpr_{i}.xml, номера из XML)...")
        try:
            resp = httpx.post(url, files=files, timeout=30.0)
            print(f"[FULL_TEST]   Ответ: {resp.status_code} {resp.text[:200]}")
            if resp.status_code != 200:
                print(f"[FULL_TEST]   Ошибка: {resp.text}")
        except Exception as e:
            print(f"[FULL_TEST]   Ошибка запроса: {e}")
            sys.exit(1)

        if i < count:
            print(f"[FULL_TEST] Пауза {args.delay} с...")
            time.sleep(args.delay)

    print()
    print("[FULL_TEST] Все события отправлены. Проверь логи API и (при mock upstream) вывод mock_upstream.py.")


if __name__ == "__main__":
    main()
