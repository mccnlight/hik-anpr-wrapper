#!/usr/bin/env python3
"""
Тест сопоставления снег/номер по очередям (event_pairing).

Использование:
  python scripts/run_pairing_test.py
  python scripts/run_pairing_test.py --plate-dir test_data/plate --snow-dir test_data/snow --delay 2

В папках plate и snow лежат фото с именами по порядку: 1.jpg, 2.jpg, 3.jpg (или .png).
Скрипт по очереди (с паузой --delay сек) пушит кадр снега и событие номеров с одним и тем же индексом,
проверяет try_match() и выводит, какие пары образовались. Ожидается: (1,1), (2,2), (3,3).
"""
import argparse
import re
import sys
import time
from pathlib import Path

# Корень проекта = родитель папки scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from event_pairing import push_snow, push_plate, try_match, queue_sizes


def _number_from_name(path: Path) -> int | None:
    """Из имени файла извлекаем число: 1.jpg -> 1, 02.png -> 2."""
    stem = path.stem
    m = re.match(r"^(\d+)$", stem)
    if m:
        return int(m.group(1))
    return None


def _collect_images(dir_path: Path) -> list[tuple[int, Path]]:
    """Собирает (номер, путь) по всем изображениям в папке, сортировка по номеру."""
    if not dir_path.is_dir():
        return []
    out = []
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            n = _number_from_name(p)
            if n is not None:
                out.append((n, p))
    out.sort(key=lambda x: x[0])
    return out


def _read_image_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Тест очередей снег/номер с задержкой между событиями")
    parser.add_argument(
        "--plate-dir",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "plate",
        help="Папка с тестовыми фото номерной камеры (1.jpg, 2.jpg, ...)",
    )
    parser.add_argument(
        "--snow-dir",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "snow",
        help="Папка с тестовыми кадрами снеговой камеры (1.jpg, 2.jpg, ...)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Пауза в секундах между «событиями» (пара снег+номер)",
    )
    args = parser.parse_args()

    plate_list = _collect_images(args.plate_dir)
    snow_list = _collect_images(args.snow_dir)

    if not plate_list:
        print(f"[TEST] В папке номеров нет подходящих файлов (ожидаются 1.jpg, 2.jpg, ...): {args.plate_dir}")
        sys.exit(1)
    if not snow_list:
        print(f"[TEST] В папке снега нет подходящих файлов (ожидаются 1.jpg, 2.jpg, ...): {args.snow_dir}")
        sys.exit(1)

    # Общие индексы
    plate_nums = {n for n, _ in plate_list}
    snow_nums = {n for n, _ in snow_list}
    common = sorted(plate_nums & snow_nums)
    if not common:
        print("[TEST] Нет общих номеров в двух папках. Нумерация должна совпадать (1, 2, 3, ...).")
        sys.exit(1)

    plate_by_num = {n: p for n, p in plate_list}
    snow_by_num = {n: p for n, p in snow_list}

    print(f"[TEST] Номерные кадры: {len(plate_list)}, снеговые: {len(snow_list)}, общие индексы: {common}")
    print(f"[TEST] Задержка между событиями: {args.delay} с")
    print()

    pairs_formed: list[tuple[str, int, str]] = []  # (event_id, snow_bytes_len, plate_value)
    pair_index = 0

    for idx in common:
        pair_index += 1
        snow_path = snow_by_num[idx]
        plate_path = plate_by_num[idx]
        snow_bytes = _read_image_bytes(snow_path)
        plate_bytes = _read_image_bytes(plate_path)

        # Минимальный event_data для push_plate (мерджер не нужен для проверки сопоставления)
        event_data = {
            "camera_id": "test-camera",
            "event_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_id": f"test-{idx}",
            "plate": f"TEST_{idx}",
            "confidence": 0.0,
            "direction": "unknown",
            "lane": 0,
            "vehicle": {},
        }

        push_snow(snow_bytes)
        print(f"  [TEST] Push snow #{idx} ({len(snow_bytes)} bytes)")

        push_plate(
            event_data=event_data,
            detection_bytes=plate_bytes,
            feature_bytes=None,
            license_bytes=None,
            plate_photo_1=plate_bytes,
            plate_photo_2=None,
            camera_plate_for_gemini=f"TEST_{idx}",
            main_plate=f"TEST_{idx}",
            merger=None,
        )
        print(f"  [TEST] Push plate #{idx} ({len(plate_bytes)} bytes)")

        # Сразу проверяем пару
        while True:
            pair = try_match()
            if pair is None:
                break
            snow_b, plate_item = pair
            ev_id = plate_item.event_data.get("event_id", "?")
            plate_val = plate_item.event_data.get("plate", "?")
            pairs_formed.append((ev_id, len(snow_b), plate_val))
            print(f"  [TEST] Paired: snow {len(snow_b)} bytes + plate '{plate_val}' (event_id={ev_id})")

        ns, np_ = queue_sizes()
        print(f"  [TEST] Queue sizes: snow={ns}, plate={np_}")

        if pair_index < len(common):
            print(f"  [TEST] Sleep {args.delay}s ...")
            time.sleep(args.delay)

    print()
    print("--- Итог ---")
    print(f"Образовано пар: {len(pairs_formed)}")
    for i, (ev_id, snow_len, plate_val) in enumerate(pairs_formed, 1):
        print(f"  Пара {i}: event_id={ev_id}, snow {snow_len} bytes, plate={plate_val}")
    ns, np_ = queue_sizes()
    print(f"В очередях после теста: snow={ns}, plate={np_}")

    if len(pairs_formed) == len(common):
        print("\n[TEST] OK: количество пар совпадает с количеством общих индексов.")
    else:
        print(f"\n[TEST] Ожидалось пар: {len(common)}, получено: {len(pairs_formed)}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
