import argparse
import os
import sys
import time

# Добавляем корневую директорию проекта в путь
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import cv2
from dotenv import load_dotenv

from clients.plate_recognizer import PlateRecognizerClient, parse_plr_response


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Plate Recognizer on a local image",
        epilog="Примеры:\n"
        "  python scripts/test_plr.py tests/plate_frames/b7e52cbc-3302-4cea-bf92-409c5f4e1315.jpg\n"
        "  python scripts/test_plr.py debug_raw_crop.jpg\n"
        "  python scripts/test_plr.py tests/output/b7e52cbc-3302-4cea-bf92-409c5f4e1315_orig.jpg",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Path to input image (JPEG/PNG)")
    args = parser.parse_args()

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "app.env"))

    token = os.getenv("PLR_TOKEN", "").strip()
    base_url = os.getenv("PLR_BASE_URL", "https://api.platerecognizer.com/v1/plate-reader/")
    regions = os.getenv("PLR_REGIONS", "kz")
    timeout = float(os.getenv("PLR_TIMEOUT_SECONDS", "8"))

    if not token or token.lower() in ("your_token", "ваш_токен_от_platerecognizer.com", "your_plate_recognizer_token"):
        raise SystemExit(
            "PLR_TOKEN is empty or contains example value.\n"
            "Please set your real Plate Recognizer token in app.env:\n"
            "PLR_TOKEN=your_actual_token_here"
        )
    
    # Проверяем, что токен содержит только ASCII символы
    try:
        token.encode("ascii")
    except UnicodeEncodeError:
        raise SystemExit(
            "PLR_TOKEN contains non-ASCII characters.\n"
            "Please check your token in app.env - it should contain only ASCII characters."
        )

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.input}")

    client = PlateRecognizerClient(base_url=base_url, token=token, timeout_seconds=timeout)
    start = time.time()
    payload = __import__("asyncio").run(client.recognize(img, regions))
    latency = (time.time() - start) * 1000.0
    plate, score, candidates = parse_plr_response(payload)

    print(f"Plate: {plate}")
    print(f"Score: {score}")
    print(f"Candidates: {len(candidates)}")
    print(f"Latency: {latency:.1f} ms")


if __name__ == "__main__":
    main()
