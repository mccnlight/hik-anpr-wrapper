import os
import pathlib
import sys
from typing import Iterable, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / "app.env")

from modules.upscaler import upscale_jpeg_bytes
from modules.blur import variance_of_laplacian
from modules.preprocess_ocr import preprocess_for_ocr

try:
    from modules.upscale_hat import HATUpscaler
except Exception:
    HATUpscaler = None


TESTS_DIR = pathlib.Path(__file__).resolve().parent
INPUT_DIR = TESTS_DIR / "plate_frames"
OUTPUT_DIR = TESTS_DIR / "output"


def _decode_jpeg(data: bytes) -> np.ndarray | None:
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _encode_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("failed to encode jpeg")
    return buf.tobytes()


def _iter_images(folder: pathlib.Path) -> Iterable[pathlib.Path]:
    for ext in (".jpg", ".jpeg", ".png"):
        yield from folder.glob(f"*{ext}")


def _save_pair(base_name: str, original: bytes, upscaled: bytes) -> Tuple[pathlib.Path, pathlib.Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    orig_path = OUTPUT_DIR / f"{base_name}_orig.jpg"
    up_path = OUTPUT_DIR / f"{base_name}_upscaled.jpg"
    orig_path.write_bytes(original)
    up_path.write_bytes(upscaled)
    return orig_path, up_path


def _autocrop_plate(photo_bytes: bytes) -> bytes | None:
    img = _decode_jpeg(photo_bytes)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h < 10 or w < 10:
        return None

    scale = 2.0 if max(h, w) < 200 else 1.0
    if scale != 1.0:
        img_scaled = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    else:
        img_scaled = img

    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if not contours:
        return None

    img_area = th.shape[0] * th.shape[1]
    best = None
    best_score = 0.0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < img_area * 0.08:
            continue
        ratio = cw / max(1, ch)
        if ratio < 2.0 or ratio > 8.0:
            continue
        score = area * (1.0 - abs(ratio - 3.5) / 3.5)
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch)

    if best is None:
        return None

    x, y, cw, ch = best
    pad = int(max(cw, ch) * 0.08)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_scaled.shape[1], x + cw + pad)
    y2 = min(img_scaled.shape[0], y + ch + pad)
    crop = img_scaled[y1:y2, x1:x2]

    if scale != 1.0:
        crop = cv2.resize(crop, (int(crop.shape[1] / scale), int(crop.shape[0] / scale)), interpolation=cv2.INTER_AREA)

    return _encode_jpeg(crop)


def simulate_camera_event() -> None:
    if not INPUT_DIR.exists():
        raise RuntimeError(f"input dir not found: {INPUT_DIR}")

    files = list(_iter_images(INPUT_DIR))
    if not files:
        raise RuntimeError(f"no images in {INPUT_DIR}")

    scale = int(os.getenv("UPSCALE_FACTOR", "2"))
    hat_enabled = os.getenv("ENABLE_HAT_UPSCALE", "false").lower() in ("1", "true", "yes", "on")
    hat_upscaler = HATUpscaler() if hat_enabled and HATUpscaler is not None else None
    for path in files:
        raw = path.read_bytes()
        # Приводим к JPEG, чтобы сравнение было корректным
        img = _decode_jpeg(raw)
        if img is None:
            print(f"[SKIP] cannot decode: {path.name}")
            continue
        h, w = img.shape[:2]
        original_jpeg = _encode_jpeg(img)
        if os.getenv("PLATE_AUTOCROP_ENABLE", "true").lower() == "true":
            auto_crop = _autocrop_plate(original_jpeg)
            if auto_crop:
                original_jpeg = auto_crop
        if hat_enabled and hat_upscaler is None:
            img_bgr = _decode_jpeg(original_jpeg)
            if img_bgr is None:
                print(f"[SKIP] cannot decode jpeg for fallback: {path.name}")
                continue
            h, w = img_bgr.shape[:2]
            up = cv2.resize(img_bgr, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
            upscaled = _encode_jpeg(up)
            print("[HAT] Fallback upscale used (HAT not available)")
        elif hat_upscaler is not None:
            img_bgr = _decode_jpeg(original_jpeg)
            if img_bgr is None:
                print(f"[SKIP] cannot decode jpeg for HAT: {path.name}")
                continue
            enhanced = hat_upscaler.enhance(img_bgr)
            upscaled = _encode_jpeg(enhanced)
        else:
            upscaled = upscale_jpeg_bytes(original_jpeg, scale=scale)
        up_img = _decode_jpeg(upscaled)
        if up_img is not None:
            uh, uw = up_img.shape[:2]
        else:
            uh, uw = 0, 0
        
        base = path.stem
        
        # Проверка качества: blur_score до и после upscale + предобработка
        original_img = _decode_jpeg(original_jpeg)
        preprocessed = None
        if original_img is not None and up_img is not None:
            original_blur = variance_of_laplacian(original_img)
            upscaled_blur = variance_of_laplacian(up_img)
            improvement_upscale = ((upscaled_blur - original_blur) / max(original_blur, 1.0)) * 100
            
            # Применяем предобработку к upscaled изображению (как в test_image_quality.py)
            preprocessed = preprocess_for_ocr(up_img)
            preprocessed_blur = variance_of_laplacian(preprocessed)
            improvement_preprocess = ((preprocessed_blur - upscaled_blur) / max(upscaled_blur, 1.0)) * 100
            total_improvement = ((preprocessed_blur - original_blur) / max(original_blur, 1.0)) * 100
            
            print(f"\n🔍 Качество изображения:")
            print(f"   Blur Score исходного: {original_blur:.2f}")
            print(f"   Blur Score после upscale: {upscaled_blur:.2f}")
            print(f"   Blur Score после предобработки: {preprocessed_blur:.2f}")
            
            if improvement_upscale > 0:
                print(f"   ✅ Улучшение от upscale: +{improvement_upscale:.1f}%")
            else:
                print(f"   ⚠️  Изменение от upscale: {improvement_upscale:.1f}%")
            
            if improvement_preprocess > 0:
                print(f"   ✅ Улучшение от предобработки: +{improvement_preprocess:.1f}%")
            else:
                print(f"   ⚠️  Изменение от предобработки: {improvement_preprocess:.1f}%")
            
            if total_improvement > 0:
                print(f"   ✅ Общее улучшение: +{total_improvement:.1f}%")
            
            # Оценка качества
            def get_quality_label(score: float) -> str:
                if score < 30:
                    return "🔴 Низкое"
                elif score < 60:
                    return "🟡 Среднее"
                else:
                    return "🟢 Хорошее"
            
            orig_quality = get_quality_label(original_blur)
            up_quality = get_quality_label(upscaled_blur)
            pre_quality = get_quality_label(preprocessed_blur)
            
            print(f"   Оценка: {orig_quality} -> {up_quality} -> {pre_quality}")
            
            # Сохраняем также предобработанную версию
            if preprocessed is not None:
                preprocessed_path = OUTPUT_DIR / f"{base}_preprocessed.jpg"
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(preprocessed_path), preprocessed)
                print(f"   💾 Предобработанная версия: {preprocessed_path.name}")
        orig_path, up_path = _save_pair(base, original_jpeg, upscaled)
        print(
            f"\n[OK] {path.name}: {w}x{h} -> {uw}x{uh} | "
            f"{orig_path.name} ({orig_path.stat().st_size} bytes), "
            f"{up_path.name} ({up_path.stat().st_size} bytes)"
        )


if __name__ == "__main__":
    simulate_camera_event()
