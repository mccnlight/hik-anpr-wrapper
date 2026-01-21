# anpr.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import os
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from modules.ocr import PlateOCR
from limitations.plate_rules import (
    normalize_plate,
    normalize_primary_plate,
)



ImageType = Union[str, np.ndarray]


@dataclass
class DetectionResult:
    plate: Optional[str]
    det_conf: float
    ocr_conf: float
    bbox: Optional[Tuple[int, int, int, int]]


def preprocess_plate(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Жёсткая предобработка маленького номера:
    - серое изображение
    - CLAHE (контраст)
    - bilateral filter (шум)
    - ресайз вверх
    - адаптивный порог
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Увеличиваем картинку, чтобы OCRу было проще
    h, w = gray.shape[:2]
    scale = max(2.0, 240.0 / max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Выравниваем контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    
    # Очень слабое затемнение (уменьшение яркости на 5-8%)
    # Помогает при слишком ярких фотографиях для лучшей читаемости
    darken_factor = 0.93  # Уменьшаем яркость на 7% (очень слабо)
    clahe_img = cv2.convertScaleAbs(clahe_img, alpha=darken_factor, beta=0)
    
    clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

    # --- вместо bilateral + adaptiveThreshold делаем так ---

    # Чуть сглаживаем шум
    blur = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # Глобальный порог Otsu — обычно лучше на номерах, чем адаптивный истерик
    _, th = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Небольшая морфология: убираем одиночный шум и склеиваем дырки в символах
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Для PaddleOCR делаем 3 канала
    proc = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    # Отладочные сохранения (можешь выключить)
    cv2.imwrite("debug_raw_crop.jpg", img)
    cv2.imwrite("debug_proc_crop.jpg", proc)

    return proc, clahe_bgr


def _select_best_valid_plate(trials: list[tuple[str, str, Optional[str], float]]) -> tuple[Optional[str], float]:
    """
    trials: [(variant, raw, normalized_or_none, ocr_conf), ...]
    Выбираем валидный (по формату KZ) номер с максимальной ocr_conf.
    Если валидных нет — возвращаем (None, 0.0), чтобы не перебивать номер камеры.
    """
    best_plate = None
    best_conf = 0.0
    for _, _, norm_plate, ocr_conf in trials:
        if norm_plate is None:
            continue
        if best_plate is None or ocr_conf > best_conf:
            best_plate = norm_plate
            best_conf = ocr_conf
    return best_plate, best_conf


class ANPR:
    """
    Общий движок:
    - YOLO детектит номерной знак
    - вырезаем кроп
    - предобработка
    - PaddleOCR + нормализация под KZ
    """

    def __init__(self, yolo_weights: str = "runs/detect/train4/weights/best.pt") -> None:
        """
        yolo_weights – путь к весам детектора номера.
        ОБРАТИ ВНИМАНИЕ: если у тебя файл называется иначе,
        просто поправь путь.
        """
        self.yolo = YOLO(yolo_weights)
        self.ocr = PlateOCR()
        self.det_conf_thr = 0.15

    def _load_image(self, img: ImageType) -> np.ndarray:
        if isinstance(img, str):
            image = cv2.imread(img)
            if image is None:
                raise ValueError(f"Cannot read image from path: {img}")
            return image
        if isinstance(img, np.ndarray):
            return img
        raise TypeError("img must be str path or numpy.ndarray")

    def infer(self, img: ImageType) -> Dict[str, Any]:
        """
        Основной метод:
        - img: путь к файлу или numpy-картинка
        - возвращает dict для JSON-ответа API
        """
        image = self._load_image(img)
        h, w = image.shape[:2]

        # 1. Детекция номерного знака YOLO
        det_result = self.yolo(image, conf=self.det_conf_thr, verbose=False)[0]

        # 🔴 если ничего не нашли — логируем и сохраняем кадр
        if det_result.boxes is None or len(det_result.boxes) == 0:
            os.makedirs("debug_no_det", exist_ok=True)

            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            debug_name = f"no_det_{ts}.jpg"
            debug_path = os.path.join("debug_no_det", debug_name)

            cv2.imwrite(debug_path, image)
            print(f"[ANPR] no plate detected, saved: {debug_path}")

            return DetectionResult(
                plate=None,
                det_conf=0.0,
                ocr_conf=0.0,
                bbox=None,
            ).__dict__

        # Берём бокс с максимальной уверенностью
        boxes = det_result.boxes
        confs = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        best_box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        det_conf = float(confs[best_idx])

        x1, y1, x2, y2 = best_box.tolist()

        # Ограничиваем координаты границами изображения
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return DetectionResult(
                plate=None,
                det_conf=det_conf,
                ocr_conf=0.0,
                bbox=(x1, y1, x2, y2),
            ).__dict__

        plate_crop = image[y1:y2, x1:x2]

        # 2. ????????????? ????? (??? ????????: ?????? CLAHE ? ????????)
        proc_crop, clahe_crop = preprocess_plate(plate_crop)

        # 3. OCR ?? ????? ?????????, ???????? ???????? ????? KZ ? ????????? ????????????
        ocr_trials = []

        for variant_name, crop in (("clahe", clahe_crop), ("binary", proc_crop)):
            raw_plate, ocr_conf = self.ocr.recognize(crop)
            strict_plate = normalize_primary_plate(raw_plate)
            relaxed_plate = normalize_plate(raw_plate)  # для логов

            # 👉 ЛОГИРУЕМ каждый вариант
            print(
                f"[ANPR][{variant_name}] "
                f"raw='{raw_plate}' norm_relaxed='{relaxed_plate}' "
                f"norm_strict='{strict_plate}' "
                f"ocr_conf={ocr_conf:.3f} det_conf={det_conf:.3f} "
                f"bbox=({x1},{y1},{x2},{y2})"
            )

            ocr_trials.append((variant_name, raw_plate, strict_plate, ocr_conf))

        # Выбираем только валидные по строгому формату. Если нет валидных — вернем None,
        # чтобы не перебивать номер, присланный камерой.
        plate_final, ocr_conf_final = _select_best_valid_plate(ocr_trials)

        result = DetectionResult(
            plate=plate_final,
            det_conf=det_conf,
            ocr_conf=ocr_conf_final,
            bbox=(x1, y1, x2, y2),
        )

        return result.__dict__


def test_anpr(path: str) -> None:
    """
    Утилитный запуск для локальной проверки:
    python -m anpr path/to/image.jpg
    """
    engine = ANPR()
    res = engine.infer(path)
    print(res)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m anpr path/to/image.jpg")
        sys.exit(1)
    test_anpr(sys.argv[1])
