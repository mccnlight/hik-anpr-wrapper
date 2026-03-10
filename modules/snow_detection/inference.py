"""
YOLO inference for snow detection.
Classes: 0 = truck, 1 = snow.
Logic: detect truck bbox, detect snow bbox, verify snow is inside truck region -> snow_detected bool.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

# Классы в обученной модели (должны совпадать с dataset/data.yaml)
TRUCK_CLASS_ID = 0
SNOW_CLASS_ID = 1

# Минимальная доля пересечения bbox снега с bbox грузовика, чтобы считать "снег в кузове"
# УТОЧНЕНИЕ: при необходимости подстрой порог (0.0–1.0)
SNOW_IOU_THRESHOLD = float(os.getenv("SNOW_IOU_THRESHOLD", "0.2"))

# Минимальная уверенность детекции снега
SNOW_CONF_THRESHOLD = float(os.getenv("SNOW_CONF_THRESHOLD", "0.35"))

# Минимальная уверенность детекции грузовика (берём лучший bbox по conf)
TRUCK_CONF_THRESHOLD = float(os.getenv("SNOW_TRUCK_CONF_THRESHOLD", "0.25"))

_model = None
_model_lock = __import__("threading").Lock()


def _iou_box(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """IoU двух bbox (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# Корень проекта (hik-anpr-wrapper): относительно него задаётся SNOW_YOLO_WEIGHTS
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_weights_path(path: str) -> Optional[str]:
    """Путь к best.pt: абсолютный или относительно корня проекта."""
    if not path or not path.strip():
        return None
    path = path.strip()
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    if not os.path.isabs(path):
        resolved = os.path.join(_PROJECT_ROOT, path)
        if os.path.isfile(resolved):
            return resolved
    return path if os.path.isfile(path) else None


def load_model(weights_path: Optional[str] = None):
    """Ленивая загрузка YOLO-модели. Путь из SNOW_YOLO_WEIGHTS или аргумент (относительно корня проекта)."""
    global _model
    raw = weights_path or os.getenv("SNOW_YOLO_WEIGHTS", "")
    path = _resolve_weights_path(raw)
    if not path:
        print("[SNOW_DETECT] No weights: SNOW_YOLO_WEIGHTS unset or file missing (expected: models/training_runs/snow_detection/weights/best.pt)")
        return None
    with _model_lock:
        if _model is None:
            try:
                from ultralytics import YOLO
                _model = YOLO(path)
                print(f"[SNOW_DETECT] Loaded weights: {path}")
            except Exception as e:
                print(f"[SNOW_DETECT] Failed to load model {path}: {e}")
                return None
        return _model


def run_detection(
    image: np.ndarray,
    weights_path: Optional[str] = None,
) -> Tuple[bool, float]:
    """
    Детекция по кадру (BGR).
    1) Находим bbox грузовика (class 0).
    2) Находим bbox снега (class 1).
    3) Проверяем, что снег попадает в область грузовика (IoU или центр внутри).
    Возвращает (snow_detected: bool, detection_confidence: float).
    """
    model = load_model(weights_path)
    if model is None:
        return False, 0.0

    try:
        results = model(image, verbose=False)
    except Exception as e:
        print(f"[SNOW_DETECT] Inference error: {e}")
        return False, 0.0

    truck_boxes = []  # (x1,y1,x2,y2, conf)
    snow_boxes = []

    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            if cls_id == TRUCK_CLASS_ID and conf >= TRUCK_CONF_THRESHOLD:
                truck_boxes.append((x1, y1, x2, y2, conf))
            elif cls_id == SNOW_CLASS_ID and conf >= SNOW_CONF_THRESHOLD:
                snow_boxes.append((x1, y1, x2, y2, conf))

    if not truck_boxes:
        print("[SNOW_DETECT] No truck in frame (snow_detected=False)")
        return False, 0.0

    # Лучший грузовик по уверенности (или по площади — уточни при необходимости)
    truck_boxes.sort(key=lambda t: t[4], reverse=True)
    best_truck = truck_boxes[0]
    truck_bbox = best_truck[:4]

    for sx1, sy1, sx2, sy2, snow_conf in snow_boxes:
        snow_bbox = (sx1, sy1, sx2, sy2)
        iou = _iou_box(truck_bbox, snow_bbox)
        if iou >= SNOW_IOU_THRESHOLD:
            # Снег внутри/пересекается с грузовиком
            return True, float(snow_conf)

    print(f"[SNOW_DETECT] Truck found but no snow in zone (snow_boxes={len(snow_boxes)}, iou>={SNOW_IOU_THRESHOLD})")
    return False, 0.0


def run_detection_from_bytes(image_bytes: bytes) -> Tuple[bool, float]:
    """Удобная обёртка: изображение в байтах (JPEG) -> (snow_detected, confidence)."""
    import cv2
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return False, 0.0
    return run_detection(img)
