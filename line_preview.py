"""
Простой превью-скрипт для настройки одной линии на одном RTSP-потоке.
- Линию можно ставить под углом (две точки в нормированных координатах).
- Направление задаётся стрелкой (A -> B), можно переключать.
- Значения выводятся в консоль и при необходимости сохраняются в line_config.json.

Управление:
  Q / Esc   — выйти (печатает текущие значения).
  Tab / C   — переключить выбранную точку (A или B).
  Space     — переключить направление стрелки.
  W/A/S/D   — сдвиг выбранной точки (малый шаг 0.005).
  I/J/K/L   — сдвиг выбранной точки (крупный шаг 0.02).
  R         — перечитать app.env (RTSP + координаты, если заданы).
  P         — вывести значения в консоль.
  S         — сохранить конфиг в line_config.json (pixels + normalized + direction).

Переменные окружения (app.env):
  SNOW_VIDEO_SOURCE_URL   — RTSP-поток (обязателен).
  LINE_X1, LINE_Y1, LINE_X2, LINE_Y2 — нормированные координаты точки A/B (0..1). Необязательны.
  LINE_DIRECTION          — "forward" (A->B) или "backward" (B->A), по умолчанию forward.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np


# --- env loader ---
def _load_env_vars() -> None:
    try:
        from dotenv import load_dotenv

        env_path = os.path.join(os.path.dirname(__file__), "app.env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            print(f"[LINE-PREVIEW] Loaded environment from: {env_path}")
        else:
            print(f"[LINE-PREVIEW] WARNING: app.env not found at {env_path}, using system env vars")
    except ImportError:
        print("[LINE-PREVIEW] WARNING: python-dotenv not installed, using system env vars only")


_load_env_vars()


# --- settings ---
RTSP_URL = os.getenv("SNOW_VIDEO_SOURCE_URL", "").strip().strip('"')

DEFAULT_LINE = (0.1, 0.8, 0.9, 0.8)  # A(x1,y1) -> B(x2,y2)
LINE_X1 = float(os.getenv("LINE_X1", str(DEFAULT_LINE[0])))
LINE_Y1 = float(os.getenv("LINE_Y1", str(DEFAULT_LINE[1])))
LINE_X2 = float(os.getenv("LINE_X2", str(DEFAULT_LINE[2])))
LINE_Y2 = float(os.getenv("LINE_Y2", str(DEFAULT_LINE[3])))

# шаги перемещения
STEP_SMALL = 0.005
STEP_BIG = 0.02

# Настройки детекции для предпросмотра
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CUSTOM_WEIGHTS_RAW = os.getenv("SNOW_YOLO_WEIGHTS", "").strip()
_FALLBACK_MODEL_RAW = os.getenv("SNOW_YOLO_MODEL_PATH", "yolov8n.pt").strip()
_USE_CUSTOM_MODEL = bool(_CUSTOM_WEIGHTS_RAW)

TRUCK_CLASS_ID = int(os.getenv("SNOW_TRUCK_CLASS_ID", "0" if _USE_CUSTOM_MODEL else "7"))
SNOW_CLASS_ID = int(os.getenv("SNOW_SNOW_CLASS_ID", "1"))
YOLO_MODEL_PATH = _CUSTOM_WEIGHTS_RAW or _FALLBACK_MODEL_RAW
PREVIEW_MIN_CONF = 0.25  # Мягкий порог для предпросмотра
PREVIEW_MIN_AREA = 500   # Минимальная площадь
PREVIEW_MIN_W = 20       # Минимальная ширина
PREVIEW_MIN_H = 20       # Минимальная высота
SQUARE_SCALE = float(os.getenv("SNOW_SQUARE_SCALE", "1.2"))
SQUARE_MIN_SIZE = int(os.getenv("SNOW_SQUARE_MIN_SIZE", "60"))
PREVIEW_OVERLAP_IOU = float(os.getenv("SNOW_PREVIEW_OVERLAP_IOU", "0.05"))
PREVIEW_MIN_SNOW_IN_TRUCK = float(os.getenv("SNOW_PREVIEW_MIN_SNOW_IN_TRUCK", "0.30"))
PREVIEW_DEDUP_IOU = float(os.getenv("SNOW_PREVIEW_DEDUP_IOU", "0.45"))
PREVIEW_MODEL_IOU = float(os.getenv("SNOW_PREVIEW_MODEL_IOU", "0.45"))

# Глобальная переменная для YOLO модели
_yolo_model = None


@dataclass
class LineState:
    x1: float
    y1: float
    x2: float
    y2: float

    def clamp(self) -> None:
        self.x1 = min(1.0, max(0.0, self.x1))
        self.y1 = min(1.0, max(0.0, self.y1))
        self.x2 = min(1.0, max(0.0, self.x2))
        self.y2 = min(1.0, max(0.0, self.y2))

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


def _reload_env(state: LineState) -> None:
    _load_env_vars()
    state.x1 = float(os.getenv("LINE_X1", str(state.x1)))
    state.y1 = float(os.getenv("LINE_Y1", str(state.y1)))
    state.x2 = float(os.getenv("LINE_X2", str(state.x2)))
    state.y2 = float(os.getenv("LINE_Y2", str(state.y2)))
    state.clamp()
    print(
        f"[LINE-PREVIEW] Reloaded line: "
        f"A=({state.x1:.3f},{state.y1:.3f}) B=({state.x2:.3f},{state.y2:.3f})"
    )


def _get_yolo_model():
    """Ленивая инициализация YOLO модели"""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            model_path = YOLO_MODEL_PATH
            if model_path and not os.path.isabs(model_path):
                candidate = os.path.join(_PROJECT_ROOT, model_path)
                if os.path.exists(candidate):
                    model_path = candidate
            if os.path.exists(model_path):
                _yolo_model = YOLO(model_path)
                print(f"[LINE-PREVIEW] YOLO model loaded: {model_path}")
            else:
                print(f"[LINE-PREVIEW] WARNING: YOLO model not found at {YOLO_MODEL_PATH}")
        except ImportError:
            print("[LINE-PREVIEW] WARNING: ultralytics not installed, truck detection disabled")
        except Exception as e:
            print(f"[LINE-PREVIEW] ERROR: Failed to load YOLO model: {e}")
    return _yolo_model


def _detect_for_preview(frame: np.ndarray, model) -> list:
    """Детектирует объекты для предпросмотра (truck/snow для custom-модели)."""
    if model is None:
        return []
    
    detections = []
    preview_vehicle_classes = [TRUCK_CLASS_ID, 5, 2]
    
    try:
        results = model(frame, verbose=False, conf=PREVIEW_MIN_CONF, iou=PREVIEW_MODEL_IOU)
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                if _USE_CUSTOM_MODEL:
                    # Для обученной модели ожидаем truck=0 и snow=1.
                    if cls_id not in (TRUCK_CLASS_ID, SNOW_CLASS_ID) or conf < PREVIEW_MIN_CONF:
                        continue
                else:
                    # Для базовой COCO-модели показываем крупный транспорт.
                    if cls_id not in preview_vehicle_classes or conf < PREVIEW_MIN_CONF:
                        continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                w = x2 - x1
                h = y2 - y1
                area = w * h
                
                if area < PREVIEW_MIN_AREA or w < PREVIEW_MIN_W or h < PREVIEW_MIN_H:
                    continue

                label = "snow" if _USE_CUSTOM_MODEL and cls_id == SNOW_CLASS_ID else "truck"
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'conf': conf,
                    'class_id': cls_id,
                    'label': label,
                })
    except Exception as e:
        print(f"[LINE-PREVIEW] Error in truck detection: {e}")
        return []
    
    return detections


def _iou_box(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _dedupe_by_iou(detections: list, iou_thr: float) -> list:
    """Убирает дубли боксов одного класса (оставляет более уверенный)."""
    if not detections:
        return []
    ordered = sorted(detections, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
    kept = []
    for det in ordered:
        box = det.get("bbox")
        if not box:
            continue
        cls_id = det.get("class_id")
        duplicate = False
        for k in kept:
            if k.get("class_id") != cls_id:
                continue
            if _iou_box(box, k.get("bbox")) >= iou_thr:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def _intersection_ratio(inner_box: Tuple[int, int, int, int], outer_box: Tuple[int, int, int, int]) -> float:
    """Доля inner_box, покрытая outer_box: intersection_area / area(inner_box)."""
    ix = _iou_box(inner_box, outer_box)
    if ix <= 0.0:
        return 0.0
    # Восстанавливаем через явное пересечение, чтобы не зависеть от IoU.
    ax1, ay1, ax2, ay2 = inner_box
    bx1, by1, bx2, by2 = outer_box
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    inner_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / float(inner_area)


def _center_inside(inner_box: Tuple[int, int, int, int], outer_box: Tuple[int, int, int, int]) -> bool:
    cx = (inner_box[0] + inner_box[2]) / 2.0
    cy = (inner_box[1] + inner_box[3]) / 2.0
    return outer_box[0] <= cx <= outer_box[2] and outer_box[1] <= cy <= outer_box[3]


def _make_dynamic_square(bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
    """Создаёт динамический квадрат вокруг bbox"""
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    
    base_size = max(w, h)
    size = int(max(base_size * SQUARE_SCALE, SQUARE_MIN_SIZE))
    
    cx = x1 + w // 2
    cy = y1 + h // 2
    
    half = size // 2
    sx1 = max(0, cx - half)
    sy1 = max(0, cy - half)
    sx2 = min(frame_width - 1, cx + half)
    sy2 = min(frame_height - 1, cy + half)
    
    if sx2 <= sx1:
        sx2 = min(frame_width - 1, sx1 + 1)
    if sy2 <= sy1:
        sy2 = min(frame_height - 1, sy1 + 1)
    
    return sx1, sy1, sx2, sy2


def _estimate_cargo_bbox(truck_bbox: Tuple[int, int, int, int], class_id: int) -> Optional[Tuple[int, int, int, int]]:
    """
    Оценивает bbox кузова грузовика на основе bbox всего грузовика.
    Для самосвалов кузов - это задняя часть грузовика, выше кабины.
    """
    x1, y1, x2, y2 = truck_bbox
    w = x2 - x1
    h = y2 - y1
    
    # Для грузовиков (класс 7) кузов обычно в задней части, выше кабины
    if class_id == TRUCK_CLASS_ID:
        # Кузов занимает задние 60-70% длины грузовика (задняя часть)
        cargo_start_x = x1 + int(w * 0.30)  # Начинается с 30% от начала
        cargo_end_x = x2  # До конца грузовика
        
        # Кузов по высоте: начинается примерно с 30% от верха, занимает 35% высоты
        # Это верхняя часть задней части грузовика (выше кабины)
        cargo_start_y = y1 + int(h * 0.30)  # Начинается с 30% от верха
        cargo_height = int(h * 0.35)  # Занимает 35% высоты (уменьшено, чтобы не выходил слишком высоко)
        cargo_end_y = cargo_start_y + cargo_height
        
        # Ограничиваем границами исходного bbox
        cargo_start_x = max(x1, cargo_start_x)
        cargo_end_x = min(x2, cargo_end_x)
        cargo_start_y = max(y1, cargo_start_y)
        cargo_end_y = min(y2, cargo_end_y)
        
        # Проверяем, что кузов не пустой
        if cargo_end_x > cargo_start_x and cargo_end_y > cargo_start_y:
            return (cargo_start_x, cargo_start_y, cargo_end_x, cargo_end_y)
    
    # Для автобусов и машин кузов = весь bbox (или можно не выделять)
    return None


def _draw_overlay(frame: np.ndarray, state: LineState, selected_point: str, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x1_px, y1_px = int(state.x1 * w), int(state.y1 * h)
    x2_px, y2_px = int(state.x2 * w), int(state.y2 * h)

    out = frame.copy()
    
    # Детектируем объекты и рисуем боксы из модели
    model = _get_yolo_model()
    if model is not None:
        detections = _detect_for_preview(out, model)
        detections = _dedupe_by_iou(detections, PREVIEW_DEDUP_IOU)
        trucks = [d for d in detections if d.get("label") == "truck"]
        snows = [d for d in detections if d.get("label") == "snow"]

        # Снег валиден только если реально находится в truck:
        # 1) центр снегового бокса внутри truck
        # 2) значительная часть snow-бокса лежит в truck
        # 3) есть минимальное пересечение по IoU
        valid_snow = []
        for s in snows:
            sb = s.get("bbox")
            if not sb:
                continue
            if any(
                _iou_box(sb, tb) >= PREVIEW_OVERLAP_IOU
                and _intersection_ratio(sb, tb) >= PREVIEW_MIN_SNOW_IN_TRUCK
                and (
                    _center_inside(sb, tb)
                    or _intersection_ratio(sb, tb) >= 0.45
                )
                for t in trucks
                for tb in [t.get("bbox")]
                if tb
            ):
                valid_snow.append(s)

        truck_count = 0
        snow_count = 0
        for det in trucks:
            bbox = det.get('bbox')
            if bbox:
                truck_count += 1
                x1, y1, x2, y2 = bbox
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out, "truck", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for det in valid_snow:
            bbox = det.get("bbox")
            if bbox:
                snow_count += 1
                x1, y1, x2, y2 = bbox
                cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(out, "snow", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if trucks or snows:
            cv2.putText(
                out,
                f"YOLO: trucks={truck_count} snow_in_truck={snow_count}/{len(snows)}",
                (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 255, 180),
                2,
            )
    
    # Рисуем линию
    cv2.line(out, (x1_px, y1_px), (x2_px, y2_px), (0, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.circle(out, (x1_px, y1_px), 6, (0, 200, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(out, (x2_px, y2_px), 6, (0, 120, 255), 1, lineType=cv2.LINE_AA)

    cv2.putText(
        out,
        f"A=({state.x1:.3f},{state.y1:.3f}) B=({state.x2:.3f},{state.y2:.3f})",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        out,
        f"Selected: {selected_point} | FPS={fps:.1f}",
        (10, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return out


def _save_config(state: LineState, shape: Tuple[int, int], path: str = "line_config.json") -> None:
    h, w = shape
    payload = {
        "frame_size": {"width": w, "height": h},
        "line_normalized": {
            "start": {"x": state.x1, "y": state.y1},
            "end": {"x": state.x2, "y": state.y2},
        },
        "line_pixels": {
            "start": {"x": int(state.x1 * w), "y": int(state.y1 * h)},
            "end": {"x": int(state.x2 * w), "y": int(state.y2 * h)},
        },
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[LINE-PREVIEW] Saved to {path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    if not RTSP_URL:
        print("[LINE-PREVIEW] ERROR: SNOW_VIDEO_SOURCE_URL not set in app.env/system env")
        return 1

    # TCP-транспорт для RTSP уменьшает потери и ошибки декодера H.264 при плохой сети
    url = RTSP_URL
    if url.lower().startswith("rtsp://") and "rtsp_transport=" not in url:
        url = url + ("&" if "?" in url else "?") + "rtsp_transport=tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[LINE-PREVIEW] ERROR: cannot open RTSP: {RTSP_URL}")
        return 1
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # меньше буфер — меньше задержка и «залипание» на битых кадрах

    state = LineState(LINE_X1, LINE_Y1, LINE_X2, LINE_Y2)
    state.clamp()

    print("[LINE-PREVIEW] Controls: Q/Esc exit | Tab/C select point | WASD small | IJKL big | R reload env | P print | S save json")
    selected = "A"  # A or B

    fps = 0.0
    fc = 0
    last = time.time()
    last_frame_shape = (0, 0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            last_frame_shape = frame.shape[:2]
            fc += 1
            now = time.time()
            if now - last >= 1.0:
                fps = fc / (now - last)
                fc = 0
                last = now

            vis = _draw_overlay(frame, state, selected, fps)
            cv2.imshow("Line Preview (1 stream)", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (9, ord("c")):  # Tab or C
                selected = "B" if selected == "A" else "A"
            if key == ord("r"):
                _reload_env(state)
            if key == ord("p"):
                print(
                    f"[LINE-PREVIEW] A=({state.x1:.3f},{state.y1:.3f}) "
                    f"B=({state.x2:.3f},{state.y2:.3f})"
                )
            if key == ord("s") and last_frame_shape != (0, 0):
                _save_config(state, last_frame_shape)

            # movement helpers
            def move(dx: float, dy: float) -> None:
                if selected == "A":
                    state.x1 += dx
                    state.y1 += dy
                else:
                    state.x2 += dx
                    state.y2 += dy
                state.clamp()

            if key == ord("w"):
                move(0.0, -STEP_SMALL)
            if key == ord("s"):
                move(0.0, STEP_SMALL)
            if key == ord("a"):
                move(-STEP_SMALL, 0.0)
            if key == ord("d"):
                move(STEP_SMALL, 0.0)

            if key == ord("i"):
                move(0.0, -STEP_BIG)
            if key == ord("k"):
                move(0.0, STEP_BIG)
            if key == ord("j"):
                move(-STEP_BIG, 0.0)
            if key == ord("l"):
                move(STEP_BIG, 0.0)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[LINE-PREVIEW] Final values:")
        print(f"LINE_X1={state.x1:.3f}")
        print(f"LINE_Y1={state.y1:.3f}")
        print(f"LINE_X2={state.x2:.3f}")
        print(f"LINE_Y2={state.y2:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

