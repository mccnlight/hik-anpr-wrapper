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
from typing import Tuple

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


def _draw_overlay(frame: np.ndarray, state: LineState, selected_point: str, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x1_px, y1_px = int(state.x1 * w), int(state.y1 * h)
    x2_px, y2_px = int(state.x2 * w), int(state.y2 * h)

    out = frame.copy()
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

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"[LINE-PREVIEW] ERROR: cannot open RTSP: {RTSP_URL}")
        return 1

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

