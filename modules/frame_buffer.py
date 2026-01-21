from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BufferedFrame:
    timestamp: float
    frame: np.ndarray
    bbox: Optional[Tuple[int, int, int, int]]


class FrameBuffer:
    def __init__(
        self,
        seconds: float = 1.2,
        fps: int = 10,
        max_width: Optional[int] = 1280,
        max_height: Optional[int] = 720,
    ) -> None:
        self.seconds = max(0.1, float(seconds))
        self.fps = max(1, int(fps))
        self.max_width = max_width
        self.max_height = max_height
        self._lock = threading.Lock()
        self._frames: Deque[BufferedFrame] = deque(
            maxlen=int(self.seconds * self.fps) + 4
        )
        self._last_push_ts = 0.0

    def push(
        self,
        frame: np.ndarray,
        ts: Optional[float] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        if frame is None or frame.size == 0:
            return
        now_ts = ts if ts is not None else time.time()
        min_interval = 1.0 / float(self.fps)
        if (now_ts - self._last_push_ts) < min_interval:
            return

        resized = self._maybe_resize(frame)
        with self._lock:
            self._frames.append(
                BufferedFrame(timestamp=now_ts, frame=resized, bbox=bbox)
            )
        self._last_push_ts = now_ts

    def get_items(self) -> List[BufferedFrame]:
        with self._lock:
            return list(self._frames)

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if not self.max_width and not self.max_height:
            return frame.copy()

        scale = 1.0
        if self.max_width and w > self.max_width:
            scale = min(scale, self.max_width / float(w))
        if self.max_height and h > self.max_height:
            scale = min(scale, self.max_height / float(h))

        if scale >= 0.999:
            return frame.copy()

        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
