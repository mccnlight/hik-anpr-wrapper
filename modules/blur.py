from __future__ import annotations

import cv2
import numpy as np


def variance_of_laplacian(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
