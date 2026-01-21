from __future__ import annotations

import cv2
import numpy as np


def preprocess_for_ocr(bgr_roi: np.ndarray, apply_sharpening: bool = True) -> np.ndarray:
    """
    Предобработка ROI номера для OCR:
    - Конвертация в grayscale
    - CLAHE (контраст)
    - Denoising
    - Очень слабое затемнение (уменьшение яркости для лучшей читаемости)
    - Опциональная минимальная резкость (только для OCR, не для blur detection)
    
    Args:
        bgr_roi: Входное изображение BGR
        apply_sharpening: Применять ли резкость (True для OCR, False для blur detection)
    """
    if bgr_roi is None or bgr_roi.size == 0:
        return bgr_roi

    if bgr_roi.ndim == 2:
        gray = bgr_roi.copy()
    else:
        gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)

    # CLAHE для выравнивания контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Denoising (аккуратно, чтобы не убить детали)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Очень слабое затемнение (уменьшение яркости на 5-8%)
    # Это помогает при слишком ярких фотографиях
    darken_factor = 0.92  # Уменьшаем яркость на 8% (очень слабо)
    gray = cv2.convertScaleAbs(gray, alpha=darken_factor, beta=0)
    
    # Резкость применяется ТОЛЬКО для OCR, НЕ для blur detection
    if apply_sharpening:
        # Минимальная резкость (ослаблена, чтобы не ухудшить читаемость)
        blur = cv2.GaussianBlur(gray, (0, 0), 0.8)
        # Очень слабое повышение резкости (было 1.2/-0.2, теперь 1.05/-0.05)
        gray = cv2.addWeighted(gray, 1.05, blur, -0.05, 0)
    
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
