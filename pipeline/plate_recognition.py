from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from clients.plate_recognizer import PlateRecognizerClient, parse_plr_response
from modules.blur import variance_of_laplacian
from modules.preprocess_ocr import preprocess_for_ocr


CropFn = Callable[[np.ndarray, Optional[Tuple[int, int, int, int]]], Optional[np.ndarray]]
UpscaleFn = Callable[[np.ndarray], np.ndarray]


def select_best_roi_from_buffer(
    items: List[Any],
    crop_fn: CropFn,
    upscale_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Выбирает лучший ROI из буфера по blur_score.
    Порядок обработки:
    1. Извлечение ROI (crop_fn)
    2. HAT upscale (если upscale_fn передана)
    3. Предобработка с легкой резкостью (если preprocess_fn передана)
    4. Blur detection (variance_of_laplacian)
    """
    best_score = -1.0
    best_roi = None
    best_frame = None
    best_ts = None
    scores = []

    for item in items:
        frame = item.frame
        bbox = item.bbox
        ts = item.timestamp
        roi = crop_fn(frame, bbox)
        if roi is None or roi.size == 0:
            continue
        
        # 1. Сначала HAT upscale (если функция передана)
        if upscale_fn is not None:
            roi = upscale_fn(roi)
        
        # 2. Предобработка БЕЗ резкости для blur detection
        # (резкость ухудшает blur_score, нужна только для OCR)
        if preprocess_fn is not None:
            # Передаем apply_sharpening=False для blur detection
            roi = preprocess_fn(roi, apply_sharpening=False)
        
        # 3. Проверяем blur_score на обработанном ROI (после upscale, БЕЗ резкости)
        score = variance_of_laplacian(roi)
        scores.append((score, ts))
        if score > best_score:
            best_score = score
            best_roi = roi
            best_frame = frame
            best_ts = ts

    scores.sort(reverse=True, key=lambda x: x[0])

    return {
        "best_roi": best_roi,
        "best_frame": best_frame,
        "best_ts": best_ts,
        "best_score": best_score if best_score >= 0 else 0.0,
        "scores": scores[:5],
        "count": len(items),
    }


async def recognize_with_fallback(
    plr_client: Optional[PlateRecognizerClient],
    plr_enabled: bool,
    plr_regions: str,
    plr_min_score: float,
    try_preprocessed: bool,
    best_roi: Optional[np.ndarray],
    fallback_coro: Callable[[], Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "plr_used": False,
        "plr_plate": None,
        "plr_score": 0.0,
        "plr_candidates": [],
        "plr_raw": None,
        "plr_latency_ms": None,
        "fallback_used": False,
        "fallback_result": None,
    }

    if not plr_enabled or plr_client is None or best_roi is None:
        result["fallback_used"] = True
        result["fallback_result"] = await fallback_coro()
        return result

    async def _try_plr(img: np.ndarray) -> Tuple[Optional[str], float, list, Dict[str, Any], float]:
        start = time.time()
        payload = await plr_client.recognize(img, plr_regions)
        latency_ms = (time.time() - start) * 1000.0
        plate, score, candidates = parse_plr_response(payload)
        return plate, score, candidates, payload, latency_ms

    plate, score, candidates, payload, latency_ms = await _try_plr(best_roi)
    result.update(
        {
            "plr_plate": plate,
            "plr_score": score,
            "plr_candidates": candidates,
            "plr_raw": payload,
            "plr_latency_ms": latency_ms,
        }
    )

    if score < plr_min_score and try_preprocessed:
        # Применяем предобработку С резкостью для OCR (best_roi уже был upscaled)
        pre = preprocess_for_ocr(best_roi, apply_sharpening=True)
        p_plate, p_score, p_candidates, p_payload, p_latency = await _try_plr(pre)
        if p_score > score:
            result.update(
                {
                    "plr_plate": p_plate,
                    "plr_score": p_score,
                    "plr_candidates": p_candidates,
                    "plr_raw": p_payload,
                    "plr_latency_ms": p_latency,
                }
            )

    if (result["plr_score"] or 0.0) >= plr_min_score:
        result["plr_used"] = True
        return result

    result["fallback_used"] = True
    result["fallback_result"] = await fallback_coro()
    return result
