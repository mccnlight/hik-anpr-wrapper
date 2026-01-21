from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import httpx
import numpy as np
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class PlateRecognizerClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        # Очищаем токен от пробелов и не-ASCII символов
        self.token = str(token).strip()
        self.timeout_seconds = timeout_seconds

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
    )
    async def recognize(self, image_bgr: np.ndarray, regions: str) -> Dict[str, Any]:
        if image_bgr is None or image_bgr.size == 0:
            return {"results": []}

        ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return {"results": []}

        files = {"upload": ("plate.jpg", buf.tobytes(), "image/jpeg")}
        data = {"regions": regions}
        # Убеждаемся, что токен содержит только ASCII символы
        token_clean = self.token.encode("ascii", errors="ignore").decode("ascii")
        headers = {"Authorization": f"Token {token_clean}"}

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(self.base_url, data=data, files=files, headers=headers)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"retryable status={resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            resp.raise_for_status()
            return resp.json()


def parse_plr_response(payload: Dict[str, Any]) -> Tuple[Optional[str], float, list]:
    results = payload.get("results") or []
    if not results:
        return None, 0.0, []
    best = results[0]
    plate = best.get("plate")
    score = float(best.get("score") or 0.0)
    candidates = best.get("candidates") or []
    return plate, score, candidates
