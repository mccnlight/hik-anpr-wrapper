# modules/ocr.py
from __future__ import annotations

from typing import Tuple, Any, List

import cv2
import numpy as np
from paddleocr import PaddleOCR


class PlateOCR:
    """
    Обёртка над PaddleOCR, заточенная под распознавание госномера.
    Используем общий OCR-pipeline 3.x и сами вытаскиваем rec_texts / rec_scores.
    """

    def __init__(self) -> None:
        # Минимальные настройки:
        # - отключаем лишние модули (ориентация документа, выпрямление, textline-cls)
        # - оставляем стандартный детектор + рекогнайзер
        self.ocr = PaddleOCR(
            lang="en",                      # цифры + латиница — нам достаточно
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # чуть поднимаем лимит размера, чтобы маленькие номера скейлились
            text_det_limit_side_len=256,
            text_det_limit_type="max",
            # не режем по порогу, сами решим
            text_rec_score_thresh=None,
            device="cpu",                   # если будет GPU: "gpu:0"
        )

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            return img

        # 1) BGR -> RGB
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]

        # 2) деликатное масштабирование: хотим ширину ~220–260 px
        target_w = 240
        scale = target_w / max(w, 1)
        if scale > 1.0:  # только увеличиваем, не уменьшаем
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 3) лёгкий Gaussian blur + unsharp mask
        blur = cv2.GaussianBlur(img_rgb, (3, 3), 0)
        sharp = cv2.addWeighted(img_rgb, 1.5, blur, -0.5, 0)

        return sharp


    def recognize(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Принимает уже вырезанную область номера (crop из YOLO).
        Возвращает (распознанный_текст, уверенность).
        Если не удалось — ("", 0.0).
        """
        if img is None or img.size == 0:
            return "", 0.0

        img_rgb = self._prepare_image(img)

        # В НОВОЙ версии нужно вызывать predict(), без det/cls
        result_list = self.ocr.predict(img_rgb)

        # Для дебага: посмотрим, что вообще прилетает от PaddleOCR
        # print("RAW OCR RESULT OBJECTS:", result_list)

        if not result_list:
            return "", 0.0

        first = result_list[0]

        # 🔴 ВАЖНО: у тебя first — dict, а не объект с .res
        if isinstance(first, dict):
            res_dict: dict[str, Any] = first
        else:
            res_dict = getattr(first, "res", {}) or {}

        # В general OCR pipeline нас интересуют rec_texts и rec_scores
        texts: List[str] = list(res_dict.get("rec_texts") or [])
        scores = res_dict.get("rec_scores") or []

        print("REC_TEXTS:", texts)
        print("REC_SCORES:", scores)

        if not texts:
            return "", 0.0

        # Выбираем лучший по score
        if isinstance(scores, np.ndarray) and scores.size > 0:
            best_idx = int(scores.argmax())
            best_score = float(scores[best_idx])
        elif isinstance(scores, (list, tuple)) and len(scores) > 0:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_score = float(scores[best_idx])
        else:
            # если по какой-то причине score нет — возьмём первую строку
            best_idx, best_score = 0, 0.0

        raw_text = str(texts[best_idx])

        # Нормализация: убираем пробелы, приводим к верхнему регистру
        text = raw_text.replace(" ", "").upper()

        return text, best_score
