from __future__ import annotations

import os
import threading
import urllib.parse
from typing import Optional, Tuple

import numpy as np
import importlib.util
import pathlib
import sys

import cv2
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv("app.env")

_HAT_IMPORT_ERROR = None
HAT = None  # type: ignore[assignment]

try:
    from basicsr.archs.hat_arch import HAT as _HAT

    HAT = _HAT
except Exception as exc:  # pragma: no cover - handled at runtime
    _HAT_IMPORT_ERROR = exc


def _load_hat_arch_direct() -> Optional[type]:
    for p in sys.path:
        candidate = pathlib.Path(p) / "hat" / "archs" / "hat_arch.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("hat_arch", candidate)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return getattr(mod, "HAT", None)
    return None


if HAT is None:
    HAT = _load_hat_arch_direct()
    if HAT is None:
        _HAT_IMPORT_ERROR = _HAT_IMPORT_ERROR or RuntimeError("hat_arch.py not found")


class HATUpscaler:
    _instance: Optional["HATUpscaler"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "HATUpscaler":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        if HAT is None:
            raise RuntimeError(
                "HAT model import failed. Install basicsr==1.3.4.9 and "
                "git+https://github.com/XPixelGroup/HAT.git. "
                f"Original error: {_HAT_IMPORT_ERROR}"
            )

        variant = os.getenv("HAT_VARIANT", "hat-s").lower().strip()
        self.window_size = 16
        self.upscale = 4

        if variant == "hat-s":
            self.compress_ratio = 24
            self.squeeze_factor = 24
            self.depths = [6, 6, 6, 6, 6, 6]
            self.embed_dim = 144
            self.num_heads = [6, 6, 6, 6, 6, 6]
            self.mlp_ratio = 2
        elif variant in ("hat", "hat-l"):
            self.compress_ratio = 3
            self.squeeze_factor = 30
            self.depths = [6] * 12
            self.embed_dim = 180
            self.num_heads = [6] * 12
            self.mlp_ratio = 2
        else:
            raise ValueError(f"Unknown HAT_VARIANT='{variant}'")

        self.device = torch.device("cpu")
        self.tile = int(os.getenv("HAT_TILE", "0"))
        self.tile_overlap = self.window_size
        self.postprocess = os.getenv("HAT_POSTPROCESS", "true").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.clahe_enable = os.getenv("HAT_CLAHE_ENABLE", "true").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.clahe_clip = float(os.getenv("HAT_CLAHE_CLIP", "2.0"))
        self.clahe_tile = int(os.getenv("HAT_CLAHE_TILE", "8"))
        self.sharpen_amount = float(os.getenv("HAT_SHARPEN_AMOUNT", "1.0"))
        self.sharpen_sigma = float(os.getenv("HAT_SHARPEN_SIGMA", "0.8"))

        weights_path = self._resolve_weights_path()
        self.model = HAT(
            in_chans=3,
            out_chans=3,
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            squeeze_factor=self.squeeze_factor,
            mlp_ratio=self.mlp_ratio,
            upsampler="pixelshuffle",
            upscale=self.upscale,
        )

        state_dict = self._load_state_dict(weights_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

    def _resolve_weights_path(self) -> str:
        weights_path = os.getenv("HAT_WEIGHTS_PATH", "").strip()
        weights_url = os.getenv("HAT_WEIGHTS_URL", "").strip()

        if weights_path and os.path.exists(weights_path):
            return weights_path

        if not weights_url:
            raise RuntimeError("HAT_WEIGHTS_PATH or HAT_WEIGHTS_URL must be set")

        parsed = urllib.parse.urlparse(weights_url)
        filename = os.path.basename(parsed.path) or "hat_weights.pth"
        if not weights_path:
            weights_path = os.path.join("models", filename)

        if not os.path.exists(weights_path):
            os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
            self._download_file(weights_url, weights_path)

        return weights_path

    @staticmethod
    def _download_file(url: str, dst_path: str) -> None:
        import requests

        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(dst_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    @staticmethod
    def _load_state_dict(weights_path: str) -> dict:
        ckpt = torch.load(weights_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "params_ema" in ckpt:
                return ckpt["params_ema"]
            if "params" in ckpt:
                return ckpt["params"]
            if "state_dict" in ckpt:
                return ckpt["state_dict"]
        return ckpt

    def _pad_to_window(self, img: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, h, w = img.shape
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h == 0 and pad_w == 0:
            return img, 0, 0
        padded = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
        return padded, pad_h, pad_w

    def _tile_inference(self, img: torch.Tensor) -> torch.Tensor:
        _, _, h, w = img.shape
        tile = self.tile
        overlap = min(self.tile_overlap, tile // 2) if tile > 0 else 0
        stride = tile - overlap
        if tile <= 0 or stride <= 0 or (h <= tile and w <= tile):
            return self.model(img)

        out_h = h * self.upscale
        out_w = w * self.upscale
        output = torch.zeros((1, 3, out_h, out_w), device=img.device)
        weight = torch.zeros_like(output)

        ys = list(range(0, h, stride))
        xs = list(range(0, w, stride))
        for y in ys:
            for x in xs:
                y0 = min(y, h - tile)
                x0 = min(x, w - tile)
                tile_img = img[:, :, y0 : y0 + tile, x0 : x0 + tile]
                out_tile = self.model(tile_img)

                oy0 = y0 * self.upscale
                ox0 = x0 * self.upscale
                oy1 = oy0 + out_tile.shape[2]
                ox1 = ox0 + out_tile.shape[3]
                output[:, :, oy0:oy1, ox0:ox1] += out_tile
                weight[:, :, oy0:oy1, ox0:ox1] += 1.0

        output = output / torch.clamp(weight, min=1.0)
        return output

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr

        is_gray = img_bgr.ndim == 2
        if is_gray:
            img = np.stack([img_bgr, img_bgr, img_bgr], axis=2)
        else:
            img = img_bgr

        img_rgb = img[:, :, ::-1].copy()
        tensor = torch.from_numpy(img_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        tensor, pad_h, pad_w = self._pad_to_window(tensor)

        with torch.inference_mode():
            if self.tile > 0:
                output = self._tile_inference(tensor)
            else:
                output = self.model(tensor)

        if pad_h or pad_w:
            out_h = output.shape[2] - pad_h * self.upscale
            out_w = output.shape[3] - pad_w * self.upscale
            output = output[:, :, :out_h, :out_w]

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        output_bgr = output[:, :, ::-1]

        if self.postprocess:
            output_bgr = self._postprocess(output_bgr)

        if is_gray:
            return output_bgr[:, :, 0]
        return output_bgr

    def _postprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        try:
            result = img_bgr
            if self.clahe_enable:
                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                tile = max(2, self.clahe_tile)
                clahe = cv2.createCLAHE(
                    clipLimit=max(0.1, self.clahe_clip),
                    tileGridSize=(tile, tile),
                )
                l2 = clahe.apply(l)
                lab2 = cv2.merge((l2, a, b))
                result = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

            amount = max(0.0, self.sharpen_amount)
            if amount > 0.0:
                sigma = max(0.1, self.sharpen_sigma)
                blurred = cv2.GaussianBlur(result, (0, 0), sigma)
                result = cv2.addWeighted(result, 1.0 + amount, blurred, -amount, 0)
            return result
        except Exception:
            return img_bgr
