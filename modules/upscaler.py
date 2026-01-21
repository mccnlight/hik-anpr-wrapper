import io
import os
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


UPSCALE_ENABLED = os.getenv("UPSCALE_ENABLED", "true").lower() == "true"
UPSCALE_MODE = os.getenv("UPSCALE_MODE", "none").lower().strip()  # none|lanczos|onnx
UPSCALE_FACTOR = int(os.getenv("UPSCALE_FACTOR", "2"))
UPSCALE_MAX_SIDE = int(os.getenv("UPSCALE_MAX_SIDE", "2560"))
UPSCALE_ONNX_PATH = os.getenv("UPSCALE_ONNX_PATH", "models/realesrgan-x2plus.onnx")
UPSCALE_ONNX_SCALE = int(os.getenv("UPSCALE_ONNX_SCALE", "2"))
UPSCALE_ONNX_URL = os.getenv("UPSCALE_ONNX_URL", "").strip()
UPSCALE_DEBUG_LOG = os.getenv("UPSCALE_DEBUG_LOG", "false").lower() == "true"
UPSCALE_FORCE_LANCZOS = os.getenv("UPSCALE_FORCE_LANCZOS", "false").lower() == "true"
UPSCALE_POSTPROCESS = os.getenv("UPSCALE_POSTPROCESS", "true").lower() == "true"
UPSCALE_SHARPEN_AMOUNT = float(os.getenv("UPSCALE_SHARPEN_AMOUNT", "1.0"))
UPSCALE_SHARPEN_SIGMA = float(os.getenv("UPSCALE_SHARPEN_SIGMA", "0.8"))
UPSCALE_CLAHE_ENABLE = os.getenv("UPSCALE_CLAHE_ENABLE", "false").lower() == "true"
UPSCALE_CLAHE_CLIP = float(os.getenv("UPSCALE_CLAHE_CLIP", "2.0"))
UPSCALE_CLAHE_TILE = int(os.getenv("UPSCALE_CLAHE_TILE", "8"))


_onnx_session: Optional["ort.InferenceSession"] = None
_onnx_session_lock = None


def _ensure_onnx_model(path: str) -> bool:
    if os.path.exists(path):
        return True
    if not UPSCALE_ONNX_URL:
        return False
    try:
        import requests

        os.makedirs(os.path.dirname(path), exist_ok=True)
        resp = requests.get(UPSCALE_ONNX_URL, stream=True, timeout=30)
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def _get_onnx_session() -> Optional["ort.InferenceSession"]:
    global _onnx_session, _onnx_session_lock
    if ort is None:
        if UPSCALE_DEBUG_LOG:
            print("[UPSCALE] onnxruntime not available, fallback to Lanczos")
        return None
    if _onnx_session_lock is None:
        import threading

        _onnx_session_lock = threading.Lock()
    with _onnx_session_lock:
        if _onnx_session is not None:
            return _onnx_session
        if not _ensure_onnx_model(UPSCALE_ONNX_PATH):
            if UPSCALE_DEBUG_LOG:
                print(f"[UPSCALE] ONNX model not found: {UPSCALE_ONNX_PATH}")
            return None
        try:
            _onnx_session = ort.InferenceSession(
                UPSCALE_ONNX_PATH,
                providers=["CPUExecutionProvider"],
            )
            if UPSCALE_DEBUG_LOG:
                print(f"[UPSCALE] ONNX session ready: {UPSCALE_ONNX_PATH}")
            return _onnx_session
        except Exception:
            if UPSCALE_DEBUG_LOG:
                print("[UPSCALE] ONNX session init failed, fallback to Lanczos")
            return None


def _pad_to_multiple(img: np.ndarray, multiple: int) -> Tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return img, 0, 0
    padded = cv2.copyMakeBorder(
        img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
    )
    return padded, pad_h, pad_w


def _run_onnx_upscale(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    session = _get_onnx_session()
    if session is None:
        return None
    try:
        input_shape = session.get_inputs()[0].shape
        fixed_h = input_shape[2] if len(input_shape) >= 4 else None
        fixed_w = input_shape[3] if len(input_shape) >= 4 else None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_name = session.get_inputs()[0].name

        # Если модель принимает фиксированный размер (например 64x64) — используем тайлинг
        if isinstance(fixed_h, int) and isinstance(fixed_w, int):
            tile_h = fixed_h
            tile_w = fixed_w
            padded, pad_h, pad_w = _pad_to_multiple(img_rgb, tile_h)
            ph, pw = padded.shape[:2]
            out_h = ph * UPSCALE_ONNX_SCALE
            out_w = pw * UPSCALE_ONNX_SCALE
            output = np.zeros((out_h, out_w, 3), dtype=np.uint8)

            for y in range(0, ph, tile_h):
                for x in range(0, pw, tile_w):
                    tile = padded[y:y + tile_h, x:x + tile_w]
                    inp = tile.astype(np.float32) / 255.0
                    inp = np.transpose(inp, (2, 0, 1))[None, ...]
                    out = session.run(None, {input_name: inp})[0]
                    out = np.squeeze(out, 0)
                    out = np.transpose(out, (1, 2, 0))
                    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
                    oy = y * UPSCALE_ONNX_SCALE
                    ox = x * UPSCALE_ONNX_SCALE
                    output[oy:oy + out.shape[0], ox:ox + out.shape[1]] = out

            if pad_h or pad_w:
                out_h = output.shape[0] - pad_h * UPSCALE_ONNX_SCALE
                out_w = output.shape[1] - pad_w * UPSCALE_ONNX_SCALE
                output = output[:out_h, :out_w]

            out_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            if UPSCALE_DEBUG_LOG:
                print(f"[UPSCALE] ONNX tiled upscale: {img_bgr.shape[1]}x{img_bgr.shape[0]} -> {out_bgr.shape[1]}x{out_bgr.shape[0]}")
            return out_bgr

        # Динамический размер — прогоняем целиком
        padded, pad_h, pad_w = _pad_to_multiple(img_rgb, 4)
        inp = padded.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        out = session.run(None, {input_name: inp})[0]
        out = np.squeeze(out, 0)
        out = np.transpose(out, (1, 2, 0))
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        if pad_h or pad_w:
            out_h = out.shape[0] - pad_h * UPSCALE_ONNX_SCALE
            out_w = out.shape[1] - pad_w * UPSCALE_ONNX_SCALE
            out = out[:out_h, :out_w]
        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out_bgr
    except Exception:
        return None


def _resize_lanczos(img_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    return cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def _get_target_size(w: int, h: int, scale: int) -> Tuple[int, int, float]:
    target_w = int(w * scale)
    target_h = int(h * scale)
    max_side = max(target_w, target_h)
    if max_side > UPSCALE_MAX_SIDE:
        shrink = UPSCALE_MAX_SIDE / float(max_side)
        target_w = max(1, int(target_w * shrink))
        target_h = max(1, int(target_h * shrink))
        return target_w, target_h, shrink
    return target_w, target_h, 1.0


def _postprocess(img_bgr: np.ndarray) -> np.ndarray:
    if not UPSCALE_POSTPROCESS:
        return img_bgr
    try:
        if UPSCALE_CLAHE_ENABLE:
            # CLAHE на яркости для локального контраста
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            tile = max(2, UPSCALE_CLAHE_TILE)
            clahe = cv2.createCLAHE(clipLimit=max(0.1, UPSCALE_CLAHE_CLIP), tileGridSize=(tile, tile))
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            img_bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        # Unsharp mask для четкости символов (мягко)
        amount = max(0.0, UPSCALE_SHARPEN_AMOUNT)
        if amount > 0:
            sigma = max(0.1, UPSCALE_SHARPEN_SIGMA)
            blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
            img_bgr = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)
    except Exception:
        return img_bgr
    return img_bgr


def upscale_jpeg_bytes(photo_bytes: bytes, scale: int) -> bytes:
    if not photo_bytes or scale <= 1:
        return photo_bytes
    if not UPSCALE_ENABLED:
        if UPSCALE_DEBUG_LOG:
            print("[UPSCALE] disabled by UPSCALE_ENABLED=false")
        return photo_bytes
    if UPSCALE_MODE == "none":
        if UPSCALE_DEBUG_LOG:
            print("[UPSCALE] UPSCALE_MODE=none (pass-through)")
        return photo_bytes

    nparr = np.frombuffer(photo_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return photo_bytes

    h, w = img_bgr.shape[:2]
    target_w, target_h, _ = _get_target_size(w, h, scale)
    if target_w <= w and target_h <= h:
        if UPSCALE_DEBUG_LOG:
            print(f"[UPSCALE] target size <= original ({w}x{h}), skipping")
        return photo_bytes

    upscaled = None
    if UPSCALE_MODE == "onnx" and (not UPSCALE_FORCE_LANCZOS) and UPSCALE_ONNX_SCALE >= 2:
        upscaled = _run_onnx_upscale(img_bgr)
        if upscaled is not None:
            if UPSCALE_ONNX_SCALE != scale:
                upscaled = _resize_lanczos(upscaled, target_w, target_h)
            if UPSCALE_DEBUG_LOG:
                print(f"[UPSCALE] ONNX upscale ok: {w}x{h} -> {upscaled.shape[1]}x{upscaled.shape[0]}")
    if upscaled is None:
        upscaled = _resize_lanczos(img_bgr, target_w, target_h)
        if UPSCALE_DEBUG_LOG:
            print(f"[UPSCALE] Lanczos upscale: {w}x{h} -> {target_w}x{target_h}")

    upscaled = _postprocess(upscaled)

    ok, buf = cv2.imencode(".jpg", upscaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return photo_bytes
    return buf.tobytes()

