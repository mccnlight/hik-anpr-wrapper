"""
Чтение RTSP через subprocess ffmpeg (как в rtsp-yakor).
Используется для снеговой камеры: устойчивость к битым H.264, discardcorrupt, низкая задержка.
Подходит для Render.com (1 CPU, 2 GB): ограничение FPS и разрешения через env.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Optional, Tuple

import numpy as np


def _get_env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return int(float(v))
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


# Разрешение и FPS выхода ffmpeg (для экономии CPU/памяти на Render)
SNOW_FFMPEG_OUT_W = _get_env_int("SNOW_FFMPEG_OUT_W", 960)
SNOW_FFMPEG_OUT_H = _get_env_int("SNOW_FFMPEG_OUT_H", 540)
SNOW_FFMPEG_INPUT_FPS = _get_env_float("SNOW_FFMPEG_INPUT_FPS", 6.0)


def _resolve_ffmpeg_bin() -> Optional[str]:
    env_bin = os.getenv("FFMPEG_BIN", "").strip()
    if env_bin and os.path.exists(env_bin):
        return env_bin
    if env_bin:
        print(f"[FFMPEG] WARNING: FFMPEG_BIN set but not found: {env_bin}")
    p = shutil.which("ffmpeg")
    return p


class FFmpegRTSPReader:
    """
    Читает RTSP через ffmpeg в rawvideo bgr24.
    - rtsp_transport tcp, discardcorrupt, ignore_err — устойчивость к битому потоку.
    - Ограничение FPS и scale снижают нагрузку на CPU (Render 1 CPU).
    """

    def __init__(self, rtsp_url: str, name: str, width: int, height: int, fps: float):
        self.rtsp_url = rtsp_url
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = self.width * self.height * 3

        self._ffmpeg_bin = _resolve_ffmpeg_bin()
        self.process: Optional[subprocess.Popen] = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._lock = threading.Lock()
        self._has_frame = threading.Event()
        self._last_frame: Optional[np.ndarray] = None
        self._last_frame_ts: float = 0.0

    def start(self) -> bool:
        if not self._ffmpeg_bin:
            print("[FFMPEG] ERROR: ffmpeg not found. Set FFMPEG_BIN or ensure ffmpeg in PATH.")
            return False

        ffmpeg_loglevel = os.getenv("FFMPEG_LOGLEVEL", "fatal").strip() or "fatal"
        log_ffmpeg_stderr = os.getenv("LOG_FFMPEG_STDERR", "false").strip().lower() == "true"

        vf = f"fps={self.fps},scale={self.width}:{self.height}"
        cmd = [
            self._ffmpeg_bin,
            "-hide_banner",
            "-loglevel", ffmpeg_loglevel,
            "-nostats",
            "-rtsp_transport", "tcp",
            "-fflags", "+nobuffer+discardcorrupt",
            "-flags", "low_delay",
            "-err_detect", "ignore_err",
            "-analyzeduration", "0",
            "-probesize", "32",
            "-i", self.rtsp_url,
            "-an",
            "-vf", vf,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-",
        ]

        creationflags = 0
        if os.name == "nt":
            try:
                creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            except Exception:
                creationflags = 0

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE if log_ffmpeg_stderr else subprocess.DEVNULL,
                bufsize=self.frame_size * 4,
                creationflags=creationflags,
            )
            if self.process.stdout is None:
                self.release()
                return False

            if log_ffmpeg_stderr:
                def _read_stderr():
                    if self.process and self.process.stderr:
                        try:
                            for line in iter(self.process.stderr.readline, b""):
                                if line:
                                    msg = line.decode("utf-8", errors="ignore").strip()
                                    if msg:
                                        print(f"[FFMPEG:{self.name}] stderr: {msg}")
                        except Exception as e:
                            print(f"[FFMPEG:{self.name}] stderr reader error: {e}")

                t = threading.Thread(target=_read_stderr, daemon=True, name=f"ffmpeg-stderr-{self.name}")
                t.start()

            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name=f"ffmpeg-reader-{self.name}")
            self._thread.start()

            print(f"[FFMPEG:{self.name}] started output={self.width}x{self.height} fps={self.fps}")
            return True
        except Exception as e:
            print(f"[FFMPEG:{self.name}] start failed: {e}")
            self.release()
            return False

    def _loop(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        stdout = self.process.stdout
        need = self.frame_size

        while not self._stop.is_set():
            if self.process.poll() is not None:
                print(f"[FFMPEG:{self.name}] process exited with code {self.process.returncode}")
                break

            buf = bytearray(need)
            mv = memoryview(buf)
            got = 0

            try:
                while got < need and not self._stop.is_set():
                    chunk = stdout.read(need - got)
                    if not chunk:
                        break
                    mv[got : got + len(chunk)] = chunk
                    got += len(chunk)
            except Exception:
                break

            if got != need:
                time.sleep(0.02)
                continue

            try:
                frame = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
                with self._lock:
                    self._last_frame = frame
                    self._last_frame_ts = time.time()
                self._has_frame.set()
            except Exception:
                continue

        self._has_frame.set()

    def read(self, timeout_s: float = 1.0, stale_s: float = 3.0) -> Tuple[bool, Optional[np.ndarray]]:
        if self.process is None or self.process.poll() is not None:
            return False, None

        if not self._has_frame.wait(timeout=timeout_s):
            return False, None

        with self._lock:
            if self._last_frame is None:
                return False, None
            age = time.time() - float(self._last_frame_ts or 0.0)
            if age > stale_s:
                return False, None
            return True, self._last_frame.copy()

    def isOpened(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def release(self) -> None:
        self._stop.set()
        try:
            self._has_frame.set()
        except Exception:
            pass

        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._thread = None
