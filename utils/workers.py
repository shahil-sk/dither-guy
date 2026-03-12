# dither_guy/workers.py
# Qt background worker threads used to keep the UI responsive during
# computationally expensive operations.
#
# Classes
# ───────
#   DitherWorker        — runs apply_dither() in a background thread;
#                         emits finished(result, elapsed, is_preview) or error(msg)
#   VideoExportWorker   — reads a video file frame-by-frame, dithers each frame
#                         in a ThreadPoolExecutor, writes the output MP4, and
#                         emits frame_ready / progress / finished / error signals

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from PIL import Image

from PySide6.QtCore import QMutex, QThread, Signal

from utils.constants import _VIDEO_WORKERS
from utils.dither_kernels import apply_dither

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ---------------------------------------------------------------------------
# DitherWorker
# ---------------------------------------------------------------------------

class DitherWorker(QThread):
    """Run ``apply_dither()`` off the main thread.

    Signals
    ───────
    finished : emits ``(result_image, elapsed_seconds, is_preview)``
    error    : emits an error message string
    """

    finished = Signal(object)
    error    = Signal(str)

    def __init__(
        self,
        img,
        pixel_size: int,
        threshold: float,
        replace_color: tuple,
        method: str,
        brightness: float,
        contrast: float,
        blur: float,
        sharpen: float,
        glow_radius: float = 0,
        glow_intensity: float = 0,
        preview: bool = False,
        palette_name: str = "B&W",
        custom_palette: Optional[list] = None,
    ):
        super().__init__()
        self._img  = img;          self._ps  = pixel_size
        self._t    = threshold;    self._rc  = replace_color
        self._m    = method;       self._br  = brightness
        self._co   = contrast;     self._bl  = blur
        self._sh   = sharpen;      self._gr  = glow_radius
        self._gi   = glow_intensity
        self._prev = preview;      self._pal = palette_name
        self._cpal = custom_palette
        self._stop = False;        self._mutex = QMutex()

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            result = apply_dither(
                self._img, self._ps, self._t, self._rc, self._m,
                self._br, self._co, self._bl, self._sh,
                self._gr, self._gi, self._prev,
                self._pal, self._cpal,
            )
            elapsed = time.perf_counter() - t0
            self._mutex.lock(); ok = not self._stop; self._mutex.unlock()
            if ok:
                self.finished.emit((result, elapsed, self._prev))
        except MemoryError:
            self.error.emit("Out of memory — try a smaller image or larger pixel size.")
        except Exception as exc:
            self.error.emit(f"Processing error: {exc}")

    def stop(self) -> None:
        self._mutex.lock(); self._stop = True; self._mutex.unlock()


# ---------------------------------------------------------------------------
# VideoExportWorker
# ---------------------------------------------------------------------------

class VideoExportWorker(QThread):
    """Read a video, dither every frame in parallel, write an output MP4.

    Signals
    ───────
    frame_ready : emits the latest dithered PIL Image for live preview
    progress    : emits (current_frame, total_frames)
    finished    : emitted once all frames are written
    error       : emits an error message string
    """

    frame_ready = Signal(object)
    progress    = Signal(int, int)
    finished    = Signal()
    error       = Signal(str)

    def __init__(
        self,
        video_path: str,
        save_path: str,
        pixel_size: int,
        threshold: float,
        replace_color: tuple,
        method: str,
        brightness: float,
        contrast: float,
        blur: float,
        sharpen: float,
        glow_radius: float = 0,
        glow_intensity: float = 0,
    ):
        super().__init__()
        self._vp = video_path;   self._sp = save_path
        self._ps = pixel_size;   self._t  = threshold
        self._rc = replace_color;self._m  = method
        self._br = brightness;   self._co = contrast
        self._bl = blur;         self._sh = sharpen
        self._gr = glow_radius;  self._gi = glow_intensity
        self._stop  = False;     self._mutex = QMutex()

    def run(self) -> None:
        if not _CV2:
            self.error.emit("opencv-python not installed."); return

        cap = out = None
        try:
            cap = cv2.VideoCapture(self._vp)
            if not cap.isOpened():
                self.error.emit("Failed to open video."); return

            fps    = cap.get(cv2.CAP_PROP_FPS) or 25.
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out    = cv2.VideoWriter(
                self._sp,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height),
            )

            CHUNK = max(1, _VIDEO_WORKERS * 2)
            count = 0

            def _dither_frame(pil_frame: Image.Image) -> Image.Image:
                return apply_dither(
                    pil_frame, self._ps, self._t, self._rc, self._m,
                    self._br, self._co, self._bl, self._sh,
                    self._gr, self._gi,
                )

            frames_buf: list[Image.Image] = []
            while True:
                self._mutex.lock(); ok = not self._stop; self._mutex.unlock()
                if not ok: break

                frames_buf.clear()
                for _ in range(CHUNK):
                    ret, frame = cap.read()
                    if not ret: break
                    frames_buf.append(
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    )
                if not frames_buf: break

                with ThreadPoolExecutor(max_workers=_VIDEO_WORKERS) as ex:
                    dithered = list(ex.map(_dither_frame, frames_buf))

                for dith in dithered:
                    self._mutex.lock(); ok = not self._stop; self._mutex.unlock()
                    if not ok: break
                    out.write(cv2.cvtColor(np.array(dith), cv2.COLOR_RGB2BGR))
                    count += 1
                    self.progress.emit(count, total)

                if count % max(1, CHUNK) == 0:
                    self.frame_ready.emit(dithered[-1])

            self.finished.emit()

        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap: cap.release()
            if out: out.release()

    def stop(self) -> None:
        self._mutex.lock(); self._stop = True; self._mutex.unlock()
