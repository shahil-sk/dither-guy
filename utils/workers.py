from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from PySide6.QtCore import Signal, QThread, QMutex

from .dither_kernels import apply_dither
from .constants import _VIDEO_WORKERS

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


class DitherWorker(QThread):
    finished = Signal(object)
    error    = Signal(str)

    def __init__(self, img, pixel_size, threshold, replace_color, method,
                 brightness, contrast, blur, sharpen,
                 glow_radius=0, glow_intensity=0, preview=False,
                 palette_name="B&W", custom_palette=None):
        super().__init__()
        self._img  = img; self._ps = pixel_size; self._t = threshold
        self._rc   = replace_color; self._m = method
        self._br   = brightness; self._co = contrast
        self._bl   = blur; self._sh = sharpen
        self._gr   = glow_radius; self._gi = glow_intensity
        self._prev = preview
        self._pal  = palette_name
        self._cpal = custom_palette
        self._stop = False; self._mutex = QMutex()
        self.setPriority(QThread.Priority.LowPriority)

    def run(self):
        try:
            t0     = time.perf_counter()
            result = apply_dither(
                self._img, self._ps, self._t, self._rc, self._m,
                self._br, self._co, self._bl, self._sh,
                self._gr, self._gi, self._prev,
                self._pal, self._cpal)
            elapsed = time.perf_counter() - t0
            self._mutex.lock()
            ok = not self._stop
            self._mutex.unlock()
            if ok:
                self.finished.emit((result, elapsed, self._prev))
        except MemoryError:
            self.error.emit("Out of memory — try smaller image or larger pixel size.")
        except Exception as exc:
            self.error.emit(f"Processing error: {exc}")

    def stop(self):
        self._mutex.lock(); self._stop = True; self._mutex.unlock()


def _process_frame_worker(args):
    """Top-level worker function for parallel frame dithering."""
    frame_bytes, mode, size, ps, t, rc, m, br, co, bl, sh, gr, gi, pal, cpal = args
    img = Image.frombytes(mode, size, frame_bytes)
    out = apply_dither(img, ps, t, rc, m, br, co, bl, sh, gr, gi,
                       palette_name=pal, custom_palette=cpal)
    return out.tobytes(), out.mode, out.size


class _VideoExportBase(QThread):
    """Shared plumbing for VideoExportWorker and GifExportWorker."""
    error = Signal(str)

    def __init__(self, video_path, save_path, pixel_size, threshold,
                 replace_color, method, brightness, contrast, blur, sharpen,
                 glow_radius=0, glow_intensity=0,
                 palette_name="B&W", custom_palette=None):
        super().__init__()
        self._vp   = video_path;  self._sp  = save_path
        self._ps   = pixel_size;  self._t   = threshold
        self._rc   = replace_color; self._m = method
        self._br   = brightness;  self._co  = contrast
        self._bl   = blur;        self._sh  = sharpen
        self._gr   = glow_radius; self._gi  = glow_intensity
        self._pal  = palette_name
        self._cpal = custom_palette
        self._stop = False; self._mutex = QMutex()

    def _make_args(self, frames: list[Image.Image]) -> list:
        ps, t, rc, m   = self._ps, self._t, self._rc, self._m
        br, co, bl, sh = self._br, self._co, self._bl, self._sh
        gr, gi         = self._gr, self._gi
        pal, cpal      = self._pal, self._cpal
        return [
            (f.tobytes(), f.mode, f.size, ps, t, rc, m, br, co, bl, sh, gr, gi, pal, cpal)
            for f in frames
        ]

    def _is_running(self) -> bool:
        self._mutex.lock()
        ok = not self._stop
        self._mutex.unlock()
        return ok

    def stop(self):
        self._mutex.lock(); self._stop = True; self._mutex.unlock()


class VideoExportWorker(_VideoExportBase):
    frame_ready = Signal(object)
    progress    = Signal(int, int)
    finished    = Signal()

    def run(self):
        if not _CV2:
            self.error.emit("opencv-python not installed."); return
        cap = out = None
        try:
            cap = cv2.VideoCapture(self._vp)
            if not cap.isOpened():
                self.error.emit("Failed to open video."); return

            fps   = cap.get(cv2.CAP_PROP_FPS) or 25.
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # ThreadPoolExecutor is safe inside a QThread.
            # ProcessPoolExecutor would re-spawn the GUI on Windows (issue #7).
            CHUNK = max(1, _VIDEO_WORKERS * 2)
            count = 0
            last_dithered = None
            frames_buf: list[Image.Image] = []

            with ThreadPoolExecutor(max_workers=_VIDEO_WORKERS) as executor:
                while self._is_running():
                    frames_buf.clear()
                    for _ in range(CHUNK):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames_buf.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    if not frames_buf:
                        break

                    dithered = [
                        Image.frombytes(mode, size, data)
                        for data, mode, size in executor.map(
                            _process_frame_worker, self._make_args(frames_buf))
                    ]

                    for dith in dithered:
                        if not self._is_running():
                            break
                        if out is None:
                            out = cv2.VideoWriter(
                                self._sp,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps,
                                (dith.width, dith.height))
                        out.write(cv2.cvtColor(np.array(dith), cv2.COLOR_RGB2BGR))
                        count += 1
                        self.progress.emit(count, total)
                        last_dithered = dith

                    if last_dithered is not None and count % max(1, CHUNK) == 0:
                        self.frame_ready.emit(last_dithered)

            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap: cap.release()
            if out: out.release()


class GifExportWorker(_VideoExportBase):
    """Export a dithered animated GIF from a video file."""
    frame_ready = Signal(object)
    progress    = Signal(int, int)
    finished    = Signal()

    def run(self):
        if not _CV2:
            self.error.emit("opencv-python not installed."); return
        cap = None
        try:
            cap = cv2.VideoCapture(self._vp)
            if not cap.isOpened():
                self.error.emit("Failed to open video."); return

            fps         = cap.get(cv2.CAP_PROP_FPS) or 25.
            total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_ms = max(20, int(1000 / fps))

            frames: list[Image.Image] = []
            count = 0

            while self._is_running():
                ret, frame = cap.read()
                if not ret:
                    break
                pil      = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                dithered = apply_dither(
                    pil, self._ps, self._t, self._rc, self._m,
                    self._br, self._co, self._bl, self._sh,
                    self._gr, self._gi,
                    palette_name=self._pal, custom_palette=self._cpal)
                frames.append(dithered.convert("P", palette=Image.ADAPTIVE, colors=256))
                count += 1
                self.progress.emit(count, total)
                if count % 10 == 0:
                    self.frame_ready.emit(dithered)

            if frames and self._is_running():
                frames[0].save(
                    self._sp,
                    save_all=True,
                    append_images=frames[1:],
                    loop=0,
                    duration=duration_ms,
                    optimize=False,
                )
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap: cap.release()
