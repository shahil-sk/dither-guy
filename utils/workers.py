from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from PySide6.QtCore import Signal, QThread, QMutex

from .dither_kernels import apply_dither
from .constants import _VIDEO_WORKERS
from .gpu_kernels import GPU_BACKEND, to_gpu, from_gpu, gpu_palette_batch
from .palettes import PALETTES

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

_GPU_EXPORT_BATCH = 8  # frames per GPU batch -- tune for VRAM vs latency
_PIPELINE_SENTINEL = object()  # poison pill to stop FramePipelineWorker


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

    def run(self):
        self.setPriority(QThread.Priority.NormalPriority)
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


class FramePipelineWorker(QThread):
    """
    Persistent dither thread for video playback.

    Replaces the old pattern of spawning a new DitherWorker QThread per frame.
    Stays alive for the lifetime of the VideoTab; caller pushes frames via
    push_frame() and receives results through the frame_ready signal.

    Queue depth = 1 (maxsize=1): if a new frame arrives before the current
    one is dithered, the pending frame is replaced so we always work on the
    freshest data and never accumulate backlog lag.

    Usage:
        worker = FramePipelineWorker(get_params)
        worker.frame_ready.connect(my_slot)
        worker.start()
        ...
        worker.push_frame(pil_img, frame_id)
        ...
        worker.stop(); worker.wait()
    """

    frame_ready = Signal(object, int)   # (PIL.Image, frame_id)
    error       = Signal(str)

    def __init__(self, get_params):
        super().__init__()
        self._get_params = get_params
        # maxsize=1: push_frame replaces stale pending frame automatically
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop = False

    # ── public API ────────────────────────────────────────────────────────────

    def push_frame(self, img: Image.Image, frame_id: int) -> None:
        """Drop-in replacement for _show_async. Non-blocking, always fresh."""
        # Drain stale item so queue never blocks the caller
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait((img, frame_id))
        except queue.Full:
            pass  # race: another push snuck in; fine, that one is fresher

    def stop(self) -> None:
        self._stop = True
        # Unblock the blocking get() in run()
        try:
            self._q.put_nowait(_PIPELINE_SENTINEL)
        except queue.Full:
            pass

    # ── thread body ───────────────────────────────────────────────────────────

    def run(self) -> None:
        self.setPriority(QThread.Priority.NormalPriority)
        while not self._stop:
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _PIPELINE_SENTINEL or self._stop:
                break
            img, frame_id = item
            try:
                p = self._get_params()
                result = apply_dither(
                    img,
                    p["pixel_size"], p["threshold"], p["color"], p["method"],
                    p["brightness"], p["contrast"], p["blur"], p["sharpen"],
                    p["glow_radius"], p["glow_intensity"],
                    preview=True,
                    palette_name=p.get("palette_name", "B&W"),
                    custom_palette=p.get("custom_palette"),
                )
                self.frame_ready.emit(result, frame_id)
            except Exception as exc:
                self.error.emit(f"frame pipeline error: {exc}")


def _process_frame_worker(args):
    """Top-level worker function for parallel frame dithering."""
    frame_bytes, mode, size, ps, t, rc, m, br, co, bl, sh, gr, gi, pal, cpal = args
    img = Image.frombytes(mode, size, frame_bytes)
    out = apply_dither(img, ps, t, rc, m, br, co, bl, sh, gr, gi,
                       palette_name=pal, custom_palette=cpal)
    return out.tobytes(), out.mode, out.size


def _resolve_palette_rgb(palette_name: str, custom_palette) -> np.ndarray | None:
    """Return (K, 3) uint8 RGB palette array, or None for B&W / non-palette methods."""
    if custom_palette is not None:
        arr = np.array(custom_palette, dtype=np.uint8)
        return arr if arr.ndim == 2 and arr.shape[1] == 3 else None
    pal = PALETTES.get(palette_name)
    if pal is None or palette_name == "B&W":
        return None
    try:
        arr = np.array(pal, dtype=np.uint8)
        return arr if arr.ndim == 2 and arr.shape[1] == 3 else None
    except Exception:
        return None


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
    export_done = Signal()

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

            use_gpu_batch = GPU_BACKEND == "cuda"
            pal_rgb = _resolve_palette_rgb(self._pal, self._cpal) if use_gpu_batch else None
            if pal_rgb is None:
                use_gpu_batch = False

            BATCH = _GPU_EXPORT_BATCH if use_gpu_batch else max(1, _VIDEO_WORKERS * 2)
            count = 0
            last_dithered = None
            frames_buf: list[Image.Image] = []

            with ThreadPoolExecutor(max_workers=_VIDEO_WORKERS) as executor:
                while self._is_running():
                    frames_buf.clear()
                    for _ in range(BATCH):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames_buf.append(
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    if not frames_buf:
                        break

                    if use_gpu_batch:
                        arr = np.stack([np.array(f) for f in frames_buf])
                        arr_gpu  = to_gpu(arr)
                        arr_out  = gpu_palette_batch(arr_gpu, pal_rgb)
                        dithered = [Image.fromarray(arr_out[i]) for i in range(len(arr_out))]
                    else:
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

                    if last_dithered is not None and count % max(1, BATCH) == 0:
                        self.frame_ready.emit(last_dithered)

            self.export_done.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap: cap.release()
            if out: out.release()


class GifExportWorker(_VideoExportBase):
    """Export a dithered animated GIF from a video file."""
    frame_ready = Signal(object)
    progress    = Signal(int, int)
    export_done = Signal()

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
            self.export_done.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap: cap.release()
