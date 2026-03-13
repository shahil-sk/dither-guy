from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .constants import _MAX_PIXELS, _HISTORY_LIMIT, _DEBOUNCE_MS
from .dither_kernels import apply_dither
from .ui_widgets import ZoomableImageView, pil_to_toga_image, hsep, vsep

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ---------------------------------------------------------------------------
# Image tab
# ---------------------------------------------------------------------------

class ImageTab:
    def __init__(self, get_params, show_status):
        self.get_params   = get_params
        self.show_status  = show_status
        self.original_img: Optional[Image.Image] = None
        self.dithered_img: Optional[Image.Image] = None
        self.last_dir     = str(Path.home())
        self._history: list[Image.Image] = []
        self._canvas = ZoomableImageView("Drop an image here  or  Open")
        self._info_label: toga.Label | None = None
        self._undo_btn: toga.Button | None = None
        self._window: toga.Window | None = None

    def build(self) -> toga.Box:
        self._info_label = toga.Label(
            "Drop an image here  or  Open",
            style=Pack(padding=(4, 8), font_size=10, text_align="center"),
        )
        bar = toga.Box(style=Pack(direction=ROW, padding=4))
        for label, handler in [
            ("Invert", lambda w: self.invert()),
            ("Rot L",  lambda w: self.rotate_left()),
            ("Rot R",  lambda w: self.rotate_right()),
            ("Flip H", lambda w: self.flip_h()),
            ("Flip V", lambda w: self.flip_v()),
            ("Crop",   self._crop_handler),
        ]:
            bar.add(toga.Button(label, on_press=handler, style=Pack(padding_right=4)))
        bar.add(vsep())
        self._undo_btn = toga.Button(
            "Undo", on_press=lambda w: self.undo(),
            style=Pack(padding_left=4), enabled=False,
        )
        bar.add(self._undo_btn)
        return toga.Box(
            children=[self._info_label, self._canvas.container, bar],
            style=Pack(direction=COLUMN, flex=1),
        )

    # ------------------------------------------------------------------
    # File operations  (toga 0.5.x dialog API)
    # ------------------------------------------------------------------

    async def open_file(self, window: toga.Window):
        self._window = window
        try:
            path = await window.dialog(
                toga.OpenFileDialog(
                    "Open Image",
                    initial_directory=Path(self.last_dir),
                    file_types=["png", "jpg", "jpeg", "bmp", "gif", "tiff", "webp"],
                )
            )
            if path:
                self._load(str(path), window)
        except Exception:
            pass

    async def save_file(self, window: toga.Window):
        self._window = window
        if self.dithered_img is None:
            await window.dialog(toga.InfoDialog("Nothing to save", "Process an image first."))
            return
        try:
            path = await window.dialog(
                toga.SaveFileDialog(
                    "Save Image",
                    suggested_filename="dithered.png",
                    file_types=["png", "jpg"],
                )
            )
            if path:
                self.dithered_img.save(str(path))
                self.last_dir = str(Path(str(path)).parent)
                self.show_status(f"saved  {Path(str(path)).name}")
        except Exception as exc:
            await window.dialog(toga.ErrorDialog("Save Error", str(exc)))

    def _load(self, path: str, window: toga.Window = None):
        try:
            img = Image.open(path)
            self.original_img = img.convert("RGB")
            self._history.clear()
            if self._undo_btn:
                self._undo_btn.enabled = False
            self.last_dir = str(Path(path).parent)
            self._refresh_info()
            self.show_status(f"loaded  {Path(path).name}")
            self.process()
        except Exception as exc:
            if window:
                asyncio.ensure_future(
                    window.dialog(toga.ErrorDialog("Open Error", str(exc)))
                )

    # ------------------------------------------------------------------
    # Image operations
    # ------------------------------------------------------------------

    def _push_history(self):
        if self.original_img is None:
            return
        self._history.append(self.original_img.copy())
        if len(self._history) > _HISTORY_LIMIT:
            self._history.pop(0)
        if self._undo_btn:
            self._undo_btn.enabled = True

    def invert(self):
        if self.original_img is None: return
        self._push_history()
        self.original_img = Image.fromarray((255 - np.array(self.original_img)).astype(np.uint8))
        self.show_status("inverted"); self._refresh_info(); self.process()

    def rotate_left(self):
        if self.original_img is None: return
        self._push_history()
        self.original_img = self.original_img.rotate(90, expand=True)
        self.show_status("rotated 90 CCW"); self._refresh_info(); self.process()

    def rotate_right(self):
        if self.original_img is None: return
        self._push_history()
        self.original_img = self.original_img.rotate(-90, expand=True)
        self.show_status("rotated 90 CW"); self._refresh_info(); self.process()

    def flip_h(self):
        if self.original_img is None: return
        self._push_history()
        self.original_img = self.original_img.transpose(Image.FLIP_LEFT_RIGHT)
        self.show_status("flipped H"); self._refresh_info(); self.process()

    def flip_v(self):
        if self.original_img is None: return
        self._push_history()
        self.original_img = self.original_img.transpose(Image.FLIP_TOP_BOTTOM)
        self.show_status("flipped V"); self._refresh_info(); self.process()

    async def _crop_handler(self, widget=None):
        await self.crop()

    async def crop(self):
        if self.original_img is None or not self._window:
            return
        from .ui_dialogs import ask_crop_values
        vals = await ask_crop_values(self._window, self.original_img.width, self.original_img.height)
        if vals is None:
            return
        l, t, r, b = vals["left"], vals["top"], vals["right"], vals["bottom"]
        x2 = self.original_img.width - r
        y2 = self.original_img.height - b
        if l >= x2 or t >= y2:
            await self._window.dialog(toga.InfoDialog("Crop", "Nothing left after crop."))
            return
        self._push_history()
        self.original_img = self.original_img.crop((l, t, x2, y2))
        self.show_status(f"cropped to {self.original_img.width}x{self.original_img.height}")
        self._refresh_info(); self.process()

    def undo(self):
        if not self._history: return
        self.original_img = self._history.pop()
        if self._undo_btn: self._undo_btn.enabled = bool(self._history)
        self.show_status("undo"); self._refresh_info(); self.process()

    def schedule(self): self.process()

    def process(self):
        if self.original_img is None: return
        self.show_status("processing...")
        p = self.get_params()
        try:
            result = apply_dither(
                self.original_img,
                p["pixel_size"], p["threshold"], p["color"], p["method"],
                p["brightness"], p["contrast"], p["blur"], p["sharpen"],
                p["glow_radius"], p["glow_intensity"],
                palette_name=p.get("palette_name", "B&W"),
                custom_palette=p.get("custom_palette"),
            )
            self.dithered_img = result
            self._canvas.set_image(result)
            self.show_status(f"{p['method']}  {result.width}x{result.height}")
            if self.original_img and self._info_label:
                ow, oh = self.original_img.size
                self._info_label.text = f"{ow}x{oh}  ->  {result.width}x{result.height} px"
        except Exception as exc:
            self.show_status(f"error: {exc}")

    def _refresh_info(self):
        if self.original_img and self._info_label:
            w, h = self.original_img.size
            self._info_label.text = f"{w}x{h} px  history: {len(self._history)}"

    def zoom_in(self):  self._canvas.zoom_in()
    def zoom_out(self): self._canvas.zoom_out()
    def fit(self):      self._canvas.fit()
    def actual(self):   self._canvas.actual()


# ---------------------------------------------------------------------------
# Video tab
# ---------------------------------------------------------------------------

class VideoTab:
    def __init__(self, get_params, show_status):
        self.get_params  = get_params
        self.show_status = show_status
        self.video_cap   = None
        self.video_path: str | None = None
        self.last_dir    = str(Path.home())
        self.is_playing  = False
        self._info_label: toga.Label | None = None
        self._play_btn: toga.Button | None = None
        self._canvas = ZoomableImageView("Load a video  MP4 / AVI / MOV")
        self._window: toga.Window | None = None

    def build(self) -> toga.Box:
        self._info_label = toga.Label(
            "no video loaded",
            style=Pack(padding=(4, 8), font_size=10, text_align="center"),
        )
        self._play_btn = toga.Button(
            "Play", on_press=lambda w: self.toggle_play(),
            enabled=False, style=Pack(padding=4),
        )
        bar = toga.Box(children=[self._play_btn], style=Pack(direction=ROW, padding=4))
        if not _CV2:
            bar.add(toga.Label(
                "opencv-python not installed - video disabled",
                style=Pack(padding=(4, 8), font_size=10),
            ))
        return toga.Box(
            children=[self._info_label, self._canvas.container, bar],
            style=Pack(direction=COLUMN, flex=1),
        )

    async def open_file(self, window: toga.Window):
        self._window = window
        if not _CV2:
            await window.dialog(toga.InfoDialog("Missing", "Install opencv-python."))
            return
        try:
            path = await window.dialog(
                toga.OpenFileDialog(
                    "Open Video",
                    initial_directory=Path(self.last_dir),
                    file_types=["mp4", "avi", "mov"],
                )
            )
            if path:
                self._load_video(str(path))
        except Exception:
            pass

    def _load_video(self, path: str):
        if not _CV2: return
        if self.video_cap:
            self.video_cap.release(); self.video_cap = None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release(); return
        self.video_cap = cap
        self.video_path = path
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last_dir = str(Path(path).parent)
        if self._info_label:
            self._info_label.text = f"{Path(path).name}  {total} frames @ {fps:.1f} fps"
        if self._play_btn: self._play_btn.enabled = True
        self.show_status(f"loaded  {Path(path).name}")
        self._next_frame()

    def toggle_play(self):
        if not self.video_cap: return
        self.is_playing = not self.is_playing
        if self._play_btn: self._play_btn.text = "Pause" if self.is_playing else "Play"
        if self.is_playing:
            asyncio.ensure_future(self._play_loop())

    async def _play_loop(self):
        fps   = self.video_cap.get(cv2.CAP_PROP_FPS) if self.video_cap else 25.0
        delay = 1.0 / (fps or 25.0)
        while self.is_playing and self.video_cap:
            self._next_frame()
            await asyncio.sleep(delay)

    def _next_frame(self):
        if not self.video_cap or not self.video_cap.isOpened(): return
        ret, frame = self.video_cap.read()
        if not ret:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0); return
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        p = self.get_params()
        try:
            dith = apply_dither(
                img, p["pixel_size"], p["threshold"], p["color"], p["method"],
                p["brightness"], p["contrast"], p["blur"], p["sharpen"],
                p["glow_radius"], p["glow_intensity"],
            )
            self._canvas.set_image(dith)
        except Exception as exc:
            self.show_status(f"frame error: {exc}")

    async def save_file(self, window: toga.Window):
        self._window = window
        if not _CV2:
            await window.dialog(toga.InfoDialog("Missing", "Install opencv-python.")); return
        if not self.video_path:
            await window.dialog(toga.InfoDialog("No Video", "Load a video first.")); return
        try:
            path = await window.dialog(
                toga.SaveFileDialog(
                    "Export Video",
                    suggested_filename="dithered.mp4",
                    file_types=["mp4"],
                )
            )
        except Exception:
            return
        if not path: return
        if self.is_playing: self.toggle_play()
        if self.video_cap:  self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        p = self.get_params()
        from .workers import VideoExportWorker
        worker = VideoExportWorker(
            self.video_path, str(path),
            p["pixel_size"], p["threshold"], p["color"], p["method"],
            p["brightness"], p["contrast"], p["blur"], p["sharpen"],
            p["glow_radius"], p["glow_intensity"],
        )
        worker.start()
        self.show_status("exporting...")

    def zoom_in(self):  self._canvas.zoom_in()
    def zoom_out(self): self._canvas.zoom_out()
    def fit(self):      self._canvas.fit()
    def actual(self):   self._canvas.actual()
