import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QFileDialog, QColorDialog,
    QScrollArea, QGroupBox, QMessageBox, QSplitter, QToolBar, QCheckBox,
    QTabWidget, QProgressBar, QDialog, QSpinBox, QGridLayout,
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QMutex
from PySide6.QtGui import QPixmap, QImage, QAction, QIcon, QColor

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ---------------------------------------------------------------------------
# Pre-computed dither matrices
# ---------------------------------------------------------------------------

_BAYER_4x4 = np.array([
    [  0, 128,  32, 160],
    [192,  64, 224,  96],
    [ 48, 176,  16, 144],
    [240, 112, 208,  80],
], dtype=np.float32)

_CLUSTER_4x4 = np.array([
    [12,  5,  6, 13],
    [ 4,  0,  1,  7],
    [11,  3,  2,  8],
    [15, 10,  9, 14],
], dtype=np.float32) * (255.0 / 16.0)

_HALFTONE_4x4 = np.array([
    [ 7, 13, 11,  4],
    [12, 16, 14,  8],
    [10, 15,  6,  2],
    [ 5,  9,  3,  1],
], dtype=np.float32) * (255.0 / 17.0)

_PATTERN_8x8 = np.fromfunction(
    lambda i, j: ((i + j) % 8) * 32.0, (8, 8), dtype=np.float32)

_BLUE_NOISE_8x8 = np.array([
    [0.938, 0.250, 0.688, 0.063, 0.875, 0.188, 0.625, 0.125],
    [0.375, 0.813, 0.500, 0.938, 0.313, 0.750, 0.438, 0.563],
    [0.688, 0.125, 0.875, 0.250, 0.625, 0.063, 0.938, 0.188],
    [0.188, 0.625, 0.063, 0.750, 0.125, 0.500, 0.250, 0.813],
    [0.813, 0.438, 0.563, 0.375, 0.875, 0.313, 0.688, 0.063],
    [0.313, 0.938, 0.188, 0.813, 0.063, 0.938, 0.125, 0.563],
    [0.563, 0.063, 0.750, 0.438, 0.563, 0.375, 0.813, 0.313],
    [0.063, 0.500, 0.313, 0.875, 0.250, 0.688, 0.438, 0.750],
], dtype=np.float32) * 255.0

_DOT_CLASS = np.array([
    [0, 2, 4, 6],
    [1, 3, 5, 7],
    [0, 2, 4, 6],
    [1, 3, 5, 7],
], dtype=np.int32)

METHODS = [
    "Bayer", "Clustered-Dot", "Halftone", "Blue-Noise",
    "Pattern", "Crosshatch", "Riemersma", "Variable-Error",
    "Dot-Diffusion", "Floyd-Steinberg", "Atkinson",
    "Jarvis-Judice-Ninke", "Stucki", "Random",
]

_HISTORY_LIMIT = 20


# ---------------------------------------------------------------------------
# Dither engine
# ---------------------------------------------------------------------------

def _tile(matrix, h, w):
    mh, mw = matrix.shape
    return np.tile(matrix, ((h + mh - 1) // mh, (w + mw - 1) // mw))[:h, :w]


def _adjust(img, brightness, contrast, blur, sharpen):
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    for _ in range(int(sharpen)):
        img = img.filter(ImageFilter.SHARPEN)
    return img


def apply_glow(img, radius, intensity):
    """Screen-blend a blurred version of the lit pixels back onto the image."""
    if radius <= 0 or intensity <= 0:
        return img
    blurred  = img.filter(ImageFilter.GaussianBlur(radius=radius))
    base     = np.array(img,     dtype=np.float32)
    glow_lyr = np.array(blurred, dtype=np.float32) * (intensity / 100.0)
    out = 255.0 - (255.0 - base) * (255.0 - glow_lyr) / 255.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def apply_dither(img, pixel_size, threshold, replace_color, method,
                 brightness=1.0, contrast=1.0, blur=0, sharpen=0,
                 glow_radius=0, glow_intensity=0):
    img = _adjust(img, brightness, contrast, blur, sharpen)
    img = img.convert('L')
    sw = max(1, img.width  // pixel_size)
    sh = max(1, img.height // pixel_size)
    img = img.resize((sw, sh), Image.NEAREST)
    a = np.array(img, dtype=np.float32)
    h, w = a.shape
    t = float(threshold)

    if method == "Bayer":
        tiled = _tile(_BAYER_4x4, h, w)
        a = np.where(a + tiled / 255.0 * (255.0 - t) > t, 255, 0).astype(np.uint8)

    elif method == "Clustered-Dot":
        tiled = _tile(_CLUSTER_4x4, h, w)
        a = np.where(a + (tiled - 128.0) * (1.0 - t / 255.0) > t, 255, 0).astype(np.uint8)

    elif method == "Halftone":
        tiled = _tile(_HALFTONE_4x4, h, w)
        a = np.where(a + (tiled - 128.0) * (1.0 - t / 255.0) > t, 255, 0).astype(np.uint8)

    elif method == "Pattern":
        tiled = _tile(_PATTERN_8x8, h, w)
        a = np.where(a + (tiled - 128.0) * (1.0 - t / 255.0) > t, 255, 0).astype(np.uint8)

    elif method == "Blue-Noise":
        tiled = _tile(_BLUE_NOISE_8x8, h, w)
        a = np.where(a > t + (tiled - 128.0) * 0.5, 255, 0).astype(np.uint8)

    elif method == "Random":
        rng = np.random.default_rng()
        r = rng.integers(0, 256, (h, w), dtype=np.float32)
        a = np.where(a > t * 0.7 + r * 0.3, 255, 0).astype(np.uint8)

    elif method == "Crosshatch":
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        ch = (np.sin(xs[None, :] * 0.5) + np.sin(ys[:, None] * 0.5)) * 64.0 + 128.0
        a = np.where(a + (ch - 128.0) * (1.0 - t / 255.0) > t, 255, 0).astype(np.uint8)

    elif method == "Riemersma":
        buf = [0.0] * 16
        for y in range(h):
            for x in (range(w) if y % 2 == 0 else range(w - 1, -1, -1)):
                old = a[y, x] + buf[0]
                new = 255.0 if old > t else 0.0
                a[y, x] = new
                buf = buf[1:] + [(old - new) * 0.0625]
        a = np.clip(a, 0, 255).astype(np.uint8)

    elif method == "Variable-Error":
        for y in range(h - 1):
            for x in range(1, w - 1):
                old = a[y, x]
                new = 255.0 if old > t else 0.0
                a[y, x] = new
                e = old - new
                f = old / 255.0
                a[y,     x + 1] = np.clip(a[y,     x + 1] + e * 7.0 * f / 16.0,        0, 255)
                a[y + 1, x - 1] = np.clip(a[y + 1, x - 1] + e * 3.0 * (1 - f) / 16.0, 0, 255)
                a[y + 1, x    ] = np.clip(a[y + 1, x    ] + e * 5.0 / 16.0,             0, 255)
                a[y + 1, x + 1] = np.clip(a[y + 1, x + 1] + e * 1.0 / 16.0,            0, 255)
        a = np.clip(a, 0, 255).astype(np.uint8)

    elif method == "Dot-Diffusion":
        for y in range(h):
            for x in range(w):
                cm  = _DOT_CLASS[y % 4, x % 4]
                old = a[y, x]
                new = 255.0 if old > t else 0.0
                a[y, x] = new
                if x + 1 < w:
                    a[y, x + 1] = np.clip(a[y, x + 1] + (old - new) / (cm + 1), 0, 255)
        a = np.clip(a, 0, 255).astype(np.uint8)

    elif method == "Floyd-Steinberg":
        for y in range(h):
            for x in range(w):
                old = a[y, x]
                new = 255.0 if old > t else 0.0
                a[y, x] = new
                e = old - new
                if x + 1 < w:             a[y,     x + 1] += e * 0.4375
                if y + 1 < h:
                    if x > 0:             a[y + 1, x - 1] += e * 0.1875
                    a[y + 1, x]           += e * 0.3125
                    if x + 1 < w:         a[y + 1, x + 1] += e * 0.0625
        a = np.clip(a, 0, 255).astype(np.uint8)

    elif method == "Atkinson":
        for y in range(h):
            for x in range(w):
                old = a[y, x]
                new = 255.0 if old > t else 0.0
                a[y, x] = new
                e = (old - new) * 0.125
                if x + 1 < w:             a[y,     x + 1] += e
                if x + 2 < w:             a[y,     x + 2] += e
                if y + 1 < h:
                    if x > 0:             a[y + 1, x - 1] += e
                    a[y + 1, x]           += e
                    if x + 1 < w:         a[y + 1, x + 1] += e
                if y + 2 < h:             a[y + 2, x]     += e
        a = np.clip(a, 0, 255).astype(np.uint8)

    elif method == "Jarvis-Judice-Ninke":
        d = 48.0
        for y in range(h):
            for x in range(w):
                old = a[y, x]
                new = 255.0 if old > t else 0.0
                a[y, x] = new
                e = old - new
                if x+1<w: a[y,x+1]       += e*7/d
                if x+2<w: a[y,x+2]       += e*5/d
                if y+1<h:
                    if x>1: a[y+1,x-2]   += e*3/d
                    if x>0: a[y+1,x-1]   += e*5/d
                    a[y+1,x]              += e*7/d
                    if x+1<w: a[y+1,x+1] += e*5/d
                    if x+2<w: a[y+1,x+2] += e*3/d
                if y+2<h:
                    if x>1: a[y+2,x-2]   += e*1/d
                    if x>0: a[y+2,x-1]   += e*3/d
                    a[y+2,x]              += e*5/d
                    if x+1<w: a[y+2,x+1] += e*3/d
                    if x+2<w: a[y+2,x+2] += e*1/d
        a = np.clip(a, 0, 255).astype(np.uint8)

    elif method == "Stucki":
        d = 42.0
        for y in range(h):
            for x in range(w):
                old = a[y, x]
                new = 255.0 if old > t else 0.0
                a[y, x] = new
                e = old - new
                if x+1<w: a[y,x+1]       += e*8/d
                if x+2<w: a[y,x+2]       += e*4/d
                if y+1<h:
                    if x>1: a[y+1,x-2]   += e*2/d
                    if x>0: a[y+1,x-1]   += e*4/d
                    a[y+1,x]              += e*8/d
                    if x+1<w: a[y+1,x+1] += e*4/d
                    if x+2<w: a[y+1,x+2] += e*2/d
                if y+2<h:
                    if x>1: a[y+2,x-2]   += e*1/d
                    if x>0: a[y+2,x-1]   += e*2/d
                    a[y+2,x]              += e*4/d
                    if x+1<w: a[y+2,x+1] += e*2/d
                    if x+2<w: a[y+2,x+2] += e*1/d
        a = np.clip(a, 0, 255).astype(np.uint8)

    img = Image.fromarray(a, mode='L')
    img = img.resize((sw * pixel_size, sh * pixel_size), Image.NEAREST)
    img = img.convert("RGB")
    data = np.array(img)
    mask = (data[:, :, 0] == 255) & (data[:, :, 1] == 255) & (data[:, :, 2] == 255)
    data[mask] = replace_color
    img = Image.fromarray(data)

    if glow_radius > 0 and glow_intensity > 0:
        img = apply_glow(img, glow_radius, glow_intensity)

    return img


def _pil_to_pixmap(img):
    raw  = img.tobytes("raw", "RGB")
    qimg = QImage(raw, img.width, img.height, img.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Crop dialog
# ---------------------------------------------------------------------------

class CropDialog(QDialog):
    """Enter pixel amounts to remove from each edge."""

    def __init__(self, img_w, img_h, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Image")
        self.setMinimumWidth(340)
        self.img_w = img_w
        self.img_h = img_h
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._info = QLabel()
        self._info.setAlignment(Qt.AlignCenter)
        self._info.setStyleSheet("font-size:11px; color:#aaa; padding:4px;")
        layout.addWidget(self._info)

        form = QWidget()
        fl   = QGridLayout(form)
        fl.setSpacing(8)

        self._spins = {}
        for row, (name, label, max_val) in enumerate([
            ("left",   "Remove from left",   self.img_w - 1),
            ("top",    "Remove from top",    self.img_h - 1),
            ("right",  "Remove from right",  self.img_w - 1),
            ("bottom", "Remove from bottom", self.img_h - 1),
        ]):
            lbl = QLabel(label)
            sp  = QSpinBox()
            sp.setMinimum(0)
            sp.setMaximum(max_val)
            sp.setValue(0)
            sp.setMinimumHeight(28)
            sp.valueChanged.connect(self._update_info)
            fl.addWidget(lbl, row, 0)
            fl.addWidget(sp,  row, 1)
            self._spins[name] = sp

        layout.addWidget(form)

        btns = QHBoxLayout()
        ok  = QPushButton("Crop")
        can = QPushButton("Cancel")
        ok.setMinimumHeight(30)
        can.setMinimumHeight(30)
        ok.clicked.connect(self.accept)
        can.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(ok)
        btns.addWidget(can)
        layout.addLayout(btns)

        self._update_info()

    def _update_info(self):
        l = self._spins["left"].value()
        t = self._spins["top"].value()
        r = self._spins["right"].value()
        b = self._spins["bottom"].value()
        nw = max(1, self.img_w - l - r)
        nh = max(1, self.img_h - t - b)
        self._info.setText(
            f"Original: {self.img_w} x {self.img_h}   →   Result: {nw} x {nh}")

    def values(self):
        return {k: sp.value() for k, sp in self._spins.items()}


# ---------------------------------------------------------------------------
# Worker: single image / frame
# ---------------------------------------------------------------------------

class DitherWorker(QThread):
    finished = Signal(object)   # (PIL.Image, float elapsed)
    progress = Signal(str)

    def __init__(self, img, pixel_size, threshold, replace_color, method,
                 brightness, contrast, blur, sharpen,
                 glow_radius=0, glow_intensity=0):
        super().__init__()
        self.img            = img
        self.pixel_size     = pixel_size
        self.threshold      = threshold
        self.replace_color  = replace_color
        self.method         = method
        self.brightness     = brightness
        self.contrast       = contrast
        self.blur           = blur
        self.sharpen        = sharpen
        self.glow_radius    = glow_radius
        self.glow_intensity = glow_intensity
        self._running       = True
        self._mutex         = QMutex()

    def run(self):
        try:
            self.progress.emit("Processing...")
            t0 = time.perf_counter()
            result = apply_dither(
                self.img, self.pixel_size, self.threshold,
                self.replace_color, self.method,
                self.brightness, self.contrast, self.blur, self.sharpen,
                self.glow_radius, self.glow_intensity,
            )
            elapsed = time.perf_counter() - t0
            self._mutex.lock()
            if self._running:
                self.finished.emit((result, elapsed))
            self._mutex.unlock()
        except Exception as exc:
            self.progress.emit(f"Error: {exc}")

    def stop(self):
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()


# ---------------------------------------------------------------------------
# Worker: full video export
# ---------------------------------------------------------------------------

class VideoExportWorker(QThread):
    frame_ready = Signal(object)
    progress    = Signal(int, int)
    finished    = Signal()
    error       = Signal(str)

    def __init__(self, video_path, save_path, pixel_size, threshold,
                 replace_color, method, brightness, contrast, blur, sharpen,
                 glow_radius=0, glow_intensity=0):
        super().__init__()
        self.video_path     = video_path
        self.save_path      = save_path
        self.pixel_size     = pixel_size
        self.threshold      = threshold
        self.replace_color  = replace_color
        self.method         = method
        self.brightness     = brightness
        self.contrast       = contrast
        self.blur           = blur
        self.sharpen        = sharpen
        self.glow_radius    = glow_radius
        self.glow_intensity = glow_intensity
        self._running       = True
        self._mutex         = QMutex()

    def run(self):
        if not _CV2:
            self.error.emit("opencv-python is not installed.")
            return
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit("Failed to open video file.")
                return
            fps    = cap.get(cv2.CAP_PROP_FPS)
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out    = cv2.VideoWriter(
                self.save_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps, (width, height))
            count = 0
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break
                pil  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                dith = apply_dither(
                    pil, self.pixel_size, self.threshold,
                    self.replace_color, self.method,
                    self.brightness, self.contrast, self.blur, self.sharpen,
                    self.glow_radius, self.glow_intensity)
                out.write(cv2.cvtColor(np.array(dith), cv2.COLOR_RGB2BGR))
                count += 1
                self.progress.emit(count, total)
                if count % 5 == 0:
                    self.frame_ready.emit(dith)
            cap.release()
            out.release()
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))

    def stop(self):
        self._mutex.lock()
        self._running = False
        self._mutex.unlock()


# ---------------------------------------------------------------------------
# Shared zoomable canvas
# ---------------------------------------------------------------------------

class ZoomableLabel(QLabel):
    def __init__(self, placeholder=""):
        super().__init__(placeholder)
        self.zoom_level      = 0
        self.original_pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setScaledContents(False)

    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.zoom_level = 0
        self._redraw()

    def _redraw(self):
        if not self.original_pixmap:
            return
        if self.zoom_level == 0:
            scaled = self.original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        else:
            w = int(self.original_pixmap.width()  * self.zoom_level)
            h = int(self.original_pixmap.height() * self.zoom_level)
            scaled = self.original_pixmap.scaled(
                w, h, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.setPixmap(scaled)

    def zoom_in(self):
        if self.zoom_level == 0: self.zoom_level = 1.0
        self.zoom_level = min(self.zoom_level * 1.25, 5.0)
        self._redraw()

    def zoom_out(self):
        if self.zoom_level == 0: self.zoom_level = 1.0
        self.zoom_level = max(self.zoom_level * 0.8, 0.1)
        self._redraw()

    def fit(self):
        self.zoom_level = 0
        self._redraw()

    def actual(self):
        self.zoom_level = 1.0
        self._redraw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.zoom_level == 0 and self.original_pixmap:
            self._redraw()


# ---------------------------------------------------------------------------
# Image tab
# ---------------------------------------------------------------------------

class ImageTab(QWidget):
    status_message = Signal(str)

    def __init__(self, get_params):
        super().__init__()
        self.get_params   = get_params
        self.original_img = None
        self.dithered_img = None
        self.last_dir     = str(Path.home())
        self.worker       = None
        self._pending     = False       # re-render requested while worker runs
        self.auto_update  = True
        self._history     = []          # PIL.Image snapshots for undo
        self._timer       = QTimer()
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.process)
        self._build()
        self.setAcceptDrops(True)

    # -- layout ----------------------------------------------------------------

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.info_lbl = QLabel("No image loaded")
        self.info_lbl.setAlignment(Qt.AlignCenter)
        self.info_lbl.setStyleSheet(
            "font-size:11px; padding:6px; background:#252525;")
        layout.addWidget(self.info_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.canvas = ZoomableLabel(
            "Load an image to start dithering\n\nDrag & drop or Ctrl+O")
        self.canvas.setStyleSheet("font-size:14px; color:#666;")
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll)

        # -- primary action bar --
        bar1 = QWidget()
        bar1.setStyleSheet("background:#252525; border-top:1px solid #3a3a3a;")
        bl1 = QHBoxLayout(bar1)
        bl1.setContentsMargins(8, 4, 8, 4)
        bl1.setSpacing(8)
        for lbl, slot in [
            ("Open (Ctrl+O)", self.open_file),
            ("Save (Ctrl+S)", self.save_file),
            ("Invert",        self.invert),
        ]:
            b = QPushButton(lbl)
            b.clicked.connect(slot)
            b.setMinimumHeight(28)
            bl1.addWidget(b)

        self.auto_cb = QCheckBox("Auto")
        self.auto_cb.setChecked(True)
        self.auto_cb.stateChanged.connect(self._toggle_auto)
        bl1.addWidget(self.auto_cb)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.process)
        self.apply_btn.setVisible(False)
        self.apply_btn.setMinimumHeight(28)
        bl1.addWidget(self.apply_btn)
        bl1.addStretch()
        layout.addWidget(bar1)

        # -- transform bar --
        bar2 = QWidget()
        bar2.setStyleSheet("background:#222; border-top:1px solid #333;")
        bl2 = QHBoxLayout(bar2)
        bl2.setContentsMargins(8, 3, 8, 3)
        bl2.setSpacing(6)

        for lbl, slot in [
            ("Rot L",   self.rotate_left),
            ("Rot R",   self.rotate_right),
            ("Flip H",  self.flip_h),
            ("Flip V",  self.flip_v),
            ("Crop...", self.crop),
        ]:
            b = QPushButton(lbl)
            b.clicked.connect(slot)
            b.setMinimumHeight(24)
            b.setStyleSheet("font-size:11px; padding:3px 8px;")
            bl2.addWidget(b)

        bl2.addStretch()

        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo)
        self.undo_btn.setEnabled(False)
        self.undo_btn.setMinimumHeight(24)
        self.undo_btn.setStyleSheet("font-size:11px; padding:3px 8px;")
        bl2.addWidget(self.undo_btn)
        layout.addWidget(bar2)

    # -- auto/manual toggle ---------------------------------------------------

    def _toggle_auto(self, state):
        self.auto_update = (state == Qt.Checked)
        self.apply_btn.setVisible(not self.auto_update)

    def schedule(self):
        if self.auto_update:
            self._timer.stop()
            self._timer.start(400)

    # -- drag & drop ----------------------------------------------------------

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).suffix.lower() in {
                    '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}:
                self._load(p)
                break

    # -- file operations ------------------------------------------------------

    def _load(self, path):
        try:
            self.original_img = Image.open(path).convert("RGB")
            self._history.clear()
            self.undo_btn.setEnabled(False)
            p = Path(path)
            self.last_dir = str(p.parent)
            self._refresh_info()
            self.status_message.emit(f"Loaded: {p.name}")
            self.process()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to open:\n{exc}")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self.last_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
            ";;All Files (*.*)")
        if path:
            self._load(path)

    def save_file(self):
        if self.dithered_img is None:
            QMessageBox.warning(self, "Warning", "No image to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image",
            str(Path(self.last_dir) / "dithered.png"),
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*.*)")
        if path:
            try:
                self.dithered_img.save(path)
                self.last_dir = str(Path(path).parent)
                self.status_message.emit(f"Saved: {Path(path).name}")
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{exc}")

    # -- helpers --------------------------------------------------------------

    def _refresh_info(self):
        if self.original_img:
            self.info_lbl.setText(
                f"{self.original_img.width} x {self.original_img.height} px"
                f"   (history: {len(self._history)})")

    def _push_history(self):
        if self.original_img is None:
            return
        self._history.append(self.original_img.copy())
        if len(self._history) > _HISTORY_LIMIT:
            self._history.pop(0)
        self.undo_btn.setEnabled(True)

    def _require_image(self, op="perform this action"):
        if self.original_img is None:
            QMessageBox.warning(self, "Warning", f"Load an image first to {op}.")
            return False
        return True

    # -- invert ---------------------------------------------------------------

    def invert(self):
        if not self._require_image("invert"): return
        self._push_history()
        data = 255 - np.array(self.original_img)
        self.original_img = Image.fromarray(data.astype(np.uint8))
        self.status_message.emit("Inverted")
        self._refresh_info()
        self.process()

    # -- transforms -----------------------------------------------------------

    def rotate_left(self):
        if not self._require_image("rotate"): return
        self._push_history()
        self.original_img = self.original_img.rotate(90, expand=True)
        self.status_message.emit("Rotated 90 CCW")
        self._refresh_info()
        self.process()

    def rotate_right(self):
        if not self._require_image("rotate"): return
        self._push_history()
        self.original_img = self.original_img.rotate(-90, expand=True)
        self.status_message.emit("Rotated 90 CW")
        self._refresh_info()
        self.process()

    def flip_h(self):
        if not self._require_image("flip"): return
        self._push_history()
        self.original_img = self.original_img.transpose(Image.FLIP_LEFT_RIGHT)
        self.status_message.emit("Flipped horizontal")
        self._refresh_info()
        self.process()

    def flip_v(self):
        if not self._require_image("flip"): return
        self._push_history()
        self.original_img = self.original_img.transpose(Image.FLIP_TOP_BOTTOM)
        self.status_message.emit("Flipped vertical")
        self._refresh_info()
        self.process()

    def crop(self):
        if not self._require_image("crop"): return
        dlg = CropDialog(self.original_img.width, self.original_img.height, self)
        if dlg.exec() != QDialog.Accepted:
            return
        v  = dlg.values()
        l, t, r, b = v["left"], v["top"], v["right"], v["bottom"]
        x2 = self.original_img.width  - r
        y2 = self.original_img.height - b
        if l >= x2 or t >= y2:
            QMessageBox.warning(self, "Crop", "Crop margins leave no image area.")
            return
        self._push_history()
        self.original_img = self.original_img.crop((l, t, x2, y2))
        self.status_message.emit(
            f"Cropped to {self.original_img.width} x {self.original_img.height}")
        self._refresh_info()
        self.process()

    # -- undo -----------------------------------------------------------------

    def undo(self):
        if not self._history:
            return
        self.original_img = self._history.pop()
        self.undo_btn.setEnabled(bool(self._history))
        self.status_message.emit("Undo")
        self._refresh_info()
        self.process()

    # -- dither processing ----------------------------------------------------

    def process(self):
        """Start a new render, or mark pending if one is already running.

        Never overwrites self.worker while it is still running — that is
        what caused the QThread abort.  Instead we set _pending=True and
        let _on_done() restart the render once the current one finishes.
        """
        if self.original_img is None:
            return

        if self.worker and self.worker.isRunning():
            # Signal the running worker to abort early, mark a re-run.
            self._pending = True
            self.worker.stop()
            return

        self._pending = False
        self.status_message.emit("Processing...")
        p = self.get_params()

        self.worker = DitherWorker(
            self.original_img,
            p['pixel_size'], p['threshold'], p['color'], p['method'],
            p['brightness'], p['contrast'], p['blur'], p['sharpen'],
            p['glow_radius'], p['glow_intensity'])
        self.worker.finished.connect(self._on_done)
        self.worker.progress.connect(self.status_message)
        # deleteLater keeps Qt from holding a destroyed-thread reference.
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def _on_done(self, payload):
        img, elapsed = payload
        self.dithered_img = img
        self.canvas.set_image(_pil_to_pixmap(img))
        self.canvas.setStyleSheet("")
        self.status_message.emit(f"Done  ({elapsed * 1000:.0f} ms)")

        # If params changed while we were rendering, kick off one more pass.
        if self._pending:
            QTimer.singleShot(0, self.process)

    def cleanup(self):
        """Stop any running worker and wait for it to finish."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(10000)

    def zoom_in(self):    self.canvas.zoom_in()
    def zoom_out(self):   self.canvas.zoom_out()
    def fit(self):        self.canvas.fit()
    def actual(self):     self.canvas.actual()
    def zoom_level(self): return self.canvas.zoom_level


# ---------------------------------------------------------------------------
# Video tab
# ---------------------------------------------------------------------------

class VideoTab(QWidget):
    status_message = Signal(str)

    def __init__(self, get_params):
        super().__init__()
        self.get_params    = get_params
        self.video_cap     = None
        self.video_path    = None
        self.current_frame = None
        self.is_playing    = False
        self.export_worker = None
        self.last_dir      = str(Path.home())
        self._play_timer   = QTimer()
        self._play_timer.timeout.connect(self._next_frame)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.info_lbl = QLabel("No video loaded")
        self.info_lbl.setAlignment(Qt.AlignCenter)
        self.info_lbl.setStyleSheet(
            "font-size:11px; padding:6px; background:#252525;")
        layout.addWidget(self.info_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.canvas = ZoomableLabel(
            "Load a video to start\n\nSupported: MP4, AVI, MOV")
        self.canvas.setStyleSheet("font-size:14px; color:#666;")
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(5)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        bar = QWidget()
        bar.setStyleSheet("background:#252525; border-top:1px solid #3a3a3a;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(8)

        btn_open = QPushButton("Open Video")
        btn_open.clicked.connect(self.open_file)
        btn_open.setMinimumHeight(28)
        bl.addWidget(btn_open)

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        self.play_btn.setMinimumHeight(28)
        bl.addWidget(self.play_btn)

        btn_exp = QPushButton("Export (Ctrl+S)")
        btn_exp.clicked.connect(self.export_video)
        btn_exp.setMinimumHeight(28)
        bl.addWidget(btn_exp)
        bl.addStretch()
        layout.addWidget(bar)

        if not _CV2:
            warn = QLabel(
                "opencv-python not installed — video features unavailable")
            warn.setAlignment(Qt.AlignCenter)
            warn.setStyleSheet("color:#cc4444; font-size:11px; padding:4px;")
            layout.addWidget(warn)

    def open_file(self):
        if not _CV2:
            QMessageBox.warning(
                self, "Missing dependency",
                "Install opencv-python to use video features.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", self.last_dir,
            "Video Files (*.mp4 *.avi *.mov);;All Files (*.*)")
        if not path:
            return
        if self.video_cap:
            self.video_cap.release()
        self.video_cap  = cv2.VideoCapture(path)
        self.video_path = path
        if not self.video_cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open video.")
            return
        fps   = self.video_cap.get(cv2.CAP_PROP_FPS)
        total = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        p     = Path(path)
        self.last_dir = str(p.parent)
        self.info_lbl.setText(f"{p.name}  —  {total} frames @ {fps:.1f} fps")
        self.play_btn.setEnabled(True)
        self.status_message.emit(f"Loaded: {p.name}")
        self._next_frame()

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self._play_timer.stop()
            self.play_btn.setText("Play")
        else:
            if not self.video_cap:
                return
            fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 25
            self.is_playing = True
            self.play_btn.setText("Pause")
            self._play_timer.start(int(1000 / fps))

    def _next_frame(self):
        if not self.video_cap or not self.video_cap.isOpened():
            return
        ret, frame = self.video_cap.read()
        if not ret:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.is_playing:
                self._next_frame()
            return
        self.current_frame = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._show(self.current_frame)

    def _show(self, img):
        p = self.get_params()
        try:
            dith = apply_dither(
                img, p['pixel_size'], p['threshold'],
                p['color'], p['method'],
                p['brightness'], p['contrast'], p['blur'], p['sharpen'],
                p['glow_radius'], p['glow_intensity'])
            self.canvas.set_image(_pil_to_pixmap(dith))
            self.canvas.setStyleSheet("")
        except Exception as exc:
            self.status_message.emit(f"Frame error: {exc}")

    def export_video(self):
        if not _CV2:
            QMessageBox.warning(
                self, "Missing dependency",
                "Install opencv-python to export video.")
            return
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "No video loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Video",
            str(Path(self.last_dir) / "dithered.mp4"),
            "MP4 Files (*.mp4)")
        if not path:
            return
        if self.is_playing:
            self.toggle_play()
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        p = self.get_params()
        self.export_worker = VideoExportWorker(
            self.video_path, path,
            p['pixel_size'], p['threshold'], p['color'], p['method'],
            p['brightness'], p['contrast'], p['blur'], p['sharpen'],
            p['glow_radius'], p['glow_intensity'])
        self.export_worker.frame_ready.connect(
            lambda img: self.canvas.set_image(_pil_to_pixmap(img)))
        self.export_worker.progress.connect(self._on_progress)
        self.export_worker.finished.connect(self._on_export_done)
        self.export_worker.error.connect(
            lambda msg: QMessageBox.critical(self, "Error", msg))
        self.export_worker.finished.connect(self.export_worker.deleteLater)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_message.emit("Exporting video...")
        self.export_worker.start()

    def _on_progress(self, cur, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(cur)
        self.status_message.emit(f"Exporting {cur}/{total} frames")

    def _on_export_done(self):
        self.progress_bar.setVisible(False)
        self.status_message.emit("Export complete")
        QMessageBox.information(self, "Done", "Video exported successfully.")
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def cleanup(self):
        """Stop playback and any running export, release video handle."""
        if self.is_playing:
            self.toggle_play()
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.stop()
            self.export_worker.wait(10000)
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None

    def zoom_in(self):    self.canvas.zoom_in()
    def zoom_out(self):   self.canvas.zoom_out()
    def fit(self):        self.canvas.fit()
    def actual(self):     self.canvas.actual()
    def zoom_level(self): return self.canvas.zoom_level


# ---------------------------------------------------------------------------
# Shared right-side control panel
# ---------------------------------------------------------------------------

class ControlPanel(QWidget):
    params_changed = Signal()

    def __init__(self):
        super().__init__()
        self.current_color = (101, 138, 0)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(6, 6, 6, 6)

        # Algorithm
        ag = QGroupBox("Algorithm")
        al = QVBoxLayout()
        al.setSpacing(6)
        self.method_combo = QComboBox()
        self.method_combo.addItems(METHODS)
        self.method_combo.setMinimumHeight(28)
        self.method_combo.currentTextChanged.connect(self.params_changed)
        al.addWidget(self.method_combo)
        ag.setLayout(al)
        layout.addWidget(ag)

        # Dither parameters
        dg = QGroupBox("Dither")
        dl = QVBoxLayout()
        dl.setSpacing(4)
        self.pixel_lbl,  self.pixel_sl  = self._slider(dl, "Pixel Size: 4",  1,   20,   4)
        self.thresh_lbl, self.thresh_sl = self._slider(dl, "Threshold: 128", 0,  255, 128)
        dg.setLayout(dl)
        layout.addWidget(dg)

        # Adjustments
        adj = QGroupBox("Adjustments")
        al2 = QVBoxLayout()
        al2.setSpacing(4)
        self.bright_lbl, self.bright_sl = self._slider(al2, "Brightness: 1.0", 0, 200, 100)
        self.contr_lbl,  self.contr_sl  = self._slider(al2, "Contrast: 1.0",   0, 200, 100)
        self.blur_lbl,   self.blur_sl   = self._slider(al2, "Blur: 0",          0,  10,   0)
        self.sharp_lbl,  self.sharp_sl  = self._slider(al2, "Sharpen: 0",       0,   5,   0)
        btn_reset = QPushButton("Reset")
        btn_reset.clicked.connect(self._reset)
        btn_reset.setMinimumHeight(26)
        al2.addWidget(btn_reset)
        adj.setLayout(al2)
        layout.addWidget(adj)

        # Glow
        gg = QGroupBox("Glow")
        gl = QVBoxLayout()
        gl.setSpacing(4)
        self.glow_radius_lbl, self.glow_radius_sl = self._slider(
            gl, "Radius: 0", 0, 40, 0)
        self.glow_intens_lbl, self.glow_intens_sl = self._slider(
            gl, "Intensity: 0", 0, 100, 0)
        gg.setLayout(gl)
        layout.addWidget(gg)

        # Color
        cg = QGroupBox("Color")
        cl = QVBoxLayout()
        cl.setSpacing(6)
        self.swatch = QLabel()
        self.swatch.setFixedHeight(32)
        self._refresh_swatch()
        cl.addWidget(self.swatch)
        btn_color = QPushButton("Pick Color")
        btn_color.clicked.connect(self._pick_color)
        btn_color.setMinimumHeight(28)
        cl.addWidget(btn_color)
        cg.setLayout(cl)
        layout.addWidget(cg)

        layout.addStretch()

        ver = QLabel("Dither Guy v3.3")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet("font-size:10px; color:#444; padding:6px;")
        layout.addWidget(ver)

        pairs = [
            (self.pixel_sl,       lambda v: self.pixel_lbl.setText(f"Pixel Size: {v}")),
            (self.thresh_sl,      lambda v: self.thresh_lbl.setText(f"Threshold: {v}")),
            (self.bright_sl,      lambda v: self.bright_lbl.setText(f"Brightness: {v/100:.1f}")),
            (self.contr_sl,       lambda v: self.contr_lbl.setText(f"Contrast: {v/100:.1f}")),
            (self.blur_sl,        lambda v: self.blur_lbl.setText(f"Blur: {v}")),
            (self.sharp_sl,       lambda v: self.sharp_lbl.setText(f"Sharpen: {v}")),
            (self.glow_radius_sl, lambda v: self.glow_radius_lbl.setText(f"Radius: {v}")),
            (self.glow_intens_sl, lambda v: self.glow_intens_lbl.setText(f"Intensity: {v}")),
        ]
        for sl, lbl_fn in pairs:
            sl.valueChanged.connect(lbl_fn)
            sl.valueChanged.connect(lambda _: self.params_changed.emit())

    @staticmethod
    def _slider(layout, text, mn, mx, val):
        lbl = QLabel(text)
        sl  = QSlider(Qt.Horizontal)
        sl.setMinimum(mn)
        sl.setMaximum(mx)
        sl.setValue(val)
        layout.addWidget(lbl)
        layout.addWidget(sl)
        return lbl, sl

    def _reset(self):
        self.bright_sl.setValue(100)
        self.contr_sl.setValue(100)
        self.blur_sl.setValue(0)
        self.sharp_sl.setValue(0)
        self.glow_radius_sl.setValue(0)
        self.glow_intens_sl.setValue(0)

    def _pick_color(self):
        c = QColorDialog.getColor(
            QColor(*self.current_color), self, "Pick Color")
        if c.isValid():
            self.current_color = (c.red(), c.green(), c.blue())
            self._refresh_swatch()
            self.params_changed.emit()

    def _refresh_swatch(self):
        r, g, b = self.current_color
        self.swatch.setStyleSheet(
            f"background-color:rgb({r},{g},{b});"
            "border:2px solid #555; border-radius:4px;")

    def get_params(self):
        return {
            "method":         self.method_combo.currentText(),
            "pixel_size":     self.pixel_sl.value(),
            "threshold":      self.thresh_sl.value(),
            "brightness":     self.bright_sl.value() / 100.0,
            "contrast":       self.contr_sl.value()  / 100.0,
            "blur":           self.blur_sl.value(),
            "sharpen":        self.sharp_sl.value(),
            "color":          self.current_color,
            "glow_radius":    self.glow_radius_sl.value(),
            "glow_intensity": self.glow_intens_sl.value(),
        }


# ---------------------------------------------------------------------------
# Dark theme
# ---------------------------------------------------------------------------

_THEME = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
}
QToolBar {
    background-color: #252525;
    border-bottom: 1px solid #3a3a3a;
    padding: 3px; spacing: 6px;
}
QToolBar QLabel { color: #658a00; font-weight: bold; padding: 0 6px; }
QTabWidget::pane  { border: 1px solid #3a3a3a; }
QTabBar::tab {
    background: #2a2a2a; border: 1px solid #3a3a3a;
    padding: 7px 20px; border-bottom: none;
}
QTabBar::tab:selected { background: #1e1e1e; color: #8ab800; }
QTabBar::tab:hover    { background: #333; }
QGroupBox {
    border: 1px solid #3a3a3a; border-radius: 5px;
    margin-top: 8px; padding-top: 8px;
    font-weight: bold; font-size: 11px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton {
    background-color: #2d2d2d; border: 1px solid #3a3a3a;
    border-radius: 4px; padding: 6px; font-weight: 500;
}
QPushButton:hover   { background-color: #3a3a3a; border-color: #4a4a4a; }
QPushButton:pressed { background-color: #252525; }
QPushButton:disabled { background-color: #1e1e1e; color: #555; }
QComboBox {
    background-color: #2d2d2d; border: 1px solid #3a3a3a;
    border-radius: 4px; padding: 4px;
}
QComboBox:hover { border-color: #4a4a4a; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #e0e0e0;
    margin-right: 6px;
}
QComboBox QAbstractItemView {
    background-color: #2d2d2d; border: 1px solid #3a3a3a;
    selection-background-color: #658a00;
}
QCheckBox { spacing: 6px; }
QCheckBox::indicator {
    width: 15px; height: 15px; border-radius: 3px;
    border: 1px solid #3a3a3a; background: #2d2d2d;
}
QCheckBox::indicator:checked { background: #658a00; border-color: #658a00; }
QSlider::groove:horizontal {
    height: 5px; background: #2d2d2d; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #658a00; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
QSlider::handle:horizontal:hover { background: #7aa800; }
QScrollArea { border: 1px solid #3a3a3a; background-color: #1a1a1a; }
QScrollBar:vertical, QScrollBar:horizontal {
    background: #2d2d2d; width: 10px; height: 10px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #4a4a4a; border-radius: 5px; min-height: 20px;
}
QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover { background: #5a5a5a; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QProgressBar {
    border: 1px solid #3a3a3a; border-radius: 3px;
    background: #2d2d2d; text-align: center;
}
QProgressBar::chunk { background: #658a00; border-radius: 2px; }
QStatusBar { background-color: #252525; color: #888; font-size: 11px; }
QSplitter::handle { background: #3a3a3a; width: 1px; }
QSpinBox {
    background-color: #2d2d2d; border: 1px solid #3a3a3a;
    border-radius: 4px; padding: 4px;
}
QSpinBox:hover { border-color: #4a4a4a; }
QSpinBox::up-button, QSpinBox::down-button {
    background: #3a3a3a; width: 18px; border-radius: 2px;
}
QDialog { background-color: #1e1e1e; }
"""


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class DitherGuy(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dither Guy")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(_THEME)
        self._load_icon()

        self.controls  = ControlPanel()
        self.image_tab = ImageTab(self.controls.get_params)
        self.video_tab = VideoTab(self.controls.get_params)

        self.controls.params_changed.connect(self._on_params_changed)
        self.image_tab.status_message.connect(self.statusBar().showMessage)
        self.video_tab.status_message.connect(self.statusBar().showMessage)

        self._build_ui()
        self.statusBar().showMessage("Ready — load an image or video")

    def _load_icon(self):
        for name in ("app_icon.png", "app_png.icon"):
            p = Path(name)
            if p.exists():
                self.setWindowIcon(QIcon(str(p)))
                break

    def _build_ui(self):
        self._build_toolbar()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.image_tab, "Image")
        self.tabs.addTab(self.video_tab, "Video")
        self.tabs.currentChanged.connect(lambda _: self._update_zoom_lbl())
        splitter.addWidget(self.tabs)

        ctrl_scroll = QScrollArea()
        ctrl_scroll.setWidgetResizable(True)
        ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        ctrl_scroll.setWidget(self.controls)
        ctrl_scroll.setMinimumWidth(260)
        ctrl_scroll.setMaximumWidth(320)
        splitter.addWidget(ctrl_scroll)

        splitter.setSizes([900, 280])

    def _build_toolbar(self):
        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        def act(label, shortcut, slot):
            a = QAction(label, self)
            if shortcut:
                a.setShortcut(shortcut)
            a.triggered.connect(slot)
            tb.addAction(a)

        act("Open",    "Ctrl+O", self._open)
        act("Save",    "Ctrl+S", self._save)
        tb.addSeparator()
        act("Zoom In",  "Ctrl++", self._zoom_in)
        act("Zoom Out", "Ctrl+-", self._zoom_out)
        act("Fit",      "Ctrl+0", self._fit)
        act("1:1",      "Ctrl+1", self._actual)
        tb.addSeparator()
        act("Undo",     "Ctrl+Z", lambda: self.image_tab.undo())
        tb.addSeparator()

        self.zoom_lbl = QLabel("Fit")
        self.zoom_lbl.setMinimumWidth(50)
        self.zoom_lbl.setAlignment(Qt.AlignCenter)
        tb.addWidget(self.zoom_lbl)

    def _active(self):
        return self.image_tab if self.tabs.currentIndex() == 0 \
               else self.video_tab

    def _on_params_changed(self):
        if self.tabs.currentIndex() == 0:
            self.image_tab.schedule()

    def _open(self):  self._active().open_file()

    def _save(self):
        if self.tabs.currentIndex() == 0:
            self.image_tab.save_file()
        else:
            self.video_tab.export_video()

    def _zoom_in(self):
        self._active().zoom_in()
        self._update_zoom_lbl()

    def _zoom_out(self):
        self._active().zoom_out()
        self._update_zoom_lbl()

    def _fit(self):
        self._active().fit()
        self.zoom_lbl.setText("Fit")

    def _actual(self):
        self._active().actual()
        self._update_zoom_lbl()

    def _update_zoom_lbl(self):
        z = self._active().zoom_level()
        self.zoom_lbl.setText("Fit" if z == 0 else f"{int(z * 100)}%")

    def closeEvent(self, event):
        """Wait for any running threads before letting Qt destroy widgets."""
        self.image_tab.cleanup()

        if self.image_tab.worker and self.image_tab.worker.isRunning():
            # Still not done after 10 s — defer the close.
            self.statusBar().showMessage("Waiting for worker to stop...")
            event.ignore()
            return

        self.video_tab.cleanup()
        super().closeEvent(event)


# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Dither Guy")
    w = DitherGuy()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
