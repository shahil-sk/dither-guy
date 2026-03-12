# dither_guy/ui_widgets.py
# Small reusable Qt widgets and layout helpers that are shared across
# multiple UI panels.
#
# Contents
# ────────
#   _pil_to_pixmap()    — convert PIL Image → QPixmap
#   _hsep() / _vsep()   — thin horizontal / vertical separator frames
#   _make_slider()      — labelled QSlider with inline value display
#   HistogramWidget     — 64-bin luminance histogram rendered with QPainter
#   ZoomableLabel       — QLabel canvas with Ctrl+scroll zoom and fit/actual helpers
#   CropDialog          — modal dialog for per-edge pixel cropping

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui  import (
    QColor, QImage, QLinearGradient, QPainter, QBrush, QPen, QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from utils.theme import (
    _G0, _G1, _G2, _G3, _P0, _P2, _P4, _P5, _P6,
    _FG2, _FG3, _MONO_FONT,
)


# ---------------------------------------------------------------------------
# PIL ↔ Qt conversions
# ---------------------------------------------------------------------------

def _pil_to_pixmap(img: Image.Image) -> QPixmap:
    """Convert a PIL RGB Image to a QPixmap (no temporary file)."""
    raw  = img.tobytes("raw", "RGB")
    qimg = QImage(raw, img.width, img.height,
                  img.width * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _hsep() -> QFrame:
    """One-pixel horizontal separator line."""
    f = QFrame(); f.setFrameShape(QFrame.HLine); f.setFrameShadow(QFrame.Sunken)
    return f


def _vsep() -> QFrame:
    """One-pixel vertical separator line."""
    f = QFrame(); f.setFrameShape(QFrame.VLine); f.setFrameShadow(QFrame.Sunken)
    f.setFixedWidth(1); return f


def _make_slider(parent_layout, label_base: str, mn: int, mx: int, val: int,
                 fmt: str = "{v}"):
    """
    Append a labelled QSlider row to *parent_layout*.

    Returns (label_widget, value_label_widget, slider_widget).
    The caller is responsible for connecting slider signals.
    """
    row = QWidget(); rl = QHBoxLayout(row)
    rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(4)

    lbl = QLabel(label_base)
    lbl.setStyleSheet(
        f"color:{_FG2}; font-size:11px; font-family:{_MONO_FONT};")
    lbl.setFixedWidth(76)

    val_lbl = QLabel(fmt.format(v=val))
    val_lbl.setStyleSheet(
        f"color:{_G0}; font-size:11px; font-family:{_MONO_FONT};"
        "font-weight:bold; min-width:36px;")
    val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    rl.addWidget(lbl); rl.addStretch(); rl.addWidget(val_lbl)
    parent_layout.addWidget(row)

    sl = QSlider(Qt.Horizontal)
    sl.setMinimum(mn); sl.setMaximum(mx)
    sl.setValue(val);  sl.setFixedHeight(18)
    parent_layout.addWidget(sl)

    return lbl, val_lbl, sl


# ---------------------------------------------------------------------------
# HistogramWidget
# ---------------------------------------------------------------------------

class HistogramWidget(QWidget):
    """64-bin luminance histogram rendered with a green gradient fill."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self._data: Optional[np.ndarray] = None
        self.setStyleSheet(f"background:{_P2}; border-radius:3px;")

    def update_data(self, img: Image.Image) -> None:
        arr = np.array(img.convert('L'), dtype=np.uint8)
        hist, _ = np.histogram(arr, bins=64, range=(0, 256))
        self._data = hist.astype(np.float32)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._data is None: return
        p  = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mx   = max(self._data.max(), 1)
        n    = len(self._data)
        bw   = w / n
        grad = QLinearGradient(0, h, 0, 0)
        grad.setColorAt(0.0, QColor(_G3))
        grad.setColorAt(1.0, QColor(_G0))
        p.setBrush(QBrush(grad)); p.setPen(Qt.NoPen)
        for i, v in enumerate(self._data):
            bh = int((v / mx) * (h - 2))
            p.drawRect(int(i * bw), h - bh, max(1, int(bw) - 1), bh)
        p.end()


# ---------------------------------------------------------------------------
# ZoomableLabel
# ---------------------------------------------------------------------------

class ZoomableLabel(QLabel):
    """QLabel canvas that supports fit / actual-size / Ctrl+scroll zooming."""

    def __init__(self, placeholder: str = ""):
        super().__init__(placeholder)
        self.zoom_level            = 0          # 0 = "fit"
        self.original_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 260)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, pixmap: QPixmap) -> None:
        self.original_pixmap = pixmap
        self._redraw()

    def _redraw(self) -> None:
        if not self.original_pixmap: return
        if self.zoom_level == 0:
            scaled = self.original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            w = int(self.original_pixmap.width()  * self.zoom_level)
            h = int(self.original_pixmap.height() * self.zoom_level)
            mode = Qt.SmoothTransformation if self.zoom_level < 2 else Qt.FastTransformation
            scaled = self.original_pixmap.scaled(w, h, Qt.KeepAspectRatio, mode)
        self.setPixmap(scaled)

    def zoom_in(self)  -> None:
        if self.zoom_level == 0: self.zoom_level = 1.0
        self.zoom_level = min(self.zoom_level * 1.25, 8.); self._redraw()

    def zoom_out(self) -> None:
        if self.zoom_level == 0: self.zoom_level = 1.0
        self.zoom_level = max(self.zoom_level * 0.8, 0.1); self._redraw()

    def fit(self)    -> None: self.zoom_level = 0;   self._redraw()
    def actual(self) -> None: self.zoom_level = 1.0; self._redraw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.zoom_level == 0 and self.original_pixmap: self._redraw()

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            self.zoom_in() if event.angleDelta().y() > 0 else self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)


# ---------------------------------------------------------------------------
# CropDialog
# ---------------------------------------------------------------------------

class CropDialog(QDialog):
    """Modal dialog for specifying per-edge pixel crops."""

    def __init__(self, img_w: int, img_h: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Image")
        self.setMinimumWidth(320)
        self.img_w = img_w; self.img_h = img_h
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10); layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Remove pixels from each edge")
        title.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:13px; color:{_G0};")
        layout.addWidget(title); layout.addWidget(_hsep())

        self._info = QLabel()
        self._info.setAlignment(Qt.AlignCenter)
        self._info.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:11px; color:{_G1};"
            f"padding:6px; background:{_P2}; border-radius:3px;")
        layout.addWidget(self._info)

        form = QWidget(); fl = QGridLayout(form); fl.setSpacing(8)
        self._spins: dict[str, QSpinBox] = {}
        for row, (name, label, mx) in enumerate([
            ("left",   "Left",   self.img_w - 1),
            ("top",    "Top",    self.img_h - 1),
            ("right",  "Right",  self.img_w - 1),
            ("bottom", "Bottom", self.img_h - 1),
        ]):
            lbl = QLabel(label); lbl.setStyleSheet(f"color:{_FG2};")
            sp  = QSpinBox(); sp.setMinimum(0); sp.setMaximum(max(mx, 1))
            sp.setValue(0); sp.setSuffix(" px"); sp.setMinimumHeight(28)
            sp.valueChanged.connect(self._update_info)
            fl.addWidget(lbl, row, 0); fl.addWidget(sp, row, 1)
            self._spins[name] = sp
        layout.addWidget(form); layout.addWidget(_hsep())

        btns = QHBoxLayout()
        ok  = QPushButton("Apply Crop"); ok.setObjectName("accent")
        can = QPushButton("Cancel")
        ok.setMinimumHeight(30); can.setMinimumHeight(30)
        ok.clicked.connect(self.accept); can.clicked.connect(self.reject)
        btns.addStretch(); btns.addWidget(can); btns.addWidget(ok)
        layout.addLayout(btns)
        self._update_info()

    def _update_info(self):
        l, t, r, b = (self._spins[k].value() for k in ("left","top","right","bottom"))
        nw = max(1, self.img_w - l - r); nh = max(1, self.img_h - t - b)
        self._info.setText(f"{self.img_w}×{self.img_h}  ──▶  {nw}×{nh}")

    def values(self) -> dict[str, int]:
        return {k: sp.value() for k, sp in self._spins.items()}
