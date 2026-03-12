from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import (
    QPixmap, QImage, QColor, QPainter, QLinearGradient, QBrush, QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QLabel, QFrame, QSizePolicy

from .theme import _P0, _P2, _P4, _P5, _G0, _G3, _MONO_FONT


# ---------------------------------------------------------------------------
# PIL → QPixmap
# ---------------------------------------------------------------------------

def pil_to_pixmap(img: Image.Image) -> QPixmap:
    raw  = img.tobytes("raw", "RGB")
    qimg = QImage(raw, img.width, img.height,
                  img.width * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Separator helpers
# ---------------------------------------------------------------------------

def hsep() -> QFrame:
    f = QFrame(); f.setFrameShape(QFrame.HLine); f.setFrameShadow(QFrame.Sunken)
    return f


def vsep() -> QFrame:
    f = QFrame(); f.setFrameShape(QFrame.VLine); f.setFrameShadow(QFrame.Sunken)
    f.setFixedWidth(1); return f


# ---------------------------------------------------------------------------
# Histogram widget
# ---------------------------------------------------------------------------

class HistogramWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)
        self._data: Optional[np.ndarray] = None
        self.setStyleSheet(f"background:{_P2}; border-radius:3px;")

    def update_data(self, img: Image.Image):
        arr = np.array(img.convert('L'), dtype=np.uint8)
        hist, _ = np.histogram(arr, bins=64, range=(0, 256))
        self._data = hist.astype(np.float32)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._data is None: return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mx = max(self._data.max(), 1)
        n  = len(self._data)
        bw = w / n
        grad = QLinearGradient(0, h, 0, 0)
        grad.setColorAt(0.0, QColor(_G3))
        grad.setColorAt(1.0, QColor(_G0))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        for i, v in enumerate(self._data):
            bh = int((v / mx) * (h - 2))
            p.drawRect(int(i * bw), h - bh, max(1, int(bw) - 1), bh)
        p.end()


# ---------------------------------------------------------------------------
# Zoomable canvas
# ---------------------------------------------------------------------------

class ZoomableLabel(QLabel):
    def __init__(self, placeholder=""):
        super().__init__(placeholder)
        self.zoom_level = 0
        self.original_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 260)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        self._redraw()

    def _redraw(self):
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

    def zoom_in(self):
        if self.zoom_level == 0: self.zoom_level = 1.
        self.zoom_level = min(self.zoom_level * 1.25, 8.); self._redraw()

    def zoom_out(self):
        if self.zoom_level == 0: self.zoom_level = 1.
        self.zoom_level = max(self.zoom_level * .8, .1); self._redraw()

    def fit(self):    self.zoom_level = 0;   self._redraw()
    def actual(self): self.zoom_level = 1.0; self._redraw()

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
# Compact slider factory
# ---------------------------------------------------------------------------

def make_slider(parent_layout, label_base: str, mn: int, mx: int, val: int,
                fmt: str = "{v}"):
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QSlider

    row = QWidget(); rl = QHBoxLayout(row)
    rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(4)
    lbl = QLabel(label_base)
    lbl.setStyleSheet(f"color:#888888; font-size:11px; font-family:{_MONO_FONT};")
    lbl.setFixedWidth(76)
    val_lbl = QLabel(fmt.format(v=val))
    val_lbl.setStyleSheet(
        f"color:{_G0}; font-size:11px; font-family:{_MONO_FONT};"
        "font-weight:bold; min-width:36px;")
    val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    rl.addWidget(lbl); rl.addStretch(); rl.addWidget(val_lbl)
    parent_layout.addWidget(row)

    sl = QSlider(Qt.Horizontal); sl.setMinimum(mn); sl.setMaximum(mx)
    sl.setValue(val); sl.setFixedHeight(18); parent_layout.addWidget(sl)
    return lbl, val_lbl, sl
