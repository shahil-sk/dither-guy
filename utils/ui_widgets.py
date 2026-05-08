from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QPixmap, QImage, QColor, QPainter, QLinearGradient, QBrush, QWheelEvent,
)
from PySide6.QtWidgets import (
    QLabel, QFrame, QSizePolicy, QWidget, QHBoxLayout, QSlider,
)

from .theme import _P0, _P2, _P4, _P5, _FG, _FG2, _FG3, _G0, _G3, _MONO_FONT


# ---------------------------------------------------------------------------
# PIL → QPixmap
# ---------------------------------------------------------------------------

def pil_to_pixmap(img: Image.Image) -> QPixmap:
    """Convert a PIL RGB image to QPixmap without holding a ref to raw bytes."""
    raw = img.tobytes("raw", "RGB")
    qimg = QImage(raw, img.width, img.height, img.width * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Separator helpers
# ---------------------------------------------------------------------------

def _sep(shape: QFrame.Shape) -> QFrame:
    f = QFrame()
    f.setFrameShape(shape)
    f.setFrameShadow(QFrame.Sunken)
    if shape == QFrame.VLine:
        f.setFixedWidth(1)
    return f

def hsep() -> QFrame:
    return _sep(QFrame.HLine)

def vsep() -> QFrame:
    return _sep(QFrame.VLine)


# ---------------------------------------------------------------------------
# Histogram widget
# ---------------------------------------------------------------------------

class HistogramWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self._data: Optional[np.ndarray] = None
        self.setStyleSheet(
            f"background:{_P2}; border-top:1px solid {_P5}; border-radius:0px;"
        )
        self.setToolTip("Luminance histogram of processed output")

    def update_data(self, img: Image.Image) -> None:
        arr = np.array(img.convert("L"), dtype=np.uint8)
        hist, _ = np.histogram(arr, bins=64, range=(0, 256))
        self._data = hist.astype(np.float32)
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._data is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mx = max(float(self._data.max()), 1.0)
        bw = w / len(self._data)
        grad = QLinearGradient(0, h, 0, 0)
        grad.setColorAt(0.0, QColor(_G3))
        grad.setColorAt(1.0, QColor(_G0))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.NoPen)
        for i, v in enumerate(self._data):
            bh = int((v / mx) * (h - 4))
            painter.drawRect(int(i * bw), h - bh, max(1, int(bw) - 1), bh)
        painter.end()


# ---------------------------------------------------------------------------
# Zoomable canvas
# ---------------------------------------------------------------------------

_ZOOM_IN_FACTOR  = 1.25
_ZOOM_OUT_FACTOR = 0.80
_ZOOM_MIN        = 0.10
_ZOOM_MAX        = 8.00


class ZoomableLabel(QLabel):
    def __init__(self, placeholder: str = ""):
        super().__init__(placeholder)
        self.zoom_level: float = 0
        self.original_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 260)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:13px;"
            f"color:{_FG3}; background:{_P0};"
        )

    # public API

    def set_image(self, pixmap: QPixmap) -> None:
        self.original_pixmap = pixmap
        self._redraw()

    def zoom_in(self)  -> None: self._apply_zoom(self.zoom_level or 1.0, _ZOOM_IN_FACTOR)
    def zoom_out(self) -> None: self._apply_zoom(self.zoom_level or 1.0, _ZOOM_OUT_FACTOR)
    def fit(self)      -> None: self.zoom_level = 0;   self._redraw()
    def actual(self)   -> None: self.zoom_level = 1.0; self._redraw()

    # internals

    def _apply_zoom(self, base: float, factor: float) -> None:
        self.zoom_level = max(_ZOOM_MIN, min(_ZOOM_MAX, base * factor))
        self._redraw()

    def _redraw(self) -> None:
        if not self.original_pixmap:
            return
        if self.zoom_level == 0:
            scaled = self.original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            w = int(self.original_pixmap.width()  * self.zoom_level)
            h = int(self.original_pixmap.height() * self.zoom_level)
            mode = Qt.SmoothTransformation if self.zoom_level < 2 else Qt.FastTransformation
            scaled = self.original_pixmap.scaled(w, h, Qt.KeepAspectRatio, mode)
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.zoom_level == 0 and self.original_pixmap:
            self._redraw()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)


# ---------------------------------------------------------------------------
# Resettable slider
# Double-click resets to default value.
# Scroll-wheel only changes value when _scroll_enabled is True.
# ---------------------------------------------------------------------------

class ResettableSlider(QSlider):
    """QSlider with double-click-to-reset and opt-in scroll-wheel."""

    # Class-level flag shared across all instances; toggled by ControlPanel.
    _scroll_enabled: bool = False

    def __init__(self, orientation, default: int, parent=None):
        super().__init__(orientation, parent)
        self._default = default

    def mouseDoubleClickEvent(self, event) -> None:
        self.setValue(self._default)
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if ResettableSlider._scroll_enabled:
            super().wheelEvent(event)
        else:
            event.ignore()  # pass up to ScrollArea so panel still scrolls


# ---------------------------------------------------------------------------
# Compact slider factory
# ---------------------------------------------------------------------------

def make_slider(
    parent_layout,
    label_base: str,
    mn: int,
    mx: int,
    val: int,
    fmt: str = "{v}",
    tooltip: str = "",
):
    """
    Add a labelled slider row to *parent_layout*.
    Returns (label_widget, value_label_widget, slider_widget).
    """
    row = QWidget()
    rl  = QHBoxLayout(row)
    rl.setContentsMargins(0, 0, 0, 0)
    rl.setSpacing(4)

    lbl = QLabel(label_base)
    lbl.setStyleSheet(
        f"color:{_FG}; font-size:11px; font-family:{_MONO_FONT};"
    )
    lbl.setMinimumWidth(72)
    lbl.setMaximumWidth(100)
    if tooltip:
        lbl.setToolTip(tooltip)

    val_lbl = QLabel(fmt.format(v=val))
    val_lbl.setStyleSheet(
        f"color:{_G0}; font-size:11px; font-family:{_MONO_FONT};"
        "font-weight:bold; min-width:32px;"
    )
    val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    rl.addWidget(lbl)
    rl.addStretch()
    rl.addWidget(val_lbl)
    parent_layout.addWidget(row)

    sl = ResettableSlider(Qt.Horizontal, default=val)
    sl.setMinimum(mn)
    sl.setMaximum(mx)
    sl.setValue(val)
    sl.setFixedHeight(20)
    if tooltip:
        sl.setToolTip(f"{tooltip}  ·  double-click to reset")
    parent_layout.addWidget(sl)

    return lbl, val_lbl, sl
