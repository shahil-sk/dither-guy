from __future__ import annotations

import io
from typing import Optional

import numpy as np
from PIL import Image

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW


# ---------------------------------------------------------------------------
# PIL image -> toga.Image
# ---------------------------------------------------------------------------

def pil_to_toga_image(img: Image.Image) -> toga.Image:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    # toga 0.5.x uses src= instead of data=
    return toga.Image(src=buf.read())


# ---------------------------------------------------------------------------
# Separators
# ---------------------------------------------------------------------------

def hsep() -> toga.Box:
    return toga.Box(style=Pack(height=1, background_color="#333333", padding=(4, 0)))


def vsep() -> toga.Box:
    return toga.Box(style=Pack(width=1, background_color="#333333", padding=(0, 4)))


# ---------------------------------------------------------------------------
# Zoomable image view
# ---------------------------------------------------------------------------

class ZoomableImageView:
    def __init__(self, placeholder: str = ""):
        self.zoom_level: float = 0.0
        self._original: Optional[Image.Image] = None

        self._label = toga.Label(
            placeholder,
            style=Pack(flex=1, text_align="center", padding=16, font_size=13),
        )
        self._image_view = toga.ImageView(
            style=Pack(flex=1),
        )
        self._image_view.style.visibility = "hidden"

        self.container = toga.Box(
            children=[self._label, self._image_view],
            style=Pack(direction=COLUMN, flex=1),
        )

    def set_image(self, img: Image.Image):
        self._original = img
        self._label.style.visibility = "hidden"
        self._image_view.style.visibility = "visible"
        self._image_view.image = pil_to_toga_image(img)

    def zoom_in(self):
        if self.zoom_level == 0.0:
            self.zoom_level = 1.0
        self.zoom_level = min(self.zoom_level * 1.25, 8.0)

    def zoom_out(self):
        if self.zoom_level == 0.0:
            self.zoom_level = 1.0
        self.zoom_level = max(self.zoom_level * 0.8, 0.1)

    def fit(self):    self.zoom_level = 0.0
    def actual(self): self.zoom_level = 1.0


# ---------------------------------------------------------------------------
# Slider row factory
# ---------------------------------------------------------------------------

def make_slider(
    label_text: str,
    mn: int,
    mx: int,
    val: int,
    on_change=None,
) -> tuple:
    value_label = toga.Label(
        str(val),
        style=Pack(width=40, text_align="right", font_size=11),
    )

    def _slider_changed(widget):
        v = int(widget.value)
        value_label.text = str(v)
        if on_change:
            on_change(v)

    slider = toga.Slider(
        min=mn,
        max=mx,
        value=val,
        on_change=_slider_changed,
        style=Pack(flex=1, padding=(0, 4)),
    )

    row = toga.Box(
        children=[
            toga.Label(label_text, style=Pack(width=96, font_size=11)),
            slider,
            value_label,
        ],
        style=Pack(direction=ROW, padding=(2, 0)),
    )
    return row, slider, value_label
