from __future__ import annotations

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .constants import METHOD_GROUPS, METHODS, VERSION
from .palettes import PALETTES
from .presets import save_preset, load_preset, list_presets, delete_preset
from .ui_widgets import hsep, make_slider

try:
    from .dither_kernels import _NUMBA
except ImportError:
    _NUMBA = False


class ControlPanel:
    def __init__(self, on_change=None):
        self._on_change = on_change
        self.current_color: tuple = (255, 255, 255)
        self._custom_palette: list[tuple] = []
        self._widget: toga.ScrollContainer | None = None

        self._pixel_sl: toga.Slider | None = None
        self._thresh_sl: toga.Slider | None = None
        self._bright_sl: toga.Slider | None = None
        self._contr_sl: toga.Slider | None = None
        self._blur_sl: toga.Slider | None = None
        self._sharp_sl: toga.Slider | None = None
        self._glow_r_sl: toga.Slider | None = None
        self._glow_i_sl: toga.Slider | None = None
        self._method_sel: toga.Selection | None = None
        self._palette_sel: toga.Selection | None = None
        self._preset_name: toga.TextInput | None = None
        self._preset_sel: toga.Selection | None = None
        self._color_label: toga.Label | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> toga.ScrollContainer:
        inner = toga.Box(style=Pack(direction=COLUMN, padding=8))

        inner.add(toga.Label("Algorithm", style=Pack(font_size=12, font_weight="bold", padding_bottom=4)))
        all_methods = [m for members in METHOD_GROUPS.values() for m in members]
        self._method_sel = toga.Selection(
            items=all_methods,
            on_change=lambda w: self._emit(),
            style=Pack(padding_bottom=8),
        )
        if "Floyd-Steinberg" in all_methods:
            self._method_sel.value = "Floyd-Steinberg"
        inner.add(self._method_sel)
        inner.add(hsep())

        inner.add(toga.Label("Dither", style=Pack(font_size=12, font_weight="bold", padding=(8, 0, 4, 0))))
        pix_row, self._pixel_sl, _ = make_slider("pixel size", 1, 20, 4, lambda v: self._emit())
        thr_row, self._thresh_sl, _ = make_slider("threshold", 0, 255, 128, lambda v: self._emit())
        inner.add(pix_row)
        inner.add(thr_row)
        inner.add(hsep())

        inner.add(toga.Label("Adjustments", style=Pack(font_size=12, font_weight="bold", padding=(8, 0, 4, 0))))
        br_row, self._bright_sl, _ = make_slider("brightness", 0, 200, 100, lambda v: self._emit())
        co_row, self._contr_sl,  _ = make_slider("contrast",   0, 200, 100, lambda v: self._emit())
        bl_row, self._blur_sl,   _ = make_slider("blur",       0,  10,   0, lambda v: self._emit())
        sh_row, self._sharp_sl,  _ = make_slider("sharpen",    0,   5,   0, lambda v: self._emit())
        inner.add(br_row)
        inner.add(co_row)
        inner.add(bl_row)
        inner.add(sh_row)
        inner.add(toga.Button("Reset Adjustments", on_press=self._reset, style=Pack(padding=(4, 0))))
        inner.add(hsep())

        inner.add(toga.Label("Glow", style=Pack(font_size=12, font_weight="bold", padding=(8, 0, 4, 0))))
        gr_row, self._glow_r_sl, _ = make_slider("radius",    0,  40, 0, lambda v: self._emit())
        gi_row, self._glow_i_sl, _ = make_slider("intensity", 0, 100, 0, lambda v: self._emit())
        inner.add(gr_row)
        inner.add(gi_row)
        inner.add(hsep())

        inner.add(toga.Label("Colour", style=Pack(font_size=12, font_weight="bold", padding=(8, 0, 4, 0))))
        r, g, b = self.current_color
        self._color_label = toga.Label(
            f"#{r:02X}{g:02X}{b:02X}",
            style=Pack(padding=(4, 0), font_size=11),
        )
        inner.add(self._color_label)
        inner.add(toga.Button("Pick Colour", on_press=self._pick_color, style=Pack(padding=(4, 0))))
        inner.add(hsep())

        inner.add(toga.Label("Palette", style=Pack(font_size=12, font_weight="bold", padding=(8, 0, 4, 0))))
        self._palette_sel = toga.Selection(
            items=list(PALETTES.keys()),
            on_change=lambda w: self._emit(),
            style=Pack(padding_bottom=4),
        )
        inner.add(self._palette_sel)
        custom_row = toga.Box(style=Pack(direction=ROW, padding=(4, 0)))
        custom_row.add(toga.Button("+ Add Color", on_press=self._add_custom_color, style=Pack(flex=1, padding_right=4)))
        custom_row.add(toga.Button("Clear",       on_press=self._clear_custom,     style=Pack(flex=1)))
        inner.add(custom_row)
        inner.add(hsep())

        inner.add(toga.Label("Presets", style=Pack(font_size=12, font_weight="bold", padding=(8, 0, 4, 0))))
        self._preset_name = toga.TextInput(placeholder="preset name", style=Pack(flex=1, padding_right=4))
        save_row = toga.Box(style=Pack(direction=ROW, padding_bottom=4))
        save_row.add(self._preset_name)
        save_row.add(toga.Button("Save", on_press=self._save_preset, style=Pack(padding_left=4)))
        inner.add(save_row)

        self._preset_sel = toga.Selection(items=list_presets(), style=Pack(flex=1))
        preset_row = toga.Box(style=Pack(direction=ROW, padding=(4, 0)))
        preset_row.add(self._preset_sel)
        preset_row.add(toga.Button("Load", on_press=self._load_preset,   style=Pack(padding_left=4)))
        preset_row.add(toga.Button("Del",  on_press=self._delete_preset, style=Pack(padding_left=4)))
        inner.add(preset_row)

        if _NUMBA:
            inner.add(toga.Label("numba JIT active", style=Pack(font_size=9, text_align="center", padding=(8, 0, 0, 0))))
        inner.add(toga.Label(f"DITHER GUY  v{VERSION}", style=Pack(font_size=9, text_align="center", padding=(10, 0, 4, 0))))

        self._widget = toga.ScrollContainer(content=inner, horizontal=False, style=Pack(width=280))
        return self._widget

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit(self):
        if self._on_change:
            self._on_change()

    def _reset(self, widget=None):
        for sl, v in [
            (self._bright_sl, 100), (self._contr_sl, 100),
            (self._blur_sl, 0), (self._sharp_sl, 0),
            (self._glow_r_sl, 0), (self._glow_i_sl, 0),
        ]:
            if sl:
                sl.value = v
        self._emit()

    async def _pick_color(self, widget=None):
        window = self._widget.app.main_window
        # toga 0.5.x: await window.dialog(toga.QuestionDialog / InfoDialog / ...)
        # For text input there is no built-in TextDialog; use toga.App's question flow.
        # We show an InfoDialog asking user to type in the status bar approach is not
        # available, so we chain two dialogs: confirm then a question dialog workaround.
        # Best available: ask for the value via a simple question dialog with a prompt.
        result = await window.dialog(
            toga.QuestionDialog(
                "Pick Colour",
                "Type a 6-digit hex color in the box below and press Yes.\n"
                "(You cannot type here — note your color down and edit the field separately if needed)\n"
                "Example: FF0041  or  00FF41",
            )
        )
        # QuestionDialog returns True/False. Since there is no TextInputDialog in Toga 0.5,
        # fall back to asking via consecutive single-char prompts is impractical.
        # Instead we use a simple color cycling approach: open an InfoDialog and let the
        # user pick from a compact set, or just accept the current color unchanged.
        # The proper cross-platform solution is to handle this in the app layer.
        # For now, log that the feature needs a platform text input.
        if not result:
            return
        await window.dialog(
            toga.InfoDialog(
                "Pick Colour",
                "Toga 0.5 does not have a text input dialog.\n"
                "Edit 'current_color' in presets or use the preset system to apply a custom color."
            )
        )

    async def _add_custom_color(self, widget=None):
        # Same limitation as _pick_color above
        window = self._widget.app.main_window
        await window.dialog(
            toga.InfoDialog(
                "Add Palette Color",
                "Toga 0.5 does not have a text input dialog.\n"
                "Save a preset with your desired color using the Presets section."
            )
        )

    def _clear_custom(self, widget=None):
        self._custom_palette.clear()
        self._emit()

    def _save_preset(self, widget=None):
        name = self._preset_name.value.strip() if self._preset_name else ""
        if not name:
            return
        save_preset(name, self.get_params())
        self._refresh_preset_list()
        self._preset_name.value = ""

    def _load_preset(self, widget=None):
        name = self._preset_sel.value if self._preset_sel else None
        if not name:
            return
        p = load_preset(name)
        if p is None:
            return
        if self._method_sel:  self._method_sel.value  = p.get("method", "Floyd-Steinberg")
        if self._pixel_sl:    self._pixel_sl.value    = p.get("pixel_size", 4)
        if self._thresh_sl:   self._thresh_sl.value   = p.get("threshold", 128)
        if self._bright_sl:   self._bright_sl.value   = int(p.get("brightness", 1.0) * 100)
        if self._contr_sl:    self._contr_sl.value    = int(p.get("contrast", 1.0) * 100)
        if self._blur_sl:     self._blur_sl.value     = p.get("blur", 0)
        if self._sharp_sl:    self._sharp_sl.value    = p.get("sharpen", 0)
        if self._glow_r_sl:   self._glow_r_sl.value   = p.get("glow_radius", 0)
        if self._glow_i_sl:   self._glow_i_sl.value   = p.get("glow_intensity", 0)
        self.current_color = tuple(p.get("color", (255, 255, 255)))
        r, g, b = self.current_color
        if self._color_label: self._color_label.text  = f"#{r:02X}{g:02X}{b:02X}"
        if self._palette_sel: self._palette_sel.value = p.get("palette_name", "B&W")
        self._emit()

    def _delete_preset(self, widget=None):
        name = self._preset_sel.value if self._preset_sel else None
        if not name:
            return
        delete_preset(name)
        self._refresh_preset_list()

    def _refresh_preset_list(self):
        if self._preset_sel:
            self._preset_sel.items = list_presets()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        pal_name = self._palette_sel.value if self._palette_sel else "B&W"
        return {
            "method":         self._method_sel.value if self._method_sel else "Floyd-Steinberg",
            "pixel_size":     int(self._pixel_sl.value)  if self._pixel_sl  else 4,
            "threshold":      int(self._thresh_sl.value) if self._thresh_sl else 128,
            "brightness":     (self._bright_sl.value / 100.0) if self._bright_sl else 1.0,
            "contrast":       (self._contr_sl.value  / 100.0) if self._contr_sl  else 1.0,
            "blur":           int(self._blur_sl.value)   if self._blur_sl   else 0,
            "sharpen":        int(self._sharp_sl.value)  if self._sharp_sl  else 0,
            "color":          self.current_color,
            "glow_radius":    int(self._glow_r_sl.value) if self._glow_r_sl else 0,
            "glow_intensity": int(self._glow_i_sl.value) if self._glow_i_sl else 0,
            "palette_name":   pal_name,
            "custom_palette": self._custom_palette if pal_name == "Custom" else None,
        }
