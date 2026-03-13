from __future__ import annotations

from pathlib import Path

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from utils.constants import METHODS, VERSION, _VIDEO_WORKERS
from utils.palettes import PALETTES
from utils.ui_control_panel import ControlPanel
from utils.ui_tabs import ImageTab, VideoTab

try:
    from utils.dither_kernels import _NUMBA
except ImportError:
    _NUMBA = False


class DitherGuy(toga.App):
    def startup(self):
        self.status_label = toga.Label(
            "",
            style=Pack(padding=(2, 8), font_size=10),
        )

        self.image_tab = ImageTab(self._get_params, self._show_status)
        self.video_tab = VideoTab(self._get_params, self._show_status)
        self.control_panel = ControlPanel(self._on_params_changed)

        # Tab container
        self.tabs = toga.OptionContainer(
            content=[
                ("Image", self.image_tab.build()),
                ("Video", self.video_tab.build()),
            ],
            style=Pack(flex=1),
        )

        # Main split: tabs on left, controls on right
        main_row = toga.Box(
            children=[
                self.tabs,
                self.control_panel.build(),
            ],
            style=Pack(direction=ROW, flex=1),
        )

        # Toolbar-like top bar
        top_bar = toga.Box(
            children=[
                toga.Label(
                    f"DITHER GUY  v{VERSION}",
                    style=Pack(padding=(4, 14), font_size=13, font_weight="bold"),
                ),
                toga.Button("Open",   on_press=self._open,    style=Pack(padding=4)),
                toga.Button("Save",   on_press=self._save,    style=Pack(padding=4)),
                toga.Button("Batch",  on_press=self._batch,   style=Pack(padding=4)),
                toga.Button("Zoom +", on_press=self._zoom_in,  style=Pack(padding=4)),
                toga.Button("Zoom -", on_press=self._zoom_out, style=Pack(padding=4)),
                toga.Button("Fit",    on_press=self._fit,      style=Pack(padding=4)),
                toga.Button("1:1",    on_press=self._actual,   style=Pack(padding=4)),
                toga.Button("Undo",   on_press=self._undo,     style=Pack(padding=4)),
                self.status_label,
            ],
            style=Pack(direction=ROW, padding=4),
        )

        root = toga.Box(
            children=[top_bar, main_row],
            style=Pack(direction=COLUMN, flex=1),
        )

        self.main_window = toga.MainWindow(title=f"DITHER GUY  v{VERSION}")
        self.main_window.content = root
        self.main_window.show()

        jit_tag = "  numba JIT active" if _NUMBA else ""
        self._show_status(
            f"ready  {len(METHODS)} algorithms  {len(PALETTES)} palettes"
            f"  {_VIDEO_WORKERS} workers{jit_tag}")

    def _get_params(self) -> dict:
        return self.control_panel.get_params()

    def _show_status(self, msg: str):
        self.status_label.text = f"  {msg}"

    def _active_tab(self):
        return (
            self.image_tab
            if self.tabs.current_tab.text == "Image"
            else self.video_tab
        )

    def _on_params_changed(self):
        if self.tabs.current_tab.text == "Image":
            self.image_tab.schedule()

    def _open(self, widget):
        self._active_tab().open_file(self.main_window)

    def _save(self, widget):
        self._active_tab().save_file(self.main_window)

    def _batch(self, widget):
        from utils.ui_dialogs import show_batch_dialog
        show_batch_dialog(self.main_window, self._get_params)

    def _zoom_in(self, widget):  self._active_tab().zoom_in()
    def _zoom_out(self, widget): self._active_tab().zoom_out()
    def _fit(self, widget):      self._active_tab().fit()
    def _actual(self, widget):   self._active_tab().actual()
    def _undo(self, widget):     self.image_tab.undo()


def main():
    return DitherGuy(
        formal_name="Dither Guy",
        app_id="com.ditherGuy.app",
        app_name="dither_guy",
    )


if __name__ == "__main__":
    main().main_loop()
