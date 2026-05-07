from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QSplitter, QTabWidget, QToolBar, QLabel,
)

from utils.constants import METHODS, VERSION, _VIDEO_WORKERS
from utils.palettes import PALETTES
from utils.theme import THEME, _G0, _G1, _G2, _G3, _P0, _P2, _P3, _P4, _P5, _FG, _MONO_FONT
from utils.ui_control_panel import ControlPanel
from utils.ui_tabs import ImageTab, VideoTab
from utils.ui_dialogs import BatchDialog
from utils.gpu_kernels import GPU_BACKEND

try:
    from utils.dither_kernels import _NUMBA
except ImportError:
    _NUMBA = False


def _build_info_str() -> str:
    gpu_tag = {
        "cuda":   "CUDA",
        "opencl": "OpenCL",
        "cpu":    "CPU",
    }.get(GPU_BACKEND, GPU_BACKEND)
    jit_tag = "JIT" if _NUMBA else ""
    parts = [
        f"{len(METHODS)} algo",
        f"{len(PALETTES)} pal",
        f"{_VIDEO_WORKERS}w",
        f"GPU:{gpu_tag}",
    ]
    if jit_tag:
        parts.append(jit_tag)
    return "  ·  ".join(parts)


class DitherGuy(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Dither Man  v{VERSION}")
        self.setMinimumSize(960, 620)
        self.setStyleSheet(THEME)
        self._load_icon()

        self.controls  = ControlPanel()
        self.image_tab = ImageTab(self.controls.get_params)
        self.video_tab = VideoTab(self.controls.get_params)

        self.controls.params_changed.connect(self._on_params_changed)
        self.controls.params_changed_preview.connect(self._on_params_preview)
        self.image_tab.status_message.connect(self._show_status)
        self.video_tab.status_message.connect(self._show_status)

        self._build_ui()
        self._show_status("ready")

    def _show_status(self, msg: str):
        self.statusBar().showMessage(f"  {msg}")

    def _load_icon(self):
        for name in ("app_icon.png", "app_icon.ico"):
            p = Path(name)
            if p.exists():
                self.setWindowIcon(QIcon(str(p)))
                return

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
        self.tabs.addTab(self.image_tab, "  Image  ")
        self.tabs.addTab(self.video_tab, "  Video  ")
        self.tabs.currentChanged.connect(lambda _: self._update_zoom_lbl())
        splitter.addWidget(self.tabs)

        ctrl_container = QWidget()
        ctrl_container.setMinimumWidth(240)
        ctrl_container.setMaximumWidth(300)
        ctrl_container.setStyleSheet(f"background:{_P2};")
        from PySide6.QtWidgets import QVBoxLayout
        cl = QVBoxLayout(ctrl_container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(self.controls)
        splitter.addWidget(ctrl_container)
        splitter.setSizes([860, 280])
        splitter.setCollapsible(1, False)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)

        brand = QLabel("DITHER MAN")
        brand.setStyleSheet(
            f"font-family:{_MONO_FONT}; color:{_FG}; font-weight:bold;"
            "font-size:14px; letter-spacing:4px; padding:0 18px 0 6px;"
        )
        tb.addWidget(brand)

        def act(label, shortcut, slot, tip=""):
            a = QAction(label, self)
            if shortcut:
                a.setShortcut(shortcut)
            if tip:
                a.setStatusTip(tip)
                a.setToolTip(tip)
            a.triggered.connect(slot)
            tb.addAction(a)
            return a

        act("open",  "Ctrl+O", self._open,  "Open image (Ctrl+O)")
        act("save",  "Ctrl+S", self._save,  "Save output (Ctrl+S)")
        act("batch", "Ctrl+B", self._batch, "Batch process folder (Ctrl+B)")
        tb.addSeparator()
        act("zoom +", "Ctrl+=", self._zoom_in,  "Zoom in (Ctrl+=)")
        act("zoom -", "Ctrl+-", self._zoom_out, "Zoom out (Ctrl+-)")
        act("fit",    "Ctrl+0", self._fit,      "Fit to window (Ctrl+0)")
        act("1:1",    "Ctrl+1", self._actual,   "Actual pixel size (Ctrl+1)")
        tb.addSeparator()
        act("undo",  "Ctrl+Z", lambda: self.image_tab.undo(), "Undo last image operation (Ctrl+Z)")
        tb.addSeparator()

        self.zoom_lbl = QLabel("fit")
        self.zoom_lbl.setMinimumWidth(52)
        self.zoom_lbl.setAlignment(Qt.AlignCenter)
        self.zoom_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; color:{_G0}; font-size:11px;"
            f"background:{_P2}; border:1px solid {_P5}; border-radius:2px;"
            "padding:2px 6px; margin:3px;"
        )
        self.zoom_lbl.setToolTip("Current zoom level  (Ctrl+= / Ctrl+- / Ctrl+0 / Ctrl+1)")
        tb.addWidget(self.zoom_lbl)

        tb.addSeparator()
        info_lbl = QLabel(_build_info_str())
        gpu_colour = _G0 if GPU_BACKEND == "cuda" else _G1
        info_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; color:{gpu_colour}; font-size:10px;"
            f"background:{_P2}; border:1px solid {_P5}; border-radius:2px;"
            "padding:2px 8px; margin:3px;"
        )
        info_lbl.setToolTip(
            f"GPU backend : {GPU_BACKEND}\n"
            f"Numba JIT   : {'enabled' if _NUMBA else 'disabled'}\n"
            f"Algorithms  : {len(METHODS)}\n"
            f"Palettes    : {len(PALETTES)}\n"
            f"Video workers: {_VIDEO_WORKERS}"
        )
        tb.addWidget(info_lbl)

    def _active(self):
        return self.image_tab if self.tabs.currentIndex() == 0 else self.video_tab

    def _on_params_changed(self):
        if self.tabs.currentIndex() == 0:
            self.image_tab.schedule(preview=False)

    def _on_params_preview(self):
        if self.tabs.currentIndex() == 0:
            self.image_tab.schedule(preview=True)

    def _open(self):  self._active().open_file()

    def _save(self):
        if self.tabs.currentIndex() == 0:
            self.image_tab.save_file()
        else:
            self.video_tab.export_video()

    def _batch(self):
        dlg = BatchDialog(self.controls.get_params, self)
        dlg.exec()

    def _zoom_in(self):  self._active().zoom_in();  self._update_zoom_lbl()
    def _zoom_out(self): self._active().zoom_out(); self._update_zoom_lbl()
    def _fit(self):      self._active().fit();       self.zoom_lbl.setText("fit")
    def _actual(self):   self._active().actual();    self._update_zoom_lbl()

    def _update_zoom_lbl(self):
        z = self._active().zoom_level
        self.zoom_lbl.setText("fit" if z == 0 else f"{int(z * 100)}%")

    def closeEvent(self, event):
        self.image_tab._stop_worker()
        self.video_tab.closeEvent(event)
        super().closeEvent(event)


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Dither Man")
    app.setApplicationVersion(VERSION)
    w = DitherGuy()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
