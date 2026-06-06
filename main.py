from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout,
    QSplitter, QTabWidget, QToolBar, QLabel, QStackedWidget
)

from utils.constants import METHODS, VERSION, _VIDEO_WORKERS
from utils.palettes import PALETTES
from utils.theme import THEME, _G0, _G1, _G2, _G3, _P0, _P2, _P3, _P4, _P5, _FG, _MONO_FONT, _SANS_FONT
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
    return "  \u00b7  ".join(parts)


class DitherGuy(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Dither guy  v{VERSION}")
        self.setMinimumSize(960, 620)
        self.setStyleSheet(THEME)
        self._load_icon()

        self.image_controls = ControlPanel()
        self.video_controls = ControlPanel()
        
        self.image_tab = ImageTab(self.image_controls.get_params)
        self.video_tab = VideoTab(self.video_controls.get_params)

        self.image_controls.params_changed.connect(self._on_params_changed)
        self.image_controls.params_changed_preview.connect(self._on_params_preview)
        
        self.video_controls.params_changed.connect(self._on_video_params_changed)
        self.video_controls.params_changed_preview.connect(self._on_video_params_changed)

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

        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.image_tab)
        self.view_stack.addWidget(self.video_tab)
        splitter.addWidget(self.view_stack)

        ctrl_container = QWidget()
        ctrl_container.setMinimumWidth(240)
        ctrl_container.setMaximumWidth(300)
        ctrl_container.setStyleSheet(f"background:{_P2};")
        from PySide6.QtWidgets import QVBoxLayout
        cl = QVBoxLayout(ctrl_container)
        cl.setContentsMargins(0, 0, 0, 0)
        
        self.ctrl_stack = QStackedWidget()
        self.ctrl_stack.addWidget(self.image_controls)
        self.ctrl_stack.addWidget(self.video_controls)
        cl.addWidget(self.ctrl_stack)
        
        splitter.addWidget(ctrl_container)
        splitter.setSizes([860, 280])
        splitter.setCollapsible(1, False)

    def _build_toolbar(self):
        # 1. Top Menu Bar (Nav Bar)
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        view_menu = menubar.addMenu("View")
        edit_menu = menubar.addMenu("Edit")
        
        def create_action(label, shortcut, slot, tip=""):
            a = QAction(label, self)
            if shortcut:
                a.setShortcut(shortcut)
            if tip:
                a.setStatusTip(tip)
                a.setToolTip(tip)
            a.triggered.connect(slot)
            return a
            
        a_open  = create_action("Open...", "Ctrl+O", self._open, "Open image")
        a_save  = create_action("Save As...", "Ctrl+S", self._save, "Save output")
        a_batch = create_action("Batch Process...", "Ctrl+B", self._batch, "Batch process folder")
        file_menu.addAction(a_open)
        file_menu.addAction(a_save)
        file_menu.addSeparator()
        file_menu.addAction(a_batch)
        
        a_undo = create_action("Undo", "Ctrl+Z", lambda: self.image_tab.undo(), "Undo last image operation")
        a_reset = create_action("Reset Defaults", "Ctrl+D", self._reset_all, "Reset settings")
        a_rand = create_action("Randomize", "Ctrl+R", self._randomize, "Randomize properties")
        edit_menu.addAction(a_undo)
        edit_menu.addSeparator()
        edit_menu.addAction(a_reset)
        edit_menu.addAction(a_rand)
        
        a_zin  = create_action("Zoom In", "Ctrl+=", self._zoom_in, "Zoom in")
        a_zout = create_action("Zoom Out", "Ctrl+-", self._zoom_out, "Zoom out")
        a_fit  = create_action("Fit on Screen", "Ctrl+0", self._fit, "Fit to window")
        a_1to1 = create_action("100%", "Ctrl+1", self._actual, "Actual pixels")
        view_menu.addAction(a_zin)
        view_menu.addAction(a_zout)
        view_menu.addAction(a_fit)
        view_menu.addAction(a_1to1)

        # 2. Top Navbar
        tb = QToolBar("Navbar")
        tb.setMovable(False)
        tb.setOrientation(Qt.Horizontal)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(Qt.TopToolBarArea, tb)

        brand = QLabel("Dg")
        brand.setAlignment(Qt.AlignCenter)
        brand.setStyleSheet(
            f"font-family:{_SANS_FONT}; color:#31c4f3; font-weight:bold;"
            f"font-size:14px; padding:2px 6px; margin: 2px 12px 2px 4px; border: 2px solid #31c4f3; border-radius: 4px;"
            f"background: #1e1e1e;"
        )
        tb.addWidget(brand)

        def nav_btn(icon_text, text, action):
            a = QAction(f"{icon_text} {text}", self)
            a.setToolTip(action.toolTip() or action.text())
            a.triggered.connect(action.trigger)
            tb.addAction(a)

        nav_btn("📂", "Open", a_open)
        nav_btn("💾", "Save", a_save)
        nav_btn("⚡", "Batch", a_batch)
        tb.addSeparator()
        nav_btn("↩", "Undo", a_undo)
        tb.addSeparator()
        nav_btn("⤢", "Fit", a_fit)
        nav_btn("🔍", "Zoom In", a_zin)
        nav_btn("🔎", "Zoom Out", a_zout)
        tb.addSeparator()

        self.zoom_lbl = QLabel("fit")
        self.zoom_lbl.setMinimumWidth(52)
        self.zoom_lbl.setAlignment(Qt.AlignCenter)
        self.zoom_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; color:{_G0}; font-size:11px;"
            f"background:{_P2}; border:1px solid {_P5}; border-radius:2px;"
            "padding:2px 8px; margin:2px;"
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
        return self.image_tab if self.view_stack.currentIndex() == 0 else self.video_tab

    def _active_controls(self):
        return self.image_controls if self.view_stack.currentIndex() == 0 else self.video_controls

    def _on_params_changed(self):
        if self.view_stack.currentIndex() == 0:
            self.image_tab.schedule(preview=False)

    def _on_params_preview(self):
        if self.view_stack.currentIndex() == 0:
            self.image_tab.schedule(preview=True)

    def _on_video_params_changed(self):
        if self.view_stack.currentIndex() == 1:
            if self.video_tab.current_frame is not None and not self.video_tab.is_playing:
                self.video_tab._show(self.video_tab.current_frame)

    def _open(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Media", self._active().last_dir,
            "Media Files (*.png *.jpg *.jpeg *.webp *.bmp *.mp4 *.avi *.mov *.mkv *.webm);;Images (*.png *.jpg *.jpeg *.webp *.bmp);;Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All (*.*)"
        )
        if not path:
            return
        
        ext = path.lower().split('.')[-1]
        if ext in ('mp4', 'avi', 'mov', 'mkv', 'webm'):
            self.view_stack.setCurrentIndex(1)
            self.ctrl_stack.setCurrentIndex(1)
            self.video_tab.last_dir = str(Path(path).parent)
            self.video_tab._load_video(path)
        else:
            self.view_stack.setCurrentIndex(0)
            self.ctrl_stack.setCurrentIndex(0)
            self.image_tab.last_dir = str(Path(path).parent)
            self.image_tab._load(path)
            
        self._update_zoom_lbl()

    def _save(self):
        if self.view_stack.currentIndex() == 0:
            self.image_tab.save_file()
        else:
            self.video_tab.export_video()

    def _randomize(self):
        self._active_controls().randomize()
        self._show_status("randomized")

    def _reset_all(self):
        if self.view_stack.currentIndex() == 0:
            self.image_controls.reset_all()
        else:
            self.video_controls.reset_all()
        self._show_status("reset to defaults")

    def _batch(self):
        dlg = BatchDialog(self._active_controls().get_params, self)
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
    import argparse
    import faulthandler
    import traceback
    from PySide6.QtWidgets import QMessageBox

    parser = argparse.ArgumentParser(description="Dither Guy")
    parser.add_argument("--debug", action="store_true", help="Enable developer debug mode")
    args = parser.parse_args()

    if args.debug:
        faulthandler.enable()
        print("Debug mode enabled. Faulthandler active.", file=sys.stderr)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("Dither Guy")
    app.setApplicationVersion(VERSION)

    if args.debug:
        def debug_excepthook(exc_type, exc_value, exc_tb):
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            print("UNHANDLED EXCEPTION:\n" + tb_str, file=sys.stderr)
            # Cannot safely use QMessageBox if Qt event loop crashed, but we can try
            try:
                QMessageBox.critical(None, "Fatal Debug Error", f"App crashed:\n\n{exc_value}\n\nCheck terminal for full stacktrace.")
            except Exception:
                pass
            sys.exit(1)
        sys.excepthook = debug_excepthook

    w = DitherGuy()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
