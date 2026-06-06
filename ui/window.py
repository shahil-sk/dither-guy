from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QSplitter, QStackedWidget, QVBoxLayout
)

from utils.constants import VERSION
from utils.theme import THEME, _P2
from utils.ui_control_panel import ControlPanel
from utils.ui_tabs import ImageTab, VideoTab

from ui.toolbar import WindowToolbar
from ui.handlers import WindowHandlers

class DitherGuy(QMainWindow, WindowHandlers, WindowToolbar):
    def __init__(self) -> None:
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

    def _show_status(self, msg: str) -> None:
        self.statusBar().showMessage(f"  {msg}")

    def closeEvent(self, event) -> None:
        self.image_tab.cleanup()
        self.video_tab.cleanup()
        event.accept()

    def _load_icon(self) -> None:
        for name in ("app_icon.png", "app_icon.ico"):
            p = Path(name)
            if p.exists():
                self.setWindowIcon(QIcon(str(p)))
                return

    def _build_ui(self) -> None:
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
        
        cl = QVBoxLayout(ctrl_container)
        cl.setContentsMargins(0, 0, 0, 0)
        
        self.ctrl_stack = QStackedWidget()
        self.ctrl_stack.addWidget(self.image_controls)
        self.ctrl_stack.addWidget(self.video_controls)
        cl.addWidget(self.ctrl_stack)
        
        splitter.addWidget(ctrl_container)
        splitter.setSizes([860, 280])
        splitter.setCollapsible(1, False)
