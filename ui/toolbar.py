from __future__ import annotations
from typing import TYPE_CHECKING

from PySide6.QtGui import QAction

if TYPE_CHECKING:
    from ui.window import DitherGuy

class WindowToolbar:
    def _build_toolbar(self: 'DitherGuy') -> None:
        # 1. Top Menu Bar (Nav Bar)
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        view_menu = menubar.addMenu("View")
        edit_menu = menubar.addMenu("Edit")
        help_menu = menubar.addMenu("Help")
        
        def create_action(label: str, shortcut: str, slot, tip: str = "") -> QAction:
            a = QAction(label, self)
            if shortcut:
                a.setShortcut(shortcut)
            if tip:
                a.setStatusTip(tip)
                a.setToolTip(tip)
            a.triggered.connect(slot)
            return a
            
        a_open  = create_action("Open", "Ctrl+O", self._open, "Open image")
        a_save  = create_action("Save As", "Ctrl+S", self._save, "Save output")
        a_batch = create_action("Batch Process", "Ctrl+B", self._batch, "Batch process folder")
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

        a_about = create_action("About Dither Guy...", "", self._show_about, "About this application")
        help_menu.addAction(a_about)
