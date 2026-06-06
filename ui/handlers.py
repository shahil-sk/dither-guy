from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox
from PySide6.QtCore import Qt, qVersion

from utils.constants import VERSION, METHODS
from utils.ui_dialogs import BatchDialog
from utils.gpu_kernels import GPU_BACKEND

if TYPE_CHECKING:
    from ui.window import DitherGuy
    from utils.ui_tabs import ImageTab, VideoTab
    from utils.ui_control_panel import ControlPanel

class WindowHandlers:
    def _active(self: 'DitherGuy') -> 'ImageTab' | 'VideoTab':
        return self.image_tab if self.view_stack.currentIndex() == 0 else self.video_tab

    def _active_controls(self: 'DitherGuy') -> 'ControlPanel':
        return self.image_controls if self.view_stack.currentIndex() == 0 else self.video_controls

    def _on_params_changed(self: 'DitherGuy') -> None:
        if self.view_stack.currentIndex() == 0:
            self.image_tab.schedule(preview=False)

    def _on_params_preview(self: 'DitherGuy') -> None:
        if self.view_stack.currentIndex() == 0:
            self.image_tab.schedule(preview=True)

    def _on_video_params_changed(self: 'DitherGuy') -> None:
        if self.view_stack.currentIndex() == 1:
            if self.video_tab.current_frame is not None and not self.video_tab.is_playing:
                self.video_tab._show(self.video_tab.current_frame)

    def _show_about(self: 'DitherGuy') -> None:
        about_text = (
            f"<h2 style='color:#31c4f3; margin-bottom: 4px;'>Dither Guy v{VERSION}</h2>"
            "<p>A high-performance GPU-accelerated dithering application.</p>"
            "<hr>"
            "<b>Maintainer:</b> Shahil SK<br>"
            "<b>GitHub:</b> <a href='https://github.com/shahil-sk/dither-guy'>github.com/shahil-sk/dither-guy</a><br>"
            "<b>Report Issues:</b> <a href='https://github.com/shahil-sk/dither-guy/issues'>Issue Tracker</a><br><br>"
            "<b>System & Core:</b><br>"
            f"&#8226; <i>Qt Version:</i> {qVersion()}<br>"
            f"&#8226; <i>Compute Backend:</i> {GPU_BACKEND.upper()}<br>"
            f"&#8226; <i>Algorithms:</i> {len(METHODS)}<br><br>"
            "<span style='color: gray; font-size: 10px;'>Powered by PySide6, PyOpenCL/CUDA, and FFmpeg.</span>"
        )
        msg = QMessageBox(self)
        msg.setWindowTitle("About Dither Guy")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.exec()

    def _open(self: 'DitherGuy') -> None:
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

    def _save(self: 'DitherGuy') -> None:
        if self.view_stack.currentIndex() == 0:
            self.image_tab.save_file()
        else:
            self.video_tab.export_video()

    def _randomize(self: 'DitherGuy') -> None:
        self._active_controls().randomize()
        self._show_status("randomized")

    def _reset_all(self: 'DitherGuy') -> None:
        if self.view_stack.currentIndex() == 0:
            self.image_controls.reset_all()
        else:
            self.video_controls.reset_all()
        self._show_status("reset to defaults")

    def _batch(self: 'DitherGuy') -> None:
        dlg = BatchDialog(self._active_controls().get_params, self)
        dlg.exec()

    def _zoom_in(self: 'DitherGuy') -> None:
        self._active().zoom_in()
        self._update_zoom_lbl()

    def _zoom_out(self: 'DitherGuy') -> None:
        self._active().zoom_out()
        self._update_zoom_lbl()

    def _fit(self: 'DitherGuy') -> None:
        self._active().fit()
        self.zoom_lbl.setText("fit")

    def _actual(self: 'DitherGuy') -> None:
        self._active().actual()
        self._update_zoom_lbl()

    def _update_zoom_lbl(self: 'DitherGuy') -> None:
        z = self._active().zoom_level
        self.zoom_lbl.setText("fit" if z == 0 else f"{int(z * 100)}%")
