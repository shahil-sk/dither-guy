# dither_guy/ui_dialogs.py
# Modal dialogs that are not part of the main window layout.
#
# Contents
# ────────
#   BatchDialog — folder-to-folder batch processing dialog; shows a progress
#                 bar and supports mid-run cancellation.

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore  import Qt
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from utils.batch      import batch_process, _BATCH_IMAGE_EXTS
from utils.constants  import _VIDEO_WORKERS
from utils.theme      import _FG2, _G0, _MONO_FONT, _P2
from utils.ui_widgets import _hsep


class BatchDialog(QDialog):
    """Apply the current dither settings to every image in a folder."""

    def __init__(self, get_params, parent=None):
        super().__init__(parent)
        self.get_params = get_params
        self.setWindowTitle("Batch Process")
        self.setMinimumWidth(460)
        self._cancel = [False]
        self._build()

    def _build(self):
        layout = QVBoxLayout(self); layout.setSpacing(10); layout.setContentsMargins(16,16,16,16)

        title = QLabel("Batch Dither — apply current settings to a folder")
        title.setStyleSheet(f"font-family:{_MONO_FONT}; font-size:12px; color:{_G0};")
        layout.addWidget(title); layout.addWidget(_hsep())

        # Input folder row
        row_in = QHBoxLayout()
        self.in_edit = QLineEdit(); self.in_edit.setPlaceholderText("Input folder…")
        self.in_edit.setMinimumHeight(28)
        btn_in = QPushButton("Browse"); btn_in.setMinimumHeight(28)
        btn_in.clicked.connect(lambda: self._browse(self.in_edit))
        row_in.addWidget(QLabel("Input :")); row_in.addWidget(self.in_edit); row_in.addWidget(btn_in)
        layout.addLayout(row_in)

        # Output folder row
        row_out = QHBoxLayout()
        self.out_edit = QLineEdit(); self.out_edit.setPlaceholderText("Output folder…")
        self.out_edit.setMinimumHeight(28)
        btn_out = QPushButton("Browse"); btn_out.setMinimumHeight(28)
        btn_out.clicked.connect(lambda: self._browse(self.out_edit))
        row_out.addWidget(QLabel("Output:")); row_out.addWidget(self.out_edit); row_out.addWidget(btn_out)
        layout.addLayout(row_out)

        # Status label
        self.info_lbl = QLabel("Waiting…")
        self.info_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:10px; color:{_FG2};"
            f"background:{_P2}; padding:6px; border-radius:3px;")
        layout.addWidget(self.info_lbl)

        # Progress bar
        self.prog = QProgressBar()
        self.prog.setVisible(False); self.prog.setFixedHeight(6); self.prog.setTextVisible(False)
        layout.addWidget(self.prog)

        layout.addWidget(_hsep())

        # Buttons
        btns = QHBoxLayout()
        self.run_btn = QPushButton("▶ Run Batch"); self.run_btn.setObjectName("accent")
        self.run_btn.setMinimumHeight(32); self.run_btn.clicked.connect(self._run)

        self.cancel_btn = QPushButton("✕ Cancel"); self.cancel_btn.setMinimumHeight(32)
        self.cancel_btn.clicked.connect(self._cancel_batch); self.cancel_btn.setVisible(False)

        close_btn = QPushButton("Close"); close_btn.setMinimumHeight(32)
        close_btn.clicked.connect(self.reject)

        btns.addWidget(self.run_btn); btns.addWidget(self.cancel_btn)
        btns.addStretch(); btns.addWidget(close_btn)
        layout.addLayout(btns)

    def _browse(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select Folder", str(Path.home()))
        if d: edit.setText(d)

    def _run(self):
        in_dir  = self.in_edit.text().strip()
        out_dir = self.out_edit.text().strip()
        if not in_dir or not out_dir:
            QMessageBox.warning(self, "Batch", "Please set both input and output folders."); return
        if not Path(in_dir).is_dir():
            QMessageBox.warning(self, "Batch", f"Input folder not found:\n{in_dir}"); return

        files = [f for f in Path(in_dir).iterdir()
                 if f.suffix.lower() in _BATCH_IMAGE_EXTS]
        if not files:
            QMessageBox.warning(self, "Batch", "No supported images found in input folder."); return

        self._cancel[0] = False
        params = self.get_params()
        self.prog.setMaximum(len(files)); self.prog.setValue(0)
        self.prog.setVisible(True)
        self.run_btn.setEnabled(False); self.cancel_btn.setVisible(True)
        self.info_lbl.setText(
            f"Processing {len(files)} images with {_VIDEO_WORKERS} workers…")
        QApplication.processEvents()

        def progress(done: int, total: int, name: str):
            self.prog.setValue(done)
            self.info_lbl.setText(f"[{done}/{total}]  {name[:40]}")
            QApplication.processEvents()

        ok, err = batch_process(in_dir, out_dir, params,
                                progress_cb=progress,
                                cancel_flag=self._cancel)

        self.prog.setVisible(False)
        self.run_btn.setEnabled(True); self.cancel_btn.setVisible(False)
        status = "cancelled" if self._cancel[0] else "complete"
        self.info_lbl.setText(
            f"Batch {status}  ·  {ok} saved  ·  {err} error(s)  →  {out_dir}")

    def _cancel_batch(self):
        self._cancel[0] = True; self.info_lbl.setText("Cancelling…")
