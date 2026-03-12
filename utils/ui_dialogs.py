from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSpinBox, QWidget,
    QLineEdit, QFileDialog, QProgressBar, QMessageBox, QApplication,
)

from .batch import batch_process, _BATCH_IMAGE_EXTS
from .constants import _VIDEO_WORKERS
from .theme import _G0, _FG2, _P2, _MONO_FONT
from .ui_widgets import hsep


class CropDialog(QDialog):
    def __init__(self, img_w: int, img_h: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Image")
        self.setMinimumWidth(320)
        self.img_w, self.img_h = img_w, img_h
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10); layout.setContentsMargins(16, 16, 16, 16)
        title = QLabel("Remove pixels from each edge")
        title.setStyleSheet(f"font-family:{_MONO_FONT}; font-size:13px; color:{_G0};")
        layout.addWidget(title); layout.addWidget(hsep())

        self._info = QLabel()
        self._info.setAlignment(Qt.AlignCenter)
        self._info.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:11px; color:#00cc33;"
            f"padding:6px; background:{_P2}; border-radius:3px;")
        layout.addWidget(self._info)

        form = QWidget(); fl = QGridLayout(form); fl.setSpacing(8)
        self._spins: dict = {}
        for row, (name, label, mx) in enumerate([
            ("left",   "Left",   self.img_w - 1),
            ("top",    "Top",    self.img_h - 1),
            ("right",  "Right",  self.img_w - 1),
            ("bottom", "Bottom", self.img_h - 1),
        ]):
            lbl = QLabel(label); lbl.setStyleSheet(f"color:{_FG2};")
            sp  = QSpinBox(); sp.setMinimum(0); sp.setMaximum(max(mx, 1))
            sp.setValue(0); sp.setSuffix(" px"); sp.setMinimumHeight(28)
            sp.valueChanged.connect(self._update_info)
            fl.addWidget(lbl, row, 0); fl.addWidget(sp, row, 1); self._spins[name] = sp
        layout.addWidget(form); layout.addWidget(hsep())

        btns = QHBoxLayout()
        ok  = QPushButton("Apply Crop"); can = QPushButton("Cancel")
        ok.setObjectName("accent"); ok.setMinimumHeight(30); can.setMinimumHeight(30)
        ok.clicked.connect(self.accept); can.clicked.connect(self.reject)
        btns.addStretch(); btns.addWidget(can); btns.addWidget(ok)
        layout.addLayout(btns); self._update_info()

    def _update_info(self):
        l, t, r, b = (self._spins[k].value() for k in ("left", "top", "right", "bottom"))
        nw = max(1, self.img_w - l - r); nh = max(1, self.img_h - t - b)
        self._info.setText(f"{self.img_w}×{self.img_h}  ──▶  {nw}×{nh}")

    def values(self):
        return {k: sp.value() for k, sp in self._spins.items()}


class BatchDialog(QDialog):
    def __init__(self, get_params, parent=None):
        super().__init__(parent)
        self.get_params = get_params
        self.setWindowTitle("Batch Process")
        self.setMinimumWidth(460)
        self._cancel = [False]
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10); layout.setContentsMargins(16, 16, 16, 16)
        title = QLabel("Batch Dither — apply current settings to a folder")
        title.setStyleSheet(f"font-family:{_MONO_FONT}; font-size:12px; color:{_G0};")
        layout.addWidget(title); layout.addWidget(hsep())

        row_in = QHBoxLayout()
        self.in_edit = QLineEdit(); self.in_edit.setPlaceholderText("Input folder…")
        self.in_edit.setMinimumHeight(28)
        btn_in = QPushButton("Browse"); btn_in.setMinimumHeight(28)
        btn_in.clicked.connect(lambda: self._browse(self.in_edit))
        row_in.addWidget(QLabel("Input :")); row_in.addWidget(self.in_edit); row_in.addWidget(btn_in)
        layout.addLayout(row_in)

        row_out = QHBoxLayout()
        self.out_edit = QLineEdit(); self.out_edit.setPlaceholderText("Output folder…")
        self.out_edit.setMinimumHeight(28)
        btn_out = QPushButton("Browse"); btn_out.setMinimumHeight(28)
        btn_out.clicked.connect(lambda: self._browse(self.out_edit))
        row_out.addWidget(QLabel("Output:")); row_out.addWidget(self.out_edit); row_out.addWidget(btn_out)
        layout.addLayout(row_out)

        self.info_lbl = QLabel("Waiting…")
        self.info_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:10px; color:{_FG2};"
            f"background:{_P2}; padding:6px; border-radius:3px;")
        layout.addWidget(self.info_lbl)

        self.prog = QProgressBar()
        self.prog.setVisible(False); self.prog.setFixedHeight(6); self.prog.setTextVisible(False)
        layout.addWidget(self.prog); layout.addWidget(hsep())

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
        self.info_lbl.setText(f"Processing {len(files)} images with {_VIDEO_WORKERS} workers…")
        QApplication.processEvents()

        def progress(done, total, name):
            self.prog.setValue(done)
            short = name[:40] if len(name) > 40 else name
            self.info_lbl.setText(f"[{done}/{total}]  {short}")
            QApplication.processEvents()

        ok, err = batch_process(in_dir, out_dir, params,
                                progress_cb=progress, cancel_flag=self._cancel)
        self.prog.setVisible(False)
        self.run_btn.setEnabled(True); self.cancel_btn.setVisible(False)
        status = "cancelled" if self._cancel[0] else "complete"
        self.info_lbl.setText(
            f"Batch {status}  ·  {ok} saved  ·  {err} error(s)  →  {out_dir}")

    def _cancel_batch(self):
        self._cancel[0] = True
        self.info_lbl.setText("Cancelling…")
