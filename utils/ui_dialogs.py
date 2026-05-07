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


# ---------------------------------------------------------------------------
# Crop dialog
# ---------------------------------------------------------------------------

class CropDialog(QDialog):
    """Let the user trim N pixels from each edge of an image."""

    _EDGES = ("left", "top", "right", "bottom")

    def __init__(self, img_w: int, img_h: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Image")
        self.setMinimumWidth(320)
        self.img_w = img_w
        self.img_h = img_h
        self._spins: dict[str, QSpinBox] = {}
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Remove pixels from each edge")
        title.setStyleSheet(f"font-family:{_MONO_FONT}; font-size:13px; color:{_G0};")
        layout.addWidget(title)
        layout.addWidget(hsep())

        self._info = QLabel()
        self._info.setAlignment(Qt.AlignCenter)
        self._info.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:11px; color:#00cc33;"
            f"padding:6px; background:{_P2}; border-radius:3px;"
        )
        layout.addWidget(self._info)

        # Spin-box form
        maxima = {
            "left":   self.img_w - 1,
            "top":    self.img_h - 1,
            "right":  self.img_w - 1,
            "bottom": self.img_h - 1,
        }
        form = QWidget()
        fl   = QGridLayout(form)
        fl.setSpacing(8)
        for row, name in enumerate(self._EDGES):
            lbl = QLabel(name.capitalize())
            lbl.setStyleSheet(f"color:{_FG2};")
            sp  = QSpinBox()
            sp.setMinimum(0)
            sp.setMaximum(max(maxima[name], 1))
            sp.setValue(0)
            sp.setSuffix(" px")
            sp.setMinimumHeight(28)
            sp.valueChanged.connect(self._update_info)
            fl.addWidget(lbl, row, 0)
            fl.addWidget(sp,  row, 1)
            self._spins[name] = sp

        layout.addWidget(form)
        layout.addWidget(hsep())

        btns = QHBoxLayout()
        ok   = QPushButton("Apply Crop")
        can  = QPushButton("Cancel")
        ok.setObjectName("accent")
        ok.setMinimumHeight(30)
        can.setMinimumHeight(30)
        ok.clicked.connect(self.accept)
        can.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(can)
        btns.addWidget(ok)
        layout.addLayout(btns)

        self._update_info()

    def _update_info(self) -> None:
        l = self._spins["left"].value()
        t = self._spins["top"].value()
        r = self._spins["right"].value()
        b = self._spins["bottom"].value()
        nw = max(1, self.img_w - l - r)
        nh = max(1, self.img_h - t - b)
        self._info.setText(f"{self.img_w}×{self.img_h} ──▶ {nw}×{nh}")

    def values(self) -> dict[str, int]:
        return {k: sp.value() for k, sp in self._spins.items()}


# ---------------------------------------------------------------------------
# Batch dialog
# ---------------------------------------------------------------------------

class BatchDialog(QDialog):
    """Apply current dither settings to every image in a folder."""

    def __init__(self, get_params, parent=None):
        super().__init__(parent)
        self.get_params = get_params
        self.setWindowTitle("Batch Process")
        self.setMinimumWidth(460)
        self._cancel = [False]
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Batch Dither — apply current settings to a folder")
        title.setStyleSheet(f"font-family:{_MONO_FONT}; font-size:12px; color:{_G0};")
        layout.addWidget(title)
        layout.addWidget(hsep())

        for attr, placeholder, caption in (
            ("in_edit",  "Input folder…",  "Input :"),
            ("out_edit", "Output folder…", "Output:"),
        ):
            row = QHBoxLayout()
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setMinimumHeight(28)
            btn = QPushButton("Browse")
            btn.setMinimumHeight(28)
            btn.clicked.connect(lambda _=None, e=edit: self._browse(e))
            row.addWidget(QLabel(caption))
            row.addWidget(edit)
            row.addWidget(btn)
            layout.addLayout(row)
            setattr(self, attr, edit)

        self.info_lbl = QLabel("Waiting…")
        self.info_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:10px; color:{_FG2};"
            f"background:{_P2}; padding:6px; border-radius:3px;"
        )
        layout.addWidget(self.info_lbl)

        self.prog = QProgressBar()
        self.prog.setVisible(False)
        self.prog.setFixedHeight(6)
        self.prog.setTextVisible(False)
        layout.addWidget(self.prog)
        layout.addWidget(hsep())

        btns = QHBoxLayout()
        self.run_btn = QPushButton("▶ Run Batch")
        self.run_btn.setObjectName("accent")
        self.run_btn.setMinimumHeight(32)
        self.run_btn.clicked.connect(self._run)

        self.cancel_btn = QPushButton("✕ Cancel")
        self.cancel_btn.setMinimumHeight(32)
        self.cancel_btn.clicked.connect(self._cancel_batch)
        self.cancel_btn.setVisible(False)

        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(32)
        close_btn.clicked.connect(self.reject)

        btns.addWidget(self.run_btn)
        btns.addWidget(self.cancel_btn)
        btns.addStretch()
        btns.addWidget(close_btn)
        layout.addLayout(btns)

    def _browse(self, edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Folder", str(Path.home()))
        if d:
            edit.setText(d)

    def _run(self) -> None:
        in_dir  = self.in_edit.text().strip()
        out_dir = self.out_edit.text().strip()

        if not in_dir or not out_dir:
            QMessageBox.warning(self, "Batch", "Please set both input and output folders.")
            return
        if not Path(in_dir).is_dir():
            QMessageBox.warning(self, "Batch", f"Input folder not found:\n{in_dir}")
            return

        files = [f for f in Path(in_dir).iterdir() if f.suffix.lower() in _BATCH_IMAGE_EXTS]
        if not files:
            QMessageBox.warning(self, "Batch", "No supported images found in input folder.")
            return

        self._cancel[0] = False
        params = self.get_params()
        self.prog.setMaximum(len(files))
        self.prog.setValue(0)
        self.prog.setVisible(True)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setVisible(True)
        self.info_lbl.setText(f"Processing {len(files)} images with {_VIDEO_WORKERS} workers…")
        QApplication.processEvents()

        def _progress(done: int, total: int, name: str) -> None:
            self.prog.setValue(done)
            short = name[:40] if len(name) > 40 else name
            self.info_lbl.setText(f"[{done}/{total}] {short}")
            QApplication.processEvents()

        ok, err = batch_process(
            in_dir, out_dir, params,
            progress_cb=_progress,
            cancel_flag=self._cancel,
        )
        self.prog.setVisible(False)
        self.run_btn.setEnabled(True)
        self.cancel_btn.setVisible(False)
        status = "cancelled" if self._cancel[0] else "complete"
        self.info_lbl.setText(
            f"Batch {status} · {ok} saved · {err} error(s) → {out_dir}"
        )

    def _cancel_batch(self) -> None:
        self._cancel[0] = True
        self.info_lbl.setText("Cancelling…")
