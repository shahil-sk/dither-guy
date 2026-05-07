from __future__ import annotations

from pathlib import Path
from typing import Optional
import itertools

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QCheckBox, QFileDialog, QMessageBox,
    QProgressBar, QSlider,
)

from .constants import _MAX_PIXELS, _HISTORY_LIMIT, _DEBOUNCE_MS
from .dither_kernels import apply_dither
from .theme import _P0, _P1, _P5, _G3, _FG3, _RE, _MONO_FONT
from .workers import DitherWorker, VideoExportWorker
from .ui_widgets import ZoomableLabel, HistogramWidget, pil_to_pixmap, hsep, vsep

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

_worker_id_counter = itertools.count(1)


# ---------------------------------------------------------------------------
# Image tab
# ---------------------------------------------------------------------------

class ImageTab(QWidget):
    status_message = Signal(str)

    _IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp);;All (*.*)"
    _DROP_EXTS    = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}

    def __init__(self, get_params):
        super().__init__()
        self.get_params = get_params
        self.original_img:  Optional[Image.Image] = None
        self.dithered_img:  Optional[Image.Image] = None
        self.last_dir       = str(Path.home())
        self.worker:        Optional[DitherWorker] = None
        self._worker_id     = 0
        self.auto_update    = True
        self._history:      list[Image.Image] = []
        self._timer         = QTimer(singleShot=True)
        self._timer.timeout.connect(self.process)
        self._preview_timer = QTimer(singleShot=True)
        self._preview_timer.timeout.connect(self._process_preview)
        self._build()
        self.setAcceptDrops(True)

    # ── Build ──────────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.info_lbl = QLabel("Drop an image here · or Ctrl+O")
        self.info_lbl.setAlignment(Qt.AlignCenter)
        self.info_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:10px; padding:4px 8px;"
            f"background:{_P0}; color:{_FG3};"
        )
        layout.addWidget(self.info_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.canvas = ZoomableLabel("▣ Drop image here · Ctrl+O")
        self.canvas.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:14px; color:{_P5}; background:{_P0};"
        )
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll, stretch=1)

        self.histogram = HistogramWidget()
        self.histogram.setVisible(False)
        layout.addWidget(self.histogram)

        bar1 = QWidget()
        bar1.setStyleSheet(f"background:{_P0};")
        bl1 = QHBoxLayout(bar1)
        bl1.setContentsMargins(8, 5, 8, 5)
        bl1.setSpacing(5)

        def _btn(label: str, slot, accent: bool = False, tip: str = "") -> QPushButton:
            b = QPushButton(label)
            if accent:
                b.setObjectName("accent")
            b.clicked.connect(slot)
            b.setMinimumHeight(28)
            b.setToolTip(tip)
            bl1.addWidget(b)
            return b

        _btn("◑ Invert",  self.invert)
        _btn("↺ L",       self.rotate_left,  tip="Rotate 90° CCW")
        _btn("↻ R",       self.rotate_right, tip="Rotate 90° CW")
        _btn("↔ H",       self.flip_h,       tip="Flip horizontal")
        _btn("↕ V",       self.flip_v,       tip="Flip vertical")
        _btn("✂ Crop",    self.crop)

        bl1.addWidget(vsep())
        self.hist_cb = QCheckBox("Histogram")
        self.hist_cb.stateChanged.connect(lambda s: self.histogram.setVisible(bool(s)))
        bl1.addWidget(self.hist_cb)

        bl1.addWidget(vsep())
        self.auto_cb = QCheckBox("Live")
        self.auto_cb.setChecked(True)
        self.auto_cb.setToolTip("Auto-update on parameter change")
        self.auto_cb.stateChanged.connect(self._toggle_auto)
        bl1.addWidget(self.auto_cb)

        self.apply_btn = QPushButton("▶ Apply")
        self.apply_btn.setObjectName("accent")
        self.apply_btn.clicked.connect(self.process)
        self.apply_btn.setVisible(False)
        self.apply_btn.setMinimumHeight(28)
        bl1.addWidget(self.apply_btn)
        bl1.addStretch()

        self.undo_btn = QPushButton("↩ Undo")
        self.undo_btn.clicked.connect(self.undo)
        self.undo_btn.setEnabled(False)
        self.undo_btn.setMinimumHeight(22)
        self.undo_btn.setStyleSheet(
            f"font-size:10px; font-family:{_MONO_FONT}; padding:2px 7px;"
        )
        bl1.addWidget(self.undo_btn)
        layout.addWidget(bar1)

    # ── Drag & drop ─────────────────────────────────────────────────────────────

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).suffix.lower() in self._DROP_EXTS:
                self._load(p)
                break

    # ── Auto-update ────────────────────────────────────────────────────────────

    def _toggle_auto(self, state: int) -> None:
        self.auto_update = bool(state)
        self.apply_btn.setVisible(not self.auto_update)

    def schedule(self, preview: bool = False) -> None:
        if not self.auto_update:
            return
        if preview:
            self._preview_timer.stop()
            self._preview_timer.start(80)
        else:
            self._timer.stop()
            self._timer.start(_DEBOUNCE_MS)

    # ── Processing ─────────────────────────────────────────────────────────────

    def _build_worker(self, preview: bool) -> DitherWorker:
        p = self.get_params()
        return DitherWorker(
            self.original_img,
            p["pixel_size"], p["threshold"], p["color"], p["method"],
            p["brightness"], p["contrast"], p["blur"], p["sharpen"],
            p["glow_radius"], p["glow_intensity"],
            preview=preview,
            palette_name=p.get("palette_name", "B&W"),
            custom_palette=p.get("custom_palette"),
        )

    def _process_preview(self) -> None:
        if self.original_img is None:
            return
        self._stop_worker()
        wid = next(_worker_id_counter)
        self._worker_id = wid
        self.worker = self._build_worker(preview=True)
        self.worker.finished.connect(lambda pl, _id=wid: self._on_done(pl, _id))
        self.worker.error.connect(lambda _: None)
        self.worker.start()

    def process(self) -> None:
        if self.original_img is None:
            return
        self._stop_worker()
        self.status_message.emit("processing...")
        wid = next(_worker_id_counter)
        self._worker_id = wid
        self.worker = self._build_worker(preview=False)
        self.worker.finished.connect(lambda pl, _id=wid: self._on_done(pl, _id))
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _stop_worker(self) -> None:
        if self.worker and self.worker.isRunning():
            self._worker_id = 0
            self.worker.stop()
            if not self.worker.wait(1500):
                self.worker.terminate()
                self.worker.wait(400)
            self.worker.deleteLater()
            self.worker = None

    # ── File I/O ─────────────────────────────────────────────────────────────────

    def _load(self, path: str) -> None:
        try:
            img = Image.open(path)
            if img.width * img.height > _MAX_PIXELS:
                ans = QMessageBox.question(
                    self, "Large Image",
                    f"Image is {img.width}×{img.height}. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if ans != QMessageBox.Yes:
                    return
            self.original_img = img.convert("RGB")
            self._history.clear()
            self.undo_btn.setEnabled(False)
            self.last_dir = str(Path(path).parent)
            self._refresh_info()
            self.status_message.emit(f"loaded {Path(path).name}")
            self.process()
        except Exception as exc:
            QMessageBox.critical(self, "Open Error", f"Failed:\n{exc}")

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self.last_dir, self._IMAGE_FILTER
        )
        if path:
            self._load(path)

    def save_file(self) -> None:
        if self.dithered_img is None:
            QMessageBox.warning(self, "Nothing to save", "Process an image first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image",
            str(Path(self.last_dir) / "dithered.png"),
            "PNG (*.png);;JPEG (*.jpg);;All (*.*)",
        )
        if path:
            try:
                self.dithered_img.save(path)
                self.last_dir = str(Path(path).parent)
                self.status_message.emit(f"saved {Path(path).name}")
            except Exception as exc:
                QMessageBox.critical(self, "Save Error", f"Failed:\n{exc}")

    # ── History ───────────────────────────────────────────────────────────────────

    def _refresh_info(self) -> None:
        if self.original_img:
            w, h = self.original_img.size
            self.info_lbl.setText(f"{w}×{h} px · history: {len(self._history)}")

    def _push_history(self) -> None:
        if self.original_img is None:
            return
        self._history.append(self.original_img.copy())
        if len(self._history) > _HISTORY_LIMIT:
            self._history.pop(0)
        self.undo_btn.setEnabled(True)

    def _require_image(self, op: str = "do this") -> bool:
        if self.original_img is None:
            QMessageBox.warning(self, "No Image", f"Load an image to {op}.")
            return False
        return True

    # ── Image operations ───────────────────────────────────────────────────────────

    def invert(self) -> None:
        if not self._require_image("invert"):
            return
        self._push_history()
        self.original_img = Image.fromarray(
            (255 - np.array(self.original_img)).astype(np.uint8)
        )
        self.status_message.emit("inverted")
        self._refresh_info()
        self.process()

    def rotate_left(self) -> None:
        if not self._require_image("rotate"):
            return
        self._push_history()
        self.original_img = self.original_img.rotate(90, expand=True)
        self.status_message.emit("rotated 90° CCW")
        self._refresh_info()
        self.process()

    def rotate_right(self) -> None:
        if not self._require_image("rotate"):
            return
        self._push_history()
        self.original_img = self.original_img.rotate(-90, expand=True)
        self.status_message.emit("rotated 90° CW")
        self._refresh_info()
        self.process()

    def flip_h(self) -> None:
        if not self._require_image("flip"):
            return
        self._push_history()
        self.original_img = self.original_img.transpose(Image.FLIP_LEFT_RIGHT)
        self.status_message.emit("flipped H")
        self._refresh_info()
        self.process()

    def flip_v(self) -> None:
        if not self._require_image("flip"):
            return
        self._push_history()
        self.original_img = self.original_img.transpose(Image.FLIP_TOP_BOTTOM)
        self.status_message.emit("flipped V")
        self._refresh_info()
        self.process()

    def crop(self) -> None:
        if not self._require_image("crop"):
            return
        from .ui_dialogs import CropDialog
        from PySide6.QtWidgets import QDialog
        dlg = CropDialog(self.original_img.width, self.original_img.height, self)
        if dlg.exec() != QDialog.Accepted:
            return
        v  = dlg.values()
        l, t, r, b = v["left"], v["top"], v["right"], v["bottom"]
        x2 = self.original_img.width  - r
        y2 = self.original_img.height - b
        if l >= x2 or t >= y2:
            QMessageBox.warning(self, "Crop", "Nothing left after crop.")
            return
        self._push_history()
        self.original_img = self.original_img.crop((l, t, x2, y2))
        self.status_message.emit(
            f"cropped → {self.original_img.width}×{self.original_img.height}"
        )
        self._refresh_info()
        self.process()

    def undo(self) -> None:
        if not self._history:
            return
        self.original_img = self._history.pop()
        self.undo_btn.setEnabled(bool(self._history))
        self.status_message.emit("undo")
        self._refresh_info()
        self.process()

    # ── Worker callbacks ───────────────────────────────────────────────────────────

    def _on_done(self, payload, worker_id: int) -> None:
        if worker_id != self._worker_id:
            return
        img, elapsed, is_preview = payload
        if not is_preview:
            self.dithered_img = img
        if self.hist_cb.isChecked():
            self.histogram.update_data(img)
        self.canvas.set_image(pil_to_pixmap(img))
        self.canvas.setStyleSheet(f"background:{_P0};")

    def _on_error(self, msg: str) -> None:
        self.status_message.emit(f"error: {msg}")
        QMessageBox.warning(self, "Processing Error", msg)

    # ── Zoom proxy ─────────────────────────────────────────────────────────────────

    def zoom_in(self)  -> None: self.canvas.zoom_in()
    def zoom_out(self) -> None: self.canvas.zoom_out()
    def fit(self)      -> None: self.canvas.fit()
    def actual(self)   -> None: self.canvas.actual()

    @property
    def zoom_level(self) -> float:
        return self.canvas.zoom_level


# ---------------------------------------------------------------------------
# Video tab
# ---------------------------------------------------------------------------

class VideoTab(QWidget):
    """
    Playback pipeline
    ─────────────────
    QTimer → _next_frame() decodes raw frame (main thread, cheap)
           → _show_async() spawns DitherWorker

    During playback, if previous worker still running:
      • signal stop (non-blocking)
      • skip spawning — current frame dropped, display stays on last rendered
      • worker cleans itself up via finished signal → _retire_worker()
    This keeps QTimer firing at fps rate with zero blocking.

    On close / open_file: _drain_frame_worker() hard-waits up to 1 s before
    releasing resources, preventing the SIGABRT.
    """

    status_message = Signal(str)

    _VIDEO_FILTER = "Video (*.mp4 *.avi *.mov);;All (*.*)"

    def __init__(self, get_params):
        super().__init__()
        self.get_params        = get_params
        self.video_cap         = None
        self.video_path        = None
        self.current_frame     = None
        self.is_playing        = False
        self.export_worker     = None
        self.last_dir          = str(Path.home())
        self._total_frames     = 0
        self._seek_dragging    = False
        self._frame_worker: Optional[DitherWorker] = None
        self._frame_id      = 0
        self._play_timer    = QTimer()
        self._play_timer.timeout.connect(self._next_frame)
        self._build()

    # ── Build ──────────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.info_lbl = QLabel("no video loaded")
        self.info_lbl.setAlignment(Qt.AlignCenter)
        self.info_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:10px; padding:4px;"
            f"background:{_P0}; color:{_FG3};"
        )
        layout.addWidget(self.info_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.canvas = ZoomableLabel("▶ Load a video · MP4 / AVI / MOV")
        self.canvas.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:14px; color:{_P5}; background:{_P0};"
        )
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll, stretch=1)

        # ── Seek bar ────────────────────────────────────────────────────────────────
        seek_row = QWidget()
        seek_row.setStyleSheet(f"background:{_P0};")
        seek_layout = QHBoxLayout(seek_row)
        seek_layout.setContentsMargins(8, 2, 8, 2)
        seek_layout.setSpacing(6)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(0)
        self.seek_slider.setValue(0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setToolTip("Seek")
        self.seek_slider.sliderPressed.connect(self._seek_press)
        self.seek_slider.sliderReleased.connect(self._seek_release)
        seek_layout.addWidget(self.seek_slider, stretch=1)

        self.frame_lbl = QLabel("0 / 0")
        self.frame_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:10px; color:{_FG3};"
            "min-width:72px;"
        )
        self.frame_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        seek_layout.addWidget(self.frame_lbl)
        layout.addWidget(seek_row)

        # ── Export progress ─────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(3)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # ── Control bar ──────────────────────────────────────────────────────────
        bar = QWidget()
        bar.setStyleSheet(f"background:{_P0}; border-top:1px solid {_G3};")
        bl  = QHBoxLayout(bar)
        bl.setContentsMargins(8, 5, 8, 5)
        bl.setSpacing(5)

        def _cbtn(label: str, slot, tip: str = "") -> QPushButton:
            b = QPushButton(label)
            b.clicked.connect(slot)
            b.setEnabled(False)
            b.setMinimumHeight(28)
            if tip:
                b.setToolTip(tip)
            bl.addWidget(b)
            return b

        self.step_back_btn = _cbtn("⧏",       self.step_back,    tip="Step back one frame")
        self.play_btn      = _cbtn("▶ Play",   self.toggle_play)
        self.step_fwd_btn  = _cbtn("⧐",       self.step_forward, tip="Step forward one frame")
        bl.addStretch()
        layout.addWidget(bar)

        if not _CV2:
            warn = QLabel("⚠ opencv-python not installed — video disabled")
            warn.setAlignment(Qt.AlignCenter)
            warn.setStyleSheet(
                f"color:{_RE}; font-family:{_MONO_FONT}; font-size:10px;"
                f"padding:4px; background:{_P1};"
            )
            layout.addWidget(warn)

    # ── File I/O ─────────────────────────────────────────────────────────────────

    def open_file(self) -> None:
        if not _CV2:
            QMessageBox.warning(self, "Missing", "Install opencv-python.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", self.last_dir, self._VIDEO_FILTER
        )
        if not path:
            return
        self._drain_frame_worker()  # hard-wait before releasing cap
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            QMessageBox.critical(self, "Error", "Cannot open video.")
            return
        self.video_cap     = cap
        self.video_path    = path
        fps                = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.last_dir      = str(Path(path).parent)
        self.info_lbl.setText(
            f"{Path(path).name} · {self._total_frames} frames @ {fps:.1f} fps"
        )
        for w in (self.play_btn, self.step_back_btn, self.step_fwd_btn):
            w.setEnabled(True)
        self.seek_slider.setMaximum(self._total_frames - 1)
        self.seek_slider.setEnabled(True)
        self.status_message.emit(f"loaded {Path(path).name}")
        self._next_frame()

    # ── Playback ────────────────────────────────────────────────────────────────

    def toggle_play(self) -> None:
        if self.is_playing:
            self.is_playing = False
            self._play_timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            if not self.video_cap:
                return
            fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.is_playing = True
            self.play_btn.setText("⏸ Pause")
            self._play_timer.start(max(1, int(1000 / fps)))

    def step_back(self) -> None:
        if not self.video_cap:
            return
        was_playing = self.is_playing
        if was_playing:
            self.toggle_play()
        pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._seek_to(max(0, pos - 2))
        if was_playing:
            self.toggle_play()

    def step_forward(self) -> None:
        if not self.video_cap:
            return
        was_playing = self.is_playing
        if was_playing:
            self.toggle_play()
        self._next_frame()
        if was_playing:
            self.toggle_play()

    def _seek_press(self) -> None:
        self._seek_dragging = True
        if self.is_playing:
            self._play_timer.stop()

    def _seek_release(self) -> None:
        self._seek_dragging = False
        self._seek_to(self.seek_slider.value())
        if self.is_playing:
            fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 25.0
            self._play_timer.start(max(1, int(1000 / fps)))

    def _seek_to(self, frame_idx: int) -> None:
        if not self.video_cap:
            return
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self._next_frame()

    def _next_frame(self) -> None:
        if not self.video_cap or not self.video_cap.isOpened():
            return
        ret, frame = self.video_cap.read()
        if not ret:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.is_playing:
                self._next_frame()
            return
        pos = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.current_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not self._seek_dragging:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(max(0, pos))
            self.seek_slider.blockSignals(False)
        self.frame_lbl.setText(f"{max(0, pos) + 1} / {self._total_frames}")
        self._show_async(self.current_frame)

    # ── Async dither pipeline ───────────────────────────────────────────────────

    def _retire_worker(self, worker: DitherWorker) -> None:
        """Called via finished signal on abandoned workers. Safe async cleanup."""
        if worker is self._frame_worker:
            # only if it somehow wasn't replaced yet
            self._frame_worker = None
        worker.deleteLater()

    def _drain_frame_worker(self) -> None:
        """
        Hard-wait for in-flight worker. Use ONLY on close / open_file.
        Blocks main thread up to 1 s, then force-terminates.
        """
        w = self._frame_worker
        if w is None:
            return
        self._frame_id     = 0
        self._frame_worker = None
        if w.isRunning():
            w.stop()
            if not w.wait(1000):
                w.terminate()
                w.wait(300)
        w.deleteLater()

    def _show_async(self, img: Image.Image) -> None:
        """
        Non-blocking frame dispatch.
        If previous worker still running: signal stop, skip this frame.
        Worker self-cleans via _retire_worker on finish.
        Do NOT touch _frame_id on drop — only update when launching new worker.
        """
        w = self._frame_worker
        if w is not None and w.isRunning():
            # previous dither still in progress — signal it to stop,
            # let it clean up via finished → _retire_worker, skip this frame
            w.stop()
            w.finished.connect(lambda _pl, _w=w: self._retire_worker(_w))
            self._frame_worker = None
            return  # drop frame — timer will deliver the next one

        fid = next(_worker_id_counter)
        self._frame_id = fid
        p = self.get_params()
        new_w = DitherWorker(
            img,
            p["pixel_size"], p["threshold"], p["color"], p["method"],
            p["brightness"], p["contrast"], p["blur"], p["sharpen"],
            p["glow_radius"], p["glow_intensity"],
            preview=True,
            palette_name=p.get("palette_name", "B&W"),
            custom_palette=p.get("custom_palette"),
        )
        new_w.finished.connect(lambda pl, _fid=fid: self._on_frame_ready(pl, _fid))
        new_w.error.connect(lambda msg: self.status_message.emit(f"frame err: {msg}"))
        self._frame_worker = new_w
        new_w.start()

    def _on_frame_ready(self, payload, frame_id: int) -> None:
        if frame_id != self._frame_id:
            # stale result from an abandoned worker — clean it up
            if self._frame_worker is not None:
                pass  # newer worker owns the slot
            return
        self._frame_worker = None  # slot free for next frame
        img, _elapsed, _is_preview = payload
        self.canvas.set_image(pil_to_pixmap(img))
        self.canvas.setStyleSheet(f"background:{_P0};")

    # ── Export ──────────────────────────────────────────────────────────────────

    def export_video(self) -> None:
        if not _CV2:
            QMessageBox.warning(self, "Missing", "Install opencv-python.")
            return
        if not self.video_path:
            QMessageBox.warning(self, "No Video", "Load a video first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Video",
            str(Path(self.last_dir) / "dithered.mp4"),
            "MP4 (*.mp4)",
        )
        if not path:
            return
        if self.is_playing:
            self.toggle_play()
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        p = self.get_params()
        self.export_worker = VideoExportWorker(
            self.video_path, path,
            p["pixel_size"], p["threshold"], p["color"], p["method"],
            p["brightness"], p["contrast"], p["blur"], p["sharpen"],
            p["glow_radius"], p["glow_intensity"],
            palette_name=p.get("palette_name", "B&W"),
            custom_palette=p.get("custom_palette"),
        )
        self.export_worker.frame_ready.connect(
            lambda img: self.canvas.set_image(pil_to_pixmap(img))
        )
        self.export_worker.progress.connect(self._on_progress)
        self.export_worker.export_done.connect(self._on_export_done)
        self.export_worker.error.connect(
            lambda msg: QMessageBox.critical(self, "Export Error", msg)
        )
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_message.emit("exporting...")
        self.export_worker.start()

    def _on_progress(self, cur: int, total: int) -> None:
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(cur)
        self.status_message.emit(f"exporting {cur}/{total} frames")

    def _on_export_done(self) -> None:
        self.progress_bar.setVisible(False)
        self.status_message.emit("export complete")
        if self.export_worker is not None:
            self.export_worker.deleteLater()
            self.export_worker = None
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        QMessageBox.information(self, "Done", "Video exported.")

    # ── Zoom proxy ─────────────────────────────────────────────────────────────────

    def zoom_in(self)  -> None: self.canvas.zoom_in()
    def zoom_out(self) -> None: self.canvas.zoom_out()
    def fit(self)      -> None: self.canvas.fit()
    def actual(self)   -> None: self.canvas.actual()

    @property
    def zoom_level(self) -> float:
        return self.canvas.zoom_level

    # ── Cleanup ──────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._play_timer.stop()
        self._drain_frame_worker()  # hard-wait — safe to destroy after
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.stop()
            self.export_worker.wait(2000)
        super().closeEvent(event)
