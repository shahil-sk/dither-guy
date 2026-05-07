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
    QProgressBar, QSlider, QStyle,
)

from .constants import _MAX_PIXELS, _HISTORY_LIMIT, _DEBOUNCE_MS
from .dither_kernels import apply_dither
from .theme import _P0, _P1, _P2, _P3, _P4, _P5, _P6, _G0, _G1, _G2, _G3, _FG, _FG2, _FG3, _RE, _AE, _MONO_FONT
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

    # ── Build ──────────────────────────────────────────────────────────────

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

        # ── Toolbar bar ─────────────────────────────────────────────────────────────
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

    # ── Drag & drop ───────────────────────────────────────────────────────────

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).suffix.lower() in self._DROP_EXTS:
                self._load(p)
                break

    # ── Auto-update ──────────────────────────────────────────────────────────

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

    # ── Processing ──────────────────────────────────────────────────────────

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

    # ── File I/O ──────────────────────────────────────────────────────────────

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

    # ── History ──────────────────────────────────────────────────────────────

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

    # ── Image operations ─────────────────────────────────────────────────────────

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

    # ── Worker callbacks ────────────────────────────────────────────────────────

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

    # ── Zoom proxy ────────────────────────────────────────────────────────────

    def zoom_in(self)  -> None: self.canvas.zoom_in()
    def zoom_out(self) -> None: self.canvas.zoom_out()
    def fit(self)      -> None: self.canvas.fit()
    def actual(self)   -> None: self.canvas.actual()

    @property
    def zoom_level(self) -> float:
        return self.canvas.zoom_level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


# ---------------------------------------------------------------------------
# Video tab
# ---------------------------------------------------------------------------

class VideoTab(QWidget):
    status_message = Signal(str)

    _VIDEO_FILTER = "Video (*.mp4 *.avi *.mov *.mkv *.webm);;All (*.*)"

    def __init__(self, get_params):
        super().__init__()
        self.get_params    = get_params
        self.video_cap     = None
        self.video_path    = None
        self.current_frame = None
        self.is_playing    = False
        self.loop          = False
        self.export_worker = None
        self.last_dir      = str(Path.home())
        self._fps          = 25.0
        self._total_frames = 0
        self._scrubbing    = False     # True while user drags seek bar
        self._play_timer   = QTimer()
        self._play_timer.timeout.connect(self._next_frame)
        self._build()
        self.setFocusPolicy(Qt.StrongFocus)

    # ── Build ──────────────────────────────────────────────────────────────

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Info bar ────────────────────────────────────────────────────────────
        self.info_lbl = QLabel("no video loaded")
        self.info_lbl.setAlignment(Qt.AlignCenter)
        self.info_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:10px; padding:4px;"
            f"background:{_P0}; color:{_FG3};"
        )
        layout.addWidget(self.info_lbl)

        # ── Canvas ──────────────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.canvas = ZoomableLabel("▶ Load a video · MP4 / AVI / MOV / MKV")
        self.canvas.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:14px; color:{_P5}; background:{_P0};"
        )
        scroll.setWidget(self.canvas)
        layout.addWidget(scroll, stretch=1)

        # ── Export progress bar (thin, shown only during export) ─────────────
        self.export_bar = QProgressBar()
        self.export_bar.setVisible(False)
        self.export_bar.setFixedHeight(4)
        self.export_bar.setTextVisible(False)
        self.export_bar.setStyleSheet(
            f"QProgressBar {{ background:{_P4}; border:none; }}"
            f"QProgressBar::chunk {{ background:{_AE}; }}"
        )
        layout.addWidget(self.export_bar)

        # ── Seek / scrub bar ───────────────────────────────────────────────
        seek_container = QWidget()
        seek_container.setStyleSheet(f"background:{_P0};")
        seek_layout = QHBoxLayout(seek_container)
        seek_layout.setContentsMargins(8, 3, 8, 0)
        seek_layout.setSpacing(6)

        self._pos_lbl = QLabel("0:00")
        self._pos_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:11px; color:{_FG2};"
            "min-width:40px;"
        )
        self._pos_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.seek_bar = QSlider(Qt.Horizontal)
        self.seek_bar.setMinimum(0)
        self.seek_bar.setMaximum(1000)   # logical 0–1000 units
        self.seek_bar.setValue(0)
        self.seek_bar.setEnabled(False)
        self.seek_bar.setFixedHeight(18)
        self.seek_bar.setToolTip("Seek / scrub  (← / → keys for ±5 s)")
        self.seek_bar.sliderPressed.connect(self._on_seek_press)
        self.seek_bar.sliderMoved.connect(self._on_seek_move)
        self.seek_bar.sliderReleased.connect(self._on_seek_release)

        self._dur_lbl = QLabel("0:00")
        self._dur_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:11px; color:{_FG3};"
            "min-width:40px;"
        )
        self._dur_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        seek_layout.addWidget(self._pos_lbl)
        seek_layout.addWidget(self.seek_bar, stretch=1)
        seek_layout.addWidget(self._dur_lbl)
        layout.addWidget(seek_container)

        # ── Transport bar ────────────────────────────────────────────────────
        bar = QWidget()
        bar.setStyleSheet(
            f"background:{_P0}; border-top:1px solid {_P5};"
        )
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(8, 5, 8, 5)
        bl.setSpacing(4)

        def _tbtn(label: str, slot, tip: str = "", oid: str = "") -> QPushButton:
            b = QPushButton(label)
            b.setMinimumHeight(28)
            b.setFixedWidth(56)
            if oid:
                b.setObjectName(oid)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bl.addWidget(b)
            return b

        # |<  Play/Pause  Stop  >|  ||Loop   [frame badge]   [fps badge]
        self.rewind_btn = _tbtn("⏮",      self._rewind,      "Rewind to start  (Home)")
        self.play_btn   = _tbtn("▶",      self.toggle_play,  "Play / Pause  (Space)",  "accent")
        self.stop_btn   = _tbtn("■",      self.stop,         "Stop & rewind  (S)")
        self.step_f_btn = _tbtn("⏭",      self._end,         "Jump to end  (End)")

        bl.addWidget(vsep())

        self.loop_btn = QPushButton("⟳  Loop")
        self.loop_btn.setCheckable(True)
        self.loop_btn.setMinimumHeight(28)
        self.loop_btn.setToolTip("Toggle loop playback  (L)")
        self.loop_btn.setStyleSheet(
            f"QPushButton {{ background:{_P4}; border:1px solid {_P5}; border-radius:3px;"
            f"color:{_FG3}; font-size:11px; padding:5px 10px; }}"
            f"QPushButton:checked {{ background:{_G3}; border-color:{_G2}; color:{_G0}; }}"
            f"QPushButton:hover {{ border-color:{_P6}; color:{_FG}; }}"
        )
        self.loop_btn.toggled.connect(self._on_loop_toggled)
        bl.addWidget(self.loop_btn)

        bl.addStretch()

        # Frame counter badge
        self._frame_badge = QLabel("-- / --")
        self._frame_badge.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:11px; color:{_FG2};"
            f"background:{_P2}; border:1px solid {_P5}; border-radius:2px;"
            "padding:2px 8px; margin:3px;"
        )
        self._frame_badge.setToolTip("Current frame / total frames")
        bl.addWidget(self._frame_badge)

        # FPS badge
        self._fps_badge = QLabel("-- fps")
        self._fps_badge.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:11px; color:{_FG3};"
            f"background:{_P2}; border:1px solid {_P5}; border-radius:2px;"
            "padding:2px 8px; margin:3px;"
        )
        self._fps_badge.setToolTip("Source video frame rate")
        bl.addWidget(self._fps_badge)

        bl.addWidget(vsep())

        export_btn = QPushButton("↥ Export")
        export_btn.setObjectName("accent")
        export_btn.setMinimumHeight(28)
        export_btn.setToolTip("Export dithered video to MP4")
        export_btn.clicked.connect(self.export_video)
        bl.addWidget(export_btn)

        layout.addWidget(bar)

        if not _CV2:
            warn = QLabel("⚠ opencv-python not installed — video disabled")
            warn.setAlignment(Qt.AlignCenter)
            warn.setStyleSheet(
                f"color:{_RE}; font-family:{_MONO_FONT}; font-size:10px;"
                f"padding:4px; background:{_P1};"
            )
            layout.addWidget(warn)

        self._set_controls_enabled(False)

    # ── Enable / disable controls atomically ───────────────────────────────

    def _set_controls_enabled(self, enabled: bool) -> None:
        for w in (self.rewind_btn, self.play_btn, self.stop_btn,
                  self.step_f_btn, self.loop_btn, self.seek_bar):
            w.setEnabled(enabled)

    # ── File I/O ──────────────────────────────────────────────────────────────

    def open_file(self) -> None:
        if not _CV2:
            QMessageBox.warning(self, "Missing", "Install opencv-python.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", self.last_dir, self._VIDEO_FILTER
        )
        if not path:
            return
        self._load_video(path)

    def _load_video(self, path: str) -> None:
        if self.is_playing:
            self._play_timer.stop()
            self.is_playing = False
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
        self._fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.last_dir      = str(Path(path).parent)

        duration = self._total_frames / self._fps
        name     = Path(path).name
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.info_lbl.setText(
            f"{name}  ·  {w}×{h}  ·  {self._total_frames} fr  ·  {self._fps:.2f} fps  ·  {_fmt_time(duration)}"
        )
        self._fps_badge.setText(f"{self._fps:.2f} fps")
        self._dur_lbl.setText(_fmt_time(duration))
        self.seek_bar.setValue(0)
        self._set_controls_enabled(True)
        self.play_btn.setText("▶")

        self.status_message.emit(f"loaded {name}")
        # Show first frame
        self._seek_to_frame(0)

    # ── Playback ─────────────────────────────────────────────────────────────

    def toggle_play(self) -> None:
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        if not self.video_cap:
            return
        self.is_playing = True
        self.play_btn.setText("⏸")
        interval = max(1, int(1000 / self._fps))
        self._play_timer.start(interval)
        self.status_message.emit("playing")

    def _pause(self) -> None:
        self.is_playing = False
        self._play_timer.stop()
        self.play_btn.setText("▶")
        self.status_message.emit("paused")

    def stop(self) -> None:
        """Stop playback and rewind to frame 0."""
        self._pause()
        self._seek_to_frame(0)
        self.status_message.emit("stopped")

    def _rewind(self) -> None:
        was_playing = self.is_playing
        if was_playing:
            self._pause()
        self._seek_to_frame(0)
        if was_playing:
            self._play()

    def _end(self) -> None:
        was_playing = self.is_playing
        if was_playing:
            self._pause()
        self._seek_to_frame(max(0, self._total_frames - 1))

    def _on_loop_toggled(self, checked: bool) -> None:
        self.loop = checked

    # ── Seek ────────────────────────────────────────────────────────────────

    def _on_seek_press(self) -> None:
        self._scrubbing = True
        if self.is_playing:
            self._play_timer.stop()

    def _on_seek_move(self, value: int) -> None:
        """Live scrub: show frame while dragging without committing to full render."""
        frame_idx = int(value / 1000 * (self._total_frames - 1))
        self._seek_to_frame(frame_idx, update_bar=False)

    def _on_seek_release(self) -> None:
        self._scrubbing = False
        frame_idx = int(self.seek_bar.value() / 1000 * (self._total_frames - 1))
        self._seek_to_frame(frame_idx)
        if self.is_playing:
            self._play_timer.start(max(1, int(1000 / self._fps)))

    def _seek_to_frame(self, frame_idx: int, update_bar: bool = True) -> None:
        if not self.video_cap:
            return
        frame_idx = max(0, min(frame_idx, self._total_frames - 1))
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        if not ret:
            return
        self.current_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._show(self.current_frame)
        self._update_position_ui(frame_idx, update_bar)

    def _update_position_ui(self, frame_idx: int, update_bar: bool = True) -> None:
        """Sync time label, frame badge, seek bar without feedback loops."""
        elapsed = frame_idx / max(1, self._fps)
        self._pos_lbl.setText(_fmt_time(elapsed))
        self._frame_badge.setText(f"{frame_idx + 1} / {self._total_frames}")
        if update_bar and self._total_frames > 1:
            pos = int(frame_idx / (self._total_frames - 1) * 1000)
            self.seek_bar.blockSignals(True)
            self.seek_bar.setValue(pos)
            self.seek_bar.blockSignals(False)

    # ── Timer tick ───────────────────────────────────────────────────────────

    def _next_frame(self) -> None:
        if not self.video_cap or not self.video_cap.isOpened():
            return
        frame_idx = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.video_cap.read()
        if not ret:
            # End of video
            if self.loop:
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._next_frame()
            else:
                self._pause()
                self._seek_to_frame(0)
            return
        self.current_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._show(self.current_frame)
        self._update_position_ui(frame_idx)

    # ── Render frame ──────────────────────────────────────────────────────────

    def _show(self, img: Image.Image) -> None:
        p = self.get_params()
        try:
            dith = apply_dither(
                img, p["pixel_size"], p["threshold"], p["color"], p["method"],
                p["brightness"], p["contrast"], p["blur"], p["sharpen"],
                p["glow_radius"], p["glow_intensity"],
                palette_name=p.get("palette_name", "B&W"),
                custom_palette=p.get("custom_palette"),
            )
            self.canvas.set_image(pil_to_pixmap(dith))
            self.canvas.setStyleSheet(f"background:{_P0};")
        except Exception as exc:
            self.status_message.emit(f"frame error: {exc}")

    # ── Keyboard shortcuts ─────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        if not self.video_cap:
            return super().keyPressEvent(event)
        key = event.key()
        if key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_S:
            self.stop()
        elif key == Qt.Key_L:
            self.loop_btn.setChecked(not self.loop_btn.isChecked())
        elif key == Qt.Key_Home:
            self._rewind()
        elif key == Qt.Key_End:
            self._end()
        elif key in (Qt.Key_Right, Qt.Key_Period):
            # +5 seconds
            cur = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._seek_to_frame(cur + int(self._fps * 5))
        elif key in (Qt.Key_Left, Qt.Key_Comma):
            # -5 seconds
            cur = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._seek_to_frame(cur - int(self._fps * 5))
        elif key == Qt.Key_BracketRight:
            # +1 frame (step forward)
            cur = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._seek_to_frame(cur + 1)
        elif key == Qt.Key_BracketLeft:
            # -1 frame (step back)
            cur = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._seek_to_frame(max(0, cur - 2))  # -2 because read() advances by 1
        else:
            super().keyPressEvent(event)

    # ── Export ───────────────────────────────────────────────────────────────

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
            self._pause()
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
        self.export_worker.progress.connect(self._on_export_progress)
        self.export_worker.finished.connect(self._on_export_done)
        self.export_worker.error.connect(
            lambda msg: QMessageBox.critical(self, "Export Error", msg)
        )
        self.export_bar.setVisible(True)
        self.export_bar.setValue(0)
        self.status_message.emit("exporting...")
        self.export_worker.start()

    def _on_export_progress(self, cur: int, total: int) -> None:
        self.export_bar.setMaximum(total)
        self.export_bar.setValue(cur)
        self.status_message.emit(f"exporting {cur}/{total} frames")

    def _on_export_done(self) -> None:
        if self.export_worker is None:
            return
        self.export_worker = None
        self.export_bar.setVisible(False)
        self.status_message.emit("export complete")
        QMessageBox.information(self, "Done", "Video exported.")
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Zoom proxy ────────────────────────────────────────────────────────────

    def zoom_in(self)  -> None: self.canvas.zoom_in()
    def zoom_out(self) -> None: self.canvas.zoom_out()
    def fit(self)      -> None: self.canvas.fit()
    def actual(self)   -> None: self.canvas.actual()

    @property
    def zoom_level(self) -> float:
        return self.canvas.zoom_level

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._play_timer.stop()
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.stop()
            self.export_worker.wait(2000)
        super().closeEvent(event)
