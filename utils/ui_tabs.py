from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QCheckBox, QFileDialog, QMessageBox,
    QProgressBar,
)

from .constants import _MAX_PIXELS, _HISTORY_LIMIT, _DEBOUNCE_MS
from .dither_kernels import apply_dither
from .theme import _P0, _P1, _P4, _P5, _G3, _FG3, _MONO_FONT
from .workers import DitherWorker, VideoExportWorker
from .ui_widgets import ZoomableLabel, HistogramWidget, pil_to_pixmap, hsep, vsep

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ---------------------------------------------------------------------------
# Image tab
# ---------------------------------------------------------------------------

class ImageTab(QWidget):
    status_message = Signal(str)

    def __init__(self, get_params):
        super().__init__()
        self.get_params   = get_params
        self.original_img: Optional[Image.Image] = None
        self.dithered_img: Optional[Image.Image] = None
        self.last_dir     = str(Path.home())
        self.worker: Optional[DitherWorker] = None
        self.auto_update  = True
        self._history: list[Image.Image] = []
        self._timer = QTimer(); self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.process)
        self._preview_timer = QTimer(); self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._process_preview)
        self._build()
        self.setAcceptDrops(True)

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)

        self.info_lbl = QLabel("Drop an image here  ·  or  Ctrl+O")
        self.info_lbl.setAlignment(Qt.AlignCenter)
        self.info_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:10px; padding:4px 8px;"
            f"background:{_P0}; color:{_FG3};")
        layout.addWidget(self.info_lbl)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        self.canvas = ZoomableLabel("▣  Drop image here  ·  Ctrl+O")
        self.canvas.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:14px; color:{_P5}; background:{_P0};")
        scroll.setWidget(self.canvas); layout.addWidget(scroll, stretch=1)

        self.histogram = HistogramWidget(); self.histogram.setVisible(False)
        layout.addWidget(self.histogram)

        # Primary bar
        bar1 = QWidget()
        bar1.setStyleSheet(f"background:{_P0};")
        bl1 = QHBoxLayout(bar1); bl1.setContentsMargins(8, 5, 8, 5); bl1.setSpacing(5)

        def mbt(label, slot, accent=False, tip=""):
            b = QPushButton(label)
            if accent: b.setObjectName("accent")
            b.clicked.connect(slot); b.setMinimumHeight(28); b.setToolTip(tip)
            bl1.addWidget(b); return b

        # mbt("▶ Open",   self.open_file, accent=True, tip="Ctrl+O")
        # mbt("▼ Save",   self.save_file, tip="Ctrl+S")
        mbt("◑ Invert", self.invert)
        mbt("↺ L", self.rotate_left)
        mbt("↻ R", self.rotate_right),
        mbt("↔ H", self.flip_h)
        mbt("↕ V", self.flip_v),
        mbt("✂ Crop", self.crop)
        
        bl1.addWidget(vsep())
        self.hist_cb = QCheckBox("Histogram")
        self.hist_cb.stateChanged.connect(lambda s: self.histogram.setVisible(bool(s)))
        bl1.addWidget(self.hist_cb)
        bl1.addWidget(vsep())
        self.auto_cb = QCheckBox("Live"); self.auto_cb.setChecked(True)
        self.auto_cb.setToolTip("Auto-update on parameter change")
        self.auto_cb.stateChanged.connect(self._toggle_auto)
        bl1.addWidget(self.auto_cb)
        self.apply_btn = QPushButton("▶ Apply"); self.apply_btn.setObjectName("accent")
        self.apply_btn.clicked.connect(self.process)
        self.apply_btn.setVisible(False); self.apply_btn.setMinimumHeight(28)
        bl1.addWidget(self.apply_btn); bl1.addStretch()
        self.undo_btn = QPushButton("↩ Undo")
        self.undo_btn.clicked.connect(self.undo); self.undo_btn.setEnabled(False)
        self.undo_btn.setMinimumHeight(22)
        self.undo_btn.setStyleSheet(
            f"font-size:10px; font-family:{_MONO_FONT}; padding:2px 7px;")
        bl1.addWidget(self.undo_btn)
        layout.addWidget(bar1)

        # # Transform bar
        # bar2 = QWidget()
        # bar2.setStyleSheet(f"background:{_P1}; border-top:1px solid {_P4};")
        # bl2 = QHBoxLayout(bar2); bl2.setContentsMargins(8, 3, 8, 3); bl2.setSpacing(3)
        # for label, slot in [
        #     ("↺ L", self.rotate_left), ("↻ R", self.rotate_right),
        #     ("↔ H", self.flip_h),      ("↕ V", self.flip_v),
        #     ("✂ Crop", self.crop),
        # ]:
        #     b = QPushButton(label); b.clicked.connect(slot)
        #     b.setMinimumHeight(22)
        #     b.setStyleSheet(f"font-size:10px; font-family:{_MONO_FONT}; padding:2px 7px;")
        #     bl2.addWidget(b)
        # bl2.addStretch()

        # layout.addWidget(bar2)

    # Drag & drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if Path(p).suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}:
                self._load(p); break

    def _toggle_auto(self, state):
        self.auto_update = bool(state)
        self.apply_btn.setVisible(not self.auto_update)

    def schedule(self, preview=False):
        if not self.auto_update: return
        if preview:
            self._preview_timer.stop(); self._preview_timer.start(80)
        else:
            self._timer.stop(); self._timer.start(_DEBOUNCE_MS)

    def _process_preview(self):
        if self.original_img is None: return
        self._stop_worker()
        p = self.get_params()
        self.worker = DitherWorker(
            self.original_img, p['pixel_size'], p['threshold'], p['color'], p['method'],
            p['brightness'], p['contrast'], p['blur'], p['sharpen'],
            p['glow_radius'], p['glow_intensity'], preview=True,
            palette_name=p.get('palette_name', 'B&W'),
            custom_palette=p.get('custom_palette'))
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(lambda _: None)
        self.worker.start()

    def _load(self, path: str):
        try:
            img = Image.open(path)
            if img.width * img.height > _MAX_PIXELS:
                ans = QMessageBox.question(self, "Large Image",
                    f"Image is {img.width}×{img.height}. Continue?",
                    QMessageBox.Yes | QMessageBox.No)
                if ans != QMessageBox.Yes: return
            self.original_img = img.convert("RGB")
            self._history.clear(); self.undo_btn.setEnabled(False)
            self.last_dir = str(Path(path).parent)
            self._refresh_info()
            self.status_message.emit(f"loaded  {Path(path).name}")
            self.process()
        except Exception as exc:
            QMessageBox.critical(self, "Open Error", f"Failed:\n{exc}")

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", self.last_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp);;All (*.*)")
        if path: self._load(path)

    def save_file(self):
        if self.dithered_img is None:
            QMessageBox.warning(self, "Nothing to save", "Process an image first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", str(Path(self.last_dir) / "dithered.png"),
            "PNG (*.png);;JPEG (*.jpg);;All (*.*)")
        if path:
            try:
                self.dithered_img.save(path)
                self.last_dir = str(Path(path).parent)
                self.status_message.emit(f"saved  {Path(path).name}")
            except Exception as exc:
                QMessageBox.critical(self, "Save Error", f"Failed:\n{exc}")

    def _refresh_info(self):
        if self.original_img:
            w, h = self.original_img.size
            self.info_lbl.setText(f"{w}×{h} px  ·  history: {len(self._history)}")

    def _push_history(self):
        if self.original_img is None: return
        self._history.append(self.original_img.copy())
        if len(self._history) > _HISTORY_LIMIT: self._history.pop(0)
        self.undo_btn.setEnabled(True)

    def _require_image(self, op="do this"):
        if self.original_img is None:
            QMessageBox.warning(self, "No Image", f"Load an image to {op}."); return False
        return True

    def invert(self):
        if not self._require_image("invert"): return
        self._push_history()
        self.original_img = Image.fromarray((255 - np.array(self.original_img)).astype(np.uint8))
        self.status_message.emit("inverted"); self._refresh_info(); self.process()

    def rotate_left(self):
        if not self._require_image("rotate"): return
        self._push_history(); self.original_img = self.original_img.rotate(90, expand=True)
        self.status_message.emit("rotated 90° CCW"); self._refresh_info(); self.process()

    def rotate_right(self):
        if not self._require_image("rotate"): return
        self._push_history(); self.original_img = self.original_img.rotate(-90, expand=True)
        self.status_message.emit("rotated 90° CW"); self._refresh_info(); self.process()

    def flip_h(self):
        if not self._require_image("flip"): return
        self._push_history()
        self.original_img = self.original_img.transpose(Image.FLIP_LEFT_RIGHT)
        self.status_message.emit("flipped H"); self._refresh_info(); self.process()

    def flip_v(self):
        if not self._require_image("flip"): return
        self._push_history()
        self.original_img = self.original_img.transpose(Image.FLIP_TOP_BOTTOM)
        self.status_message.emit("flipped V"); self._refresh_info(); self.process()

    def crop(self):
        if not self._require_image("crop"): return
        from .ui_dialogs import CropDialog
        dlg = CropDialog(self.original_img.width, self.original_img.height, self)
        from PySide6.QtWidgets import QDialog
        if dlg.exec() != QDialog.Accepted: return
        v = dlg.values(); l, t, r, b = v["left"], v["top"], v["right"], v["bottom"]
        x2 = self.original_img.width - r; y2 = self.original_img.height - b
        if l >= x2 or t >= y2:
            QMessageBox.warning(self, "Crop", "Nothing left after crop."); return
        self._push_history(); self.original_img = self.original_img.crop((l, t, x2, y2))
        self.status_message.emit(
            f"cropped → {self.original_img.width}×{self.original_img.height}")
        self._refresh_info(); self.process()

    def undo(self):
        if not self._history: return
        self.original_img = self._history.pop()
        self.undo_btn.setEnabled(bool(self._history))
        self.status_message.emit("undo"); self._refresh_info(); self.process()

    def _stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop(); self.worker.quit()
            if not self.worker.wait(1500):
                self.worker.terminate(); self.worker.wait(400)
            self.worker.deleteLater(); self.worker = None

    def process(self):
        if self.original_img is None: return
        self._stop_worker()
        self.status_message.emit("processing...")
        p = self.get_params()
        self.worker = DitherWorker(
            self.original_img, p['pixel_size'], p['threshold'], p['color'], p['method'],
            p['brightness'], p['contrast'], p['blur'], p['sharpen'],
            p['glow_radius'], p['glow_intensity'], preview=False,
            palette_name=p.get('palette_name', 'B&W'),
            custom_palette=p.get('custom_palette'))
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_done(self, payload):
        img, elapsed, is_preview = payload
        if not is_preview:
            self.dithered_img = img
            if self.hist_cb.isChecked(): self.histogram.update_data(img)
        self.canvas.set_image(pil_to_pixmap(img))
        self.canvas.setStyleSheet(f"background:{_P0};")
        p = self.get_params()
        ms_str = f"{elapsed*1000:.0f}ms"
        tag = "[preview] " if is_preview else ""
        self.status_message.emit(
            f"{tag}{p['method']}  ·  {img.width}×{img.height}  ·  {ms_str}")
        if self.original_img and not is_preview:
            ow, oh = self.original_img.size
            self.info_lbl.setText(
                f"{ow}×{oh}  ──▶  {img.width}×{img.height} px  ·  {ms_str}")

    def _on_error(self, msg: str):
        self.status_message.emit(f"error: {msg}")
        QMessageBox.warning(self, "Processing Error", msg)

    def zoom_in(self):    self.canvas.zoom_in()
    def zoom_out(self):   self.canvas.zoom_out()
    def fit(self):        self.canvas.fit()
    def actual(self):     self.canvas.actual()
    def zoom_level(self): return self.canvas.zoom_level


# ---------------------------------------------------------------------------
# Video tab
# ---------------------------------------------------------------------------

class VideoTab(QWidget):
    status_message = Signal(str)

    def __init__(self, get_params):
        super().__init__()
        self.get_params    = get_params
        self.video_cap     = None; self.video_path = None
        self.current_frame = None; self.is_playing = False
        self.export_worker = None; self.last_dir = str(Path.home())
        self._play_timer   = QTimer(); self._play_timer.timeout.connect(self._next_frame)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)

        self.info_lbl = QLabel("no video loaded"); self.info_lbl.setAlignment(Qt.AlignCenter)
        self.info_lbl.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:10px; padding:4px;"
            f"background:{_P0}; color:{_FG3};")
        layout.addWidget(self.info_lbl)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        self.canvas = ZoomableLabel("▶  Load a video  ·  MP4 / AVI / MOV")
        self.canvas.setStyleSheet(
            f"font-family:{_MONO_FONT}; font-size:14px; color:{_P5}; background:{_P0};")
        scroll.setWidget(self.canvas); layout.addWidget(scroll, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False); self.progress_bar.setFixedHeight(3)
        self.progress_bar.setTextVisible(False); layout.addWidget(self.progress_bar)

        bar = QWidget()
        bar.setStyleSheet(f"background:{_P0}; border-top:1px solid {_G3};")
        bl = QHBoxLayout(bar); bl.setContentsMargins(8, 5, 8, 5); bl.setSpacing(5)
        # btn_open = QPushButton("▶ Open Video"); btn_open.setObjectName("accent")
        # btn_open.clicked.connect(self.open_file); btn_open.setMinimumHeight(28)
        # bl.addWidget(btn_open)
        self.play_btn = QPushButton("▶ Play"); self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False); self.play_btn.setMinimumHeight(28); bl.addWidget(self.play_btn)
        # btn_exp = QPushButton("▼ Export"); btn_exp.clicked.connect(self.export_video)
        # btn_exp.setMinimumHeight(28); bl.addWidget(btn_exp)
        bl.addStretch(); layout.addWidget(bar)

        if not _CV2:
            from .theme import _RE
            warn = QLabel("⚠  opencv-python not installed — video disabled")
            warn.setAlignment(Qt.AlignCenter)
            warn.setStyleSheet(f"color:{_RE}; font-family:{_MONO_FONT}; font-size:10px;"
                               f"padding:4px; background:{_P1};")
            layout.addWidget(warn)

    def open_file(self):
        if not _CV2: QMessageBox.warning(self, "Missing", "Install opencv-python."); return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", self.last_dir, "Video (*.mp4 *.avi *.mov);;All (*.*)")
        if not path: return
        if self.video_cap: self.video_cap.release(); self.video_cap = None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release(); QMessageBox.critical(self, "Error", "Cannot open video."); return
        self.video_cap = cap; self.video_path = path
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last_dir = str(Path(path).parent)
        self.info_lbl.setText(f"{Path(path).name}  ·  {total} frames @ {fps:.1f} fps")
        self.play_btn.setEnabled(True)
        self.status_message.emit(f"loaded  {Path(path).name}")
        self._next_frame()

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False; self._play_timer.stop(); self.play_btn.setText("▶ Play")
        else:
            if not self.video_cap: return
            fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 25.
            self.is_playing = True; self.play_btn.setText("⏸ Pause")
            self._play_timer.start(max(1, int(1000 / fps)))

    def _next_frame(self):
        if not self.video_cap or not self.video_cap.isOpened(): return
        ret, frame = self.video_cap.read()
        if not ret:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if self.is_playing: self._next_frame(); return
        self.current_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._show(self.current_frame)

    def _show(self, img):
        p = self.get_params()
        try:
            dith = apply_dither(
                img, p['pixel_size'], p['threshold'], p['color'], p['method'],
                p['brightness'], p['contrast'], p['blur'], p['sharpen'],
                p['glow_radius'], p['glow_intensity'])
            self.canvas.set_image(pil_to_pixmap(dith))
            self.canvas.setStyleSheet(f"background:{_P0};")
        except Exception as exc:
            self.status_message.emit(f"frame error: {exc}")

    def export_video(self):
        if not _CV2: QMessageBox.warning(self, "Missing", "Install opencv-python."); return
        if not self.video_path: QMessageBox.warning(self, "No Video", "Load a video first."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", str(Path(self.last_dir) / "dithered.mp4"), "MP4 (*.mp4)")
        if not path: return
        if self.is_playing: self.toggle_play()
        if self.video_cap: self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        p = self.get_params()
        self.export_worker = VideoExportWorker(
            self.video_path, path, p['pixel_size'], p['threshold'], p['color'], p['method'],
            p['brightness'], p['contrast'], p['blur'], p['sharpen'],
            p['glow_radius'], p['glow_intensity'])
        self.export_worker.frame_ready.connect(
            lambda img: self.canvas.set_image(pil_to_pixmap(img)))
        self.export_worker.progress.connect(self._on_progress)
        self.export_worker.finished.connect(self._on_export_done)
        self.export_worker.error.connect(
            lambda msg: QMessageBox.critical(self, "Export Error", msg))
        self.progress_bar.setVisible(True); self.progress_bar.setValue(0)
        self.status_message.emit("exporting..."); self.export_worker.start()

    def _on_progress(self, cur, total):
        self.progress_bar.setMaximum(total); self.progress_bar.setValue(cur)
        self.status_message.emit(f"exporting {cur}/{total} frames")

    def _on_export_done(self):
        self.progress_bar.setVisible(False)
        self.status_message.emit("export complete")
        QMessageBox.information(self, "Done", "Video exported.")
        if self.video_cap: self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def zoom_in(self):    self.canvas.zoom_in()
    def zoom_out(self):   self.canvas.zoom_out()
    def fit(self):        self.canvas.fit()
    def actual(self):     self.canvas.actual()
    def zoom_level(self): return self.canvas.zoom_level

    def closeEvent(self, event):
        self._play_timer.stop()
        if self.video_cap: self.video_cap.release(); self.video_cap = None
        if self.export_worker and self.export_worker.isRunning():
            self.export_worker.stop(); self.export_worker.wait(2000)
        super().closeEvent(event)
