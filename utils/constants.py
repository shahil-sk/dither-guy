# dither_guy/constants.py
# Global constants shared across every module.  Centralising them here means
# you only need to change one file when tuning performance knobs, path
# defaults, or version strings.

from __future__ import annotations

import os
from pathlib import Path

# ── App metadata ─────────────────────────────────────────────────────────────
VERSION = "5.0"

# ── Processing limits ─────────────────────────────────────────────────────────
_MAX_PIXELS    = 8_000_000   # warn the user when image exceeds this many pixels
_HISTORY_LIMIT = 20          # max undo steps kept in memory per image session
_DEBOUNCE_MS   = 250         # milliseconds to wait before triggering a full re-render
_PREVIEW_SCALE = 0.5         # fractional scale used for live-drag previews
_VIDEO_WORKERS = min(8, (os.cpu_count() or 4))  # thread-pool size for video export

# ── Persistence ───────────────────────────────────────────────────────────────
PRESETS_DIR = Path("presets")   # directory where JSON preset files are stored
