from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .batch import batch_process, _BATCH_IMAGE_EXTS
from .constants import _VIDEO_WORKERS


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------

async def ask_crop_values(
    window: toga.Window,
    img_w: int,
    img_h: int,
) -> Optional[dict]:
    """
    Ask the user for crop margins via simple text dialogs.
    Returns dict with keys left/top/right/bottom, or None if cancelled.
    """
    margins = {}
    for side, limit in [("left", img_w - 1), ("top", img_h - 1),
                         ("right", img_w - 1), ("bottom", img_h - 1)]:
        raw = await window.text_dialog(
            "Crop Image",
            f"Pixels to remove from {side} (0 - {limit}):",
        )
        if raw is None:
            return None
        try:
            margins[side] = max(0, min(int(raw), limit))
        except ValueError:
            margins[side] = 0
    return margins


# ---------------------------------------------------------------------------
# Batch dialog
# ---------------------------------------------------------------------------

async def show_batch_dialog(window: toga.Window, get_params):
    """
    Runs a minimal batch-process flow using sequential text dialogs.
    """
    in_dir = await window.text_dialog(
        "Batch Process",
        "Enter input folder path:",
    )
    if not in_dir or not Path(in_dir).is_dir():
        await window.info_dialog("Batch", "Input folder not found or not set.")
        return

    out_dir = await window.text_dialog(
        "Batch Process",
        "Enter output folder path:",
    )
    if not out_dir:
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    files = [
        f for f in Path(in_dir).iterdir()
        if f.suffix.lower() in _BATCH_IMAGE_EXTS
    ]
    if not files:
        await window.info_dialog("Batch", "No supported images found in input folder.")
        return

    params = get_params()
    cancel_flag = [False]

    def progress(done, total, name):
        pass  # No inline progress UI in minimal Toga batch flow

    ok, err = batch_process(
        in_dir, out_dir, params,
        progress_cb=progress,
        cancel_flag=cancel_flag,
    )
    await window.info_dialog(
        "Batch Complete",
        f"{ok} saved  {err} error(s)  ->  {out_dir}",
    )
