# dither_guy/batch.py
# Batch-processing engine.
#
# Applies a fixed set of dither parameters to every supported image inside an
# input folder and writes PNG results to an output folder.  Processing is
# parallelised via a ThreadPoolExecutor so multi-core machines keep all cores
# busy during large runs.
#
# Public API
# ──────────
#   batch_process(input_folder, output_folder, params,
#                 progress_cb=None, cancel_flag=None)  →  (success_count, error_count)

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from utils.constants import _VIDEO_WORKERS
from utils.dither_kernels import apply_dither

# Supported input extensions
_BATCH_IMAGE_EXTS: set[str] = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp",
}


def batch_process(
    input_folder: str,
    output_folder: str,
    params: dict,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    cancel_flag:  Optional[list[bool]] = None,
) -> tuple[int, int]:
    """Apply dither to every image in *input_folder*, saving PNGs to *output_folder*.

    Parameters
    ──────────
    input_folder  : path to the folder containing source images
    output_folder : destination folder (created if it does not exist)
    params        : parameter dict as returned by ControlPanel.get_params()
    progress_cb   : optional callable(done: int, total: int, name: str)
                    called after each file completes
    cancel_flag   : a single-element list ``[False]``; set ``[True]`` to
                    request cancellation between frames

    Returns
    ───────
    (success_count, error_count) tuple
    """
    in_dir  = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    files   = [f for f in in_dir.iterdir() if f.suffix.lower() in _BATCH_IMAGE_EXTS]
    total   = len(files)
    success = 0
    errors  = 0

    def _process_one(fp: Path) -> tuple[bool, str]:
        try:
            img    = Image.open(fp).convert("RGB")
            result = apply_dither(
                img,
                params.get("pixel_size", 4),
                params.get("threshold", 128),
                tuple(params.get("color", (0, 255, 65))),
                params.get("method", "Floyd-Steinberg"),
                params.get("brightness", 1.0),
                params.get("contrast", 1.0),
                params.get("blur", 0),
                params.get("sharpen", 0),
                params.get("glow_radius", 0),
                params.get("glow_intensity", 0),
                preview=False,
                palette_name=params.get("palette_name", "B&W"),
            )
            out_path = out_dir / (fp.stem + ".png")
            result.save(out_path)
            return True, fp.name
        except Exception as exc:
            return False, f"{fp.name}: {exc}"

    workers = min(_VIDEO_WORKERS, max(1, total))
    done    = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process_one, fp): fp for fp in files}
        for fut in as_completed(futures):
            if cancel_flag and cancel_flag[0]:
                ex.shutdown(wait=False, cancel_futures=True)
                break
            ok, msg = fut.result()
            done += 1
            if ok:
                success += 1
            else:
                errors += 1
            if progress_cb:
                progress_cb(done, total, msg)

    return success, errors
