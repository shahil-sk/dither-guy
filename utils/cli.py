# dither_guy/cli.py
# Command-line interface for headless (no-GUI) dithering.
#
# Supports three modes:
#   1. Single image   — dither one file and write the result
#   2. Batch mode     — dither every image in a folder (--batch)
#   3. Info flags     — list available methods or palettes (--list-*)
#
# Invoked automatically by __main__.py when sys.argv contains positional
# arguments or known CLI flags.  Can also be called programmatically via
# ``from dither_guy.cli import cli; cli()``.

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from utils.constants      import _VIDEO_WORKERS
from utils.dither_kernels import apply_dither
from utils.palettes       import METHOD_GROUPS, PALETTES
from utils.batch          import batch_process, _BATCH_IMAGE_EXTS
from utils.constants      import VERSION


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dither_guy",
        description=f"Dither Guy v{VERSION} — headless dithering from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m dither_guy photo.jpg output.png
  python -m dither_guy photo.jpg output.png --method atkinson --pixel-size 3
  python -m dither_guy photo.jpg output.png --palette gameboy
  python -m dither_guy --batch input_dir/ output_dir/ --method bayer4
  python -m dither_guy --list-methods
  python -m dither_guy --list-palettes
""",
    )
    p.add_argument("input",  nargs="?", help="Input image file")
    p.add_argument("output", nargs="?", help="Output image file")
    p.add_argument("--method",          default="Floyd-Steinberg",
                   help="Dither algorithm (default: Floyd-Steinberg)")
    p.add_argument("--pixel-size",      type=int,   default=1,
                   help="Pixel block size (default: 1)")
    p.add_argument("--threshold",       type=int,   default=128,
                   help="Threshold 0-255 (default: 128)")
    p.add_argument("--brightness",      type=float, default=1.0)
    p.add_argument("--contrast",        type=float, default=1.0)
    p.add_argument("--blur",            type=int,   default=0)
    p.add_argument("--sharpen",         type=int,   default=0)
    p.add_argument("--color",           default="00ff41",
                   help="Foreground hex colour, e.g. ff0000 (default: 00ff41)")
    p.add_argument("--palette",         default="B&W",
                   help="Palette name (default: B&W). Use --list-palettes.")
    p.add_argument("--glow-radius",     type=int, default=0)
    p.add_argument("--glow-intensity",  type=int, default=0)
    p.add_argument("--batch",           nargs=2, metavar=("INPUT_DIR","OUTPUT_DIR"),
                   help="Batch-process a folder of images")
    p.add_argument("--workers",         type=int, default=_VIDEO_WORKERS,
                   help=f"Parallel workers for batch (default: {_VIDEO_WORKERS})")
    p.add_argument("--list-methods",    action="store_true")
    p.add_argument("--list-palettes",   action="store_true")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_color(hex_str: str) -> tuple:
    try:
        hx = hex_str.lstrip("#")
        return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
    except Exception:
        return (0, 255, 65)


def _progress_bar(done: int, total: int, name: str):
    pct = done / total
    bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
    print(f"\r  [{bar}] {done}/{total}  {name[:40]}", end="", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli():
    """Parse arguments and run the appropriate headless operation."""
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Info flags ────────────────────────────────────────────────────────────
    if args.list_methods:
        print("Available dither methods:")
        for group, members in METHOD_GROUPS.items():
            print(f"\n  [{group}]")
            for m in members:
                print(f"    {m}")
        return

    if args.list_palettes:
        print("Available palettes:")
        for name, colors in PALETTES.items():
            print(f"  {name:24s}  ({len(colors)} colors)")
        return

    color  = _parse_color(args.color)
    params = {
        "method":         args.method,
        "pixel_size":     args.pixel_size,
        "threshold":      args.threshold,
        "brightness":     args.brightness,
        "contrast":       args.contrast,
        "blur":           args.blur,
        "sharpen":        args.sharpen,
        "color":          color,
        "palette_name":   args.palette,
        "glow_radius":    args.glow_radius,
        "glow_intensity": args.glow_intensity,
    }

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.batch:
        in_dir, out_dir = args.batch
        print(f"Batch: {in_dir}  →  {out_dir}")
        print(f"Method: {args.method}  |  Palette: {args.palette}")
        files = [f for f in Path(in_dir).iterdir()
                 if f.suffix.lower() in _BATCH_IMAGE_EXTS]
        print(f"Found {len(files)} image(s)")
        ok, err = batch_process(in_dir, out_dir, params, progress_cb=_progress_bar)
        print(f"\nDone: {ok} ok, {err} errors")
        return

    # ── Single image mode ─────────────────────────────────────────────────────
    if not args.input or not args.output:
        parser.print_help()
        return

    print(f"  input  : {args.input}")
    print(f"  output : {args.output}")
    print(f"  method : {args.method}  |  palette: {args.palette}")

    t0     = time.perf_counter()
    img    = Image.open(args.input)
    result = apply_dither(
        img,
        params["pixel_size"],
        params["threshold"],
        params["color"],
        params["method"],
        params["brightness"],
        params["contrast"],
        params["blur"],
        params["sharpen"],
        params["glow_radius"],
        params["glow_intensity"],
        palette_name=params["palette_name"],
    )
    result.save(args.output)
    elapsed = time.perf_counter() - t0
    print(f"  done   : {result.width}×{result.height}  in  {elapsed*1000:.0f}ms")
