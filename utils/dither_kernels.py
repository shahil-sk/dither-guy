from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from .matrices import tile, ORDERED_MATRICES, _DOT_CLASS, _BAYER_4x4
from .palettes import PALETTES
from .gpu_kernels import (
    GPU_BACKEND,
    gpu_ordered_dither,
    gpu_fs_dither,
    gpu_palette_nearest,
    from_gpu,
    to_gpu,
)

# ---------------------------------------------------------------------------
# Optional JIT via numba
# ---------------------------------------------------------------------------

try:
    from numba import njit, prange
    _NUMBA = True
except ImportError:
    _NUMBA = False

    def njit(*args, **kwargs):
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range

# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

def adjust(img: Image.Image, brightness: float, contrast: float,
           blur: float, sharpen: float) -> Image.Image:
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    for _ in range(int(sharpen)):
        img = img.filter(ImageFilter.SHARPEN)
    return img


def apply_glow(img: Image.Image, radius: float, intensity: float) -> Image.Image:
    if radius <= 0 or intensity <= 0:
        return img
    blurred  = img.filter(ImageFilter.GaussianBlur(radius=radius))
    base     = np.asarray(img,     dtype=np.float32)
    glow_lyr = np.asarray(blurred, dtype=np.float32) * (intensity / 100.0)
    # Screen blend: avoids creating a temporary (255 - ...) array twice
    out = np.float32(255.0) - (np.float32(255.0) - base) * (np.float32(255.0) - glow_lyr) * np.float32(1.0 / 255.0)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

# Precomputed sRGB → XYZ matrix (shared with gpu_kernels)
_SRGB_TO_XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
], dtype=np.float32)
_XYZ_SCALE = np.array([1.0 / 0.9505, 1.0, 1.0 / 1.089], dtype=np.float32)
_LAB_EPS   = np.float32(0.008856)
_LAB_KAPPA = np.float32(903.3)


def _rgb_to_lab_batch(rgb: np.ndarray) -> np.ndarray:
    """Vectorised sRGB -> CIE-L*a*b* for array (...,3) float32 [0-255] on CPU.

    Uses a fused matrix multiply for the linear-RGB → XYZ step.
    """
    r    = rgb.astype(np.float32) / np.float32(255.0)
    mask = r > np.float32(0.04045)
    r    = np.where(mask,
                    ((r + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4),
                    r / np.float32(12.92))
    # (..., 3) @ (3, 3)^T — one BLAS call instead of 3 dot products
    orig_shape = r.shape
    flat = r.reshape(-1, 3) if r.ndim > 2 else r
    xyz  = (flat @ _SRGB_TO_XYZ.T) * _XYZ_SCALE
    xyz  = xyz.reshape(orig_shape)
    f    = np.where(xyz > _LAB_EPS,
                    np.cbrt(xyz),
                    (_LAB_KAPPA * xyz + np.float32(16.0)) / np.float32(116.0))
    L = np.float32(116.0) * f[..., 1] - np.float32(16.0)
    a = np.float32(500.0) * (f[..., 0] - f[..., 1])
    b = np.float32(200.0) * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


_rgb_to_lab = _rgb_to_lab_batch

# Palette LAB cache — keyed by content hash so the same palette colours
# always hit the cache regardless of which array object holds them.
_PAL_LAB_CACHE: dict[int, np.ndarray] = {}


def _get_pal_lab(pal: np.ndarray) -> np.ndarray:
    key = hash(pal.tobytes())
    if key not in _PAL_LAB_CACHE:
        _PAL_LAB_CACHE[key] = _rgb_to_lab_batch(pal)
    return _PAL_LAB_CACHE[key]


USE_GPU_THRESHOLD = 1_000_000  # pixels


def _should_use_gpu(arr: np.ndarray) -> bool:
    return (
        GPU_BACKEND == "cuda" and
        arr.size >= USE_GPU_THRESHOLD
    )


def _nearest_palette_indices(pixels: np.ndarray, pal: np.ndarray,
                              pal_lab: np.ndarray) -> np.ndarray:
    """Nearest palette index per pixel — GPU only for CUDA + large images, else CPU einsum."""
    if _should_use_gpu(pixels):
        return gpu_palette_nearest(pixels, pal_lab)
    pix_lab = _rgb_to_lab_batch(pixels)
    diff    = pix_lab[:, np.newaxis, :] - pal_lab[np.newaxis, :, :]
    dists   = np.einsum('nkc,nkc->nk', diff, diff)
    return np.argmin(dists, axis=1)


# ---------------------------------------------------------------------------
# Error-diffusion coefficient tables (dy, dx, weight) — weights pre-divided
# ---------------------------------------------------------------------------

_COEFF_TABLES: dict[str, list[tuple]] = {
    "Floyd-Steinberg":     [(0, 1, 7/16), (1,-1, 3/16), (1, 0, 5/16), (1, 1, 1/16)],
    "Atkinson":            [(0, 1, 1/8),  (0, 2, 1/8),  (1,-1, 1/8),  (1, 0, 1/8),
                            (1, 1, 1/8),  (2, 0, 1/8)],
    "Sierra":              [(0, 1, 5/32), (0, 2, 3/32), (1,-2, 2/32), (1,-1, 4/32),
                            (1, 0, 5/32), (1, 1, 4/32), (1, 2, 2/32), (2,-1, 2/32),
                            (2, 0, 3/32), (2, 1, 2/32)],
    "Burkes":              [(0, 1, 8/32), (0, 2, 4/32), (1,-2, 2/32), (1,-1, 4/32),
                            (1, 0, 8/32), (1, 1, 4/32), (1, 2, 2/32)],
    "Stucki":              [(0, 1, 8/42), (0, 2, 4/42), (1,-2, 2/42), (1,-1, 4/42),
                            (1, 0, 8/42), (1, 1, 4/42), (1, 2, 2/42), (2,-2, 1/42),
                            (2,-1, 2/42), (2, 0, 4/42), (2, 1, 2/42), (2, 2, 1/42)],
    "Jarvis-Judice-Ninke": [(0, 1, 7/48), (0, 2, 5/48), (1,-2, 3/48), (1,-1, 5/48),
                            (1, 0, 7/48), (1, 1, 5/48), (1, 2, 3/48), (2,-2, 1/48),
                            (2,-1, 3/48), (2, 0, 5/48), (2, 1, 3/48), (2, 2, 1/48)],
    "Sierra-Lite":         [(0, 1, 2/4),  (1,-1, 1/4),  (1, 0, 1/4)],
    "Nakano":              [(0, 1, 8/24), (1,-1, 4/24),  (1, 0, 4/24),  (1, 1, 4/24),
                            (2,-2, 1/24), (2,-1, 2/24),  (2, 0, 1/24)],
}


# ---------------------------------------------------------------------------
# Core colour error-diffusion — fully vectorised, zero Python pixel loop
# ---------------------------------------------------------------------------

def _palette_ed_vectorised(
    arr: np.ndarray,          # float32 (H, W, 3) in [0..255]
    pal: np.ndarray,          # float32 (K, 3)
    pal_lab: np.ndarray,      # float32 (K, 3) L*a*b*
    coeffs: list[tuple],      # [(dy, dx, w), ...]  — weights already divided
) -> np.ndarray:
    """Row-by-row vectorised colour error diffusion.

    For each row all W pixels are snapped to the nearest palette colour in one
    batched call (GPU or CPU einsum).  Error is spread to neighbouring rows
    using plain slice assignment — no np.add.at, no index arrays.
    """
    h, w, _ = arr.shape
    out = arr.copy()

    for y in range(h):
        row     = out[y].copy()              # (W, 3) snapshot before snap
        idxs    = _nearest_palette_indices(row, pal, pal_lab)
        snapped = pal[idxs]                  # (W, 3)
        err     = row - snapped              # (W, 3) quantisation error
        out[y]  = snapped

        for dy, dx, wt in coeffs:
            ny = y + dy
            if ny >= h:
                continue
            if dx > 0:
                if w > dx:
                    out[ny, dx:] = np.clip(out[ny, dx:] + err[:w - dx] * wt, 0, 255)
            elif dx < 0:
                adx = -dx
                if w > adx:
                    out[ny, :w - adx] = np.clip(out[ny, :w - adx] + err[adx:] * wt, 0, 255)
            else:
                out[ny] = np.clip(out[ny] + err * wt, 0, 255)

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public colour dither entry points
# ---------------------------------------------------------------------------

def palette_dither(image: Image.Image, palette: list[tuple],
                   method: str = "Floyd-Steinberg",
                   threshold: float = 128.0) -> Image.Image:
    """High-quality colour error-diffusion dither — fully vectorised."""
    arr     = np.array(image.convert("RGB"), dtype=np.float32)
    pal     = np.asarray(palette, dtype=np.float32)
    pal_lab = _get_pal_lab(pal)
    coeffs  = _COEFF_TABLES.get(method, _COEFF_TABLES["Floyd-Steinberg"])
    result  = _palette_ed_vectorised(arr, pal, pal_lab, coeffs)
    return Image.fromarray(result, mode="RGB")


def palette_dither_fast(image: Image.Image, palette: list[tuple]) -> Image.Image:
    """Ordered dither + GPU palette snap. Kept for reference / future debug use.
    Not called by apply_dither — preview and final both use palette_dither."""
    arr  = np.array(image.convert("RGB"), dtype=np.float32)
    pal  = np.asarray(palette, dtype=np.float32)
    h, w = arr.shape[:2]

    bayer = tile(_BAYER_4x4, h, w)
    noise = (bayer - 128.0) * 0.