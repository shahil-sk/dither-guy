from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from .matrices import tile, ORDERED_MATRICES, _DOT_CLASS, _BAYER_4x4
from .palettes import PALETTES

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
    out = 255.0 - (255.0 - base) * (255.0 - glow_lyr) / 255.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

# ---------------------------------------------------------------------------
# Palette dithering (colour images)
# ---------------------------------------------------------------------------

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    r = rgb / 255.0
    mask = r > 0.04045
    r = np.where(mask, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
    X = r[..., 0] * 0.4124 + r[..., 1] * 0.3576 + r[..., 2] * 0.1805
    Y = r[..., 0] * 0.2126 + r[..., 1] * 0.7152 + r[..., 2] * 0.0722
    Z = r[..., 0] * 0.0193 + r[..., 1] * 0.1192 + r[..., 2] * 0.9505
    xyz = np.stack([X / 0.9505, Y / 1.000, Z / 1.089], axis=-1)
    eps = 0.008856
    kappa = 903.3
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def palette_dither(image: Image.Image, palette: list[tuple],
                   method: str = "Floyd-Steinberg",
                   threshold: float = 128.0) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    pal = np.array(palette, dtype=np.float32)
    h, w = arr.shape[:2]
    pal_lab = _rgb_to_lab(pal)
    out = arr.copy()

    def nearest(pixel_rgb: np.ndarray) -> np.ndarray:
        p_lab = _rgb_to_lab(pixel_rgb[np.newaxis])[0]
        dists = np.sum((pal_lab - p_lab) ** 2, axis=1)
        return pal[np.argmin(dists)]

    coeff_map = {
        "Floyd-Steinberg":     [(0,1,7/16),(1,-1,3/16),(1,0,5/16),(1,1,1/16)],
        "Atkinson":            [(0,1,1/8),(0,2,1/8),(1,-1,1/8),(1,0,1/8),(1,1,1/8),(2,0,1/8)],
        "Sierra":              [(0,1,5/32),(0,2,3/32),(1,-2,2/32),(1,-1,4/32),(1,0,5/32),(1,1,4/32),(1,2,2/32),(2,-1,2/32),(2,0,3/32),(2,1,2/32)],
        "Burkes":              [(0,1,8/32),(0,2,4/32),(1,-2,2/32),(1,-1,4/32),(1,0,8/32),(1,1,4/32),(1,2,2/32)],
        "Stucki":              [(0,1,8/42),(0,2,4/42),(1,-2,2/42),(1,-1,4/42),(1,0,8/42),(1,1,4/42),(1,2,2/42),(2,-2,1/42),(2,-1,2/42),(2,0,4/42),(2,1,2/42),(2,2,1/42)],
        "Jarvis-Judice-Ninke": [(0,1,7/48),(0,2,5/48),(1,-2,3/48),(1,-1,5/48),(1,0,7/48),(1,1,5/48),(1,2,3/48),(2,-2,1/48),(2,-1,3/48),(2,0,5/48),(2,1,3/48),(2,2,1/48)],
    }
    coeffs = coeff_map.get(method, coeff_map["Floyd-Steinberg"])

    for y in range(h):
        for x in range(w):
            old = out[y, x].copy()
            new = nearest(old)
            out[y, x] = new
            err = old - new
            for dy, dx, w_coeff in coeffs:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    out[ny, nx] = np.clip(out[ny, nx] + err * w_coeff, 0, 255)

    return Image.fromarray(out.astype(np.uint8), mode="RGB")


def palette_dither_fast(image: Image.Image, palette: list[tuple]) -> Image.Image:
    arr  = np.array(image.convert("RGB"), dtype=np.float32)
    pal  = np.array(palette, dtype=np.float32)
    h, w = arr.shape[:2]
    bayer = tile(_BAYER_4x4, h, w)
    noise = (bayer - 128.0) * 0.3
    noisy = np.clip(arr + noise[:, :, np.newaxis], 0, 255)
    flat  = noisy.reshape(-1, 3)
    dists = np.sum((flat[:, np.newaxis, :] - pal[np.newaxis, :, :]) ** 2, axis=2)
    idxs  = np.argmin(dists, axis=1)
    result = pal[idxs].reshape(h, w, 3)
    return Image.fromarray(result.astype(np.uint8), mode="RGB")

# ---------------------------------------------------------------------------
# Error diffusion kernels (JIT or NumPy fallbacks)
# ---------------------------------------------------------------------------

if _NUMBA:
    @njit(cache=True, fastmath=True)
    def _fs_core(a, t):
        h, w = a.shape
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = old - new
                a[y, x] = new
                if x+1 < w: a[y, x+1] += e * 0.4375
                if y+1 < h:
                    if x > 0: a[y+1, x-1] += e * 0.1875
                    a[y+1, x] += e * 0.3125
                    if x+1 < w: a[y+1, x+1] += e * 0.0625
        return a

    @njit(cache=True, fastmath=True)
    def _atkinson_core(a, t):
        h, w = a.shape
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = (old - new) * 0.125
                a[y, x] = new
                if x+1 < w: a[y, x+1] += e
                if x+2 < w: a[y, x+2] += e
                if y+1 < h:
                    if x > 0: a[y+1, x-1] += e
                    a[y+1, x] += e
                    if x+1 < w: a[y+1, x+1] += e
                if y+2 < h: a[y+2, x] += e
        return a

    @njit(cache=True, fastmath=True)
    def _jjn_core(a, t):
        h, w = a.shape; d = 48.0
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = old - new
                a[y, x] = new
                if x+1<w: a[y,x+1]+=e*7/d
                if x+2<w: a[y,x+2]+=e*5/d
                if y+1<h:
                    if x>1: a[y+1,x-2]+=e*3/d
                    if x>0: a[y+1,x-1]+=e*5/d
                    a[y+1,x]+=e*7/d
                    if x+1<w: a[y+1,x+1]+=e*5/d
                    if x+2<w: a[y+1,x+2]+=e*3/d
                if y+2<h:
                    if x>1: a[y+2,x-2]+=e*1/d
                    if x>0: a[y+2,x-1]+=e*3/d
                    a[y+2,x]+=e*5/d
                    if x+1<w: a[y+2,x+1]+=e*3/d
                    if x+2<w: a[y+2,x+2]+=e*1/d
        return a

    @njit(cache=True, fastmath=True)
    def _stucki_core(a, t):
        h, w = a.shape; d = 42.0
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = old - new
                a[y, x] = new
                if x+1<w: a[y,x+1]+=e*8/d
                if x+2<w: a[y,x+2]+=e*4/d
                if y+1<h:
                    if x>1: a[y+1,x-2]+=e*2/d
                    if x>0: a[y+1,x-1]+=e*4/d
                    a[y+1,x]+=e*8/d
                    if x+1<w: a[y+1,x+1]+=e*4/d
                    if x+2<w: a[y+1,x+2]+=e*2/d
                if y+2<h:
                    if x>1: a[y+2,x-2]+=e*1/d
                    if x>0: a[y+2,x-1]+=e*2/d
                    a[y+2,x]+=e*4/d
                    if x+1<w: a[y+2,x+1]+=e*2/d
                    if x+2<w: a[y+2,x+2]+=e*1/d
        return a

    @njit(cache=True, fastmath=True)
    def _sierra_core(a, t):
        h, w = a.shape; d = 32.0
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = old - new
                a[y, x] = new
                if x+1<w: a[y,x+1]+=e*5/d
                if x+2<w: a[y,x+2]+=e*3/d
                if y+1<h:
                    if x>1: a[y+1,x-2]+=e*2/d
                    if x>0: a[y+1,x-1]+=e*4/d
                    a[y+1,x]+=e*5/d
                    if x+1<w: a[y+1,x+1]+=e*4/d
                    if x+2<w: a[y+1,x+2]+=e*2/d
                if y+2<h:
                    if x>0: a[y+2,x-1]+=e*2/d
                    a[y+2,x]+=e*3/d
                    if x+1<w: a[y+2,x+1]+=e*2/d
        return a

    @njit(cache=True, fastmath=True)
    def _sierra_lite_core(a, t):
        h, w = a.shape
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = old - new
                a[y, x] = new
                if x+1<w: a[y,x+1]+=e*0.5
                if y+1<h:
                    if x>0: a[y+1,x-1]+=e*0.25
                    a[y+1,x]+=e*0.25
        return a

    @njit(cache=True, fastmath=True)
    def _nakano_core(a, t):
        h, w = a.shape
        for y in range(h):
            for x in range(w):
                old = a[y, x]; new = 255.0 if old > t else 0.0; e = old - new
                a[y, x] = new
                if x+1<w: a[y,x+1]+=e*(8/24)
                if y+1<h:
                    if x>0: a[y+1,x-1]+=e*(4/24)
                    a[y+1,x]+=e*(4/24)
                    if x+1<w: a[y+1,x+1]+=e*(4/24)
                if y+2<h:
                    if x>1: a[y+2,x-2]+=e*(1/24)
                    if x>0: a[y+2,x-1]+=e*(2/24)
                    a[y+2,x]+=e*(1/24)
        return a

    def _wrap_jit(fn):
        def inner(a: np.ndarray, t: float) -> np.ndarray:
            a = a.astype(np.float64)
            np.clip(a, 0, 255, out=a)
            result = fn(a, float(t))
            return np.clip(result, 0, 255).astype(np.uint8)
        return inner

    _fs_vectorised          = _wrap_jit(_fs_core)
    _atkinson_vectorised    = _wrap_jit(_atkinson_core)
    _sierra_vectorised      = _wrap_jit(_sierra_core)
    _sierra_lite_vectorised = _wrap_jit(_sierra_lite_core)
    _nakano_vectorised      = _wrap_jit(_nakano_core)
    _jjn_vectorised         = _wrap_jit(_jjn_core)
    _stucki_vectorised      = _wrap_jit(_stucki_core)

else:
    def _fs_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
            a[y, 1:] = np.clip(a[y, 1:] + e[:-1] * 0.4375, 0, 255)
            if y+1 < h:
                a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:] * 0.1875, 0, 255)
                a[y+1]      = np.clip(a[y+1]       + e     * 0.3125, 0, 255)
                if w > 1: a[y+1, 1:] = np.clip(a[y+1, 1:] + e[:-1] * 0.0625, 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)

    def _atkinson_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = (row - nw) * 0.125
            a[y, 1:] = np.clip(a[y, 1:] + e[:-1], 0, 255)
            if w > 2: a[y, 2:] = np.clip(a[y, 2:] + e[:-2], 0, 255)
            if y+1 < h:
                a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:], 0, 255)
                a[y+1]      = np.clip(a[y+1]       + e,     0, 255)
                if w > 1: a[y+1, 1:] = np.clip(a[y+1, 1:] + e[:-1], 0, 255)
            if y+2 < h: a[y+2] = np.clip(a[y+2] + e, 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)

    def _sierra_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape; d = 32.
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
            if w > 1: a[y, 1:] = np.clip(a[y, 1:] + e[:-1] * 5/d, 0, 255)
            if w > 2: a[y, 2:] = np.clip(a[y, 2:] + e[:-2] * 3/d, 0, 255)
            if y+1 < h:
                if w > 2: a[y+1, :-2] = np.clip(a[y+1, :-2] + e[2:] * 2/d, 0, 255)
                if w > 1: a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:] * 4/d, 0, 255)
                a[y+1] = np.clip(a[y+1] + e * 5/d, 0, 255)
                if w > 1: a[y+1, 1:] = np.clip(a[y+1, 1:] + e[:-1] * 4/d, 0, 255)
                if w > 2: a[y+1, 2:] = np.clip(a[y+1, 2:] + e[:-2] * 2/d, 0, 255)
            if y+2 < h:
                if w > 1: a[y+2, :-1] = np.clip(a[y+2, :-1] + e[1:] * 2/d, 0, 255)
                a[y+2] = np.clip(a[y+2] + e * 3/d, 0, 255)
                if w > 1: a[y+2, 1:] = np.clip(a[y+2, 1:] + e[:-1] * 2/d, 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)

    def _sierra_lite_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
            a[y, 1:] = np.clip(a[y, 1:] + e[:-1] * 0.5, 0, 255)
            if y+1 < h:
                if w > 1: a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:] * 0.25, 0, 255)
                a[y+1] = np.clip(a[y+1] + e * 0.25, 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)

    def _nakano_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
            a[y, 1:] = np.clip(a[y, 1:] + e[:-1] * (8/24), 0, 255)
            if y+1 < h:
                if w > 1: a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:] * (4/24), 0, 255)
                a[y+1] = np.clip(a[y+1] + e * (4/24), 0, 255)
                if w > 1: a[y+1, 1:] = np.clip(a[y+1, 1:] + e[:-1] * (4/24), 0, 255)
            if y+2 < h:
                if w > 2: a[y+2, :-2] = np.clip(a[y+2, :-2] + e[2:] * (1/24), 0, 255)
                if w > 1: a[y+2, :-1] = np.clip(a[y+2, :-1] + e[1:] * (2/24), 0, 255)
                a[y+2] = np.clip(a[y+2] + e * (1/24), 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)

    def _jjn_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape; d = 48.
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
            if w > 1: a[y, 1:] = np.clip(a[y, 1:] + e[:-1] * 7/d, 0, 255)
            if w > 2: a[y, 2:] = np.clip(a[y, 2:] + e[:-2] * 5/d, 0, 255)
            if y+1 < h:
                if w > 2: a[y+1, :-2] = np.clip(a[y+1, :-2] + e[2:] * 3/d, 0, 255)
                if w > 1: a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:] * 5/d, 0, 255)
                a[y+1] = np.clip(a[y+1] + e * 7/d, 0, 255)
                if w > 1: a[y+1, 1:] = np.clip(a[y+1, 1:] + e[:-1] * 5/d, 0, 255)
                if w > 2: a[y+1, 2:] = np.clip(a[y+1, 2:] + e[:-2] * 3/d, 0, 255)
            if y+2 < h:
                if w > 2: a[y+2, :-2] = np.clip(a[y+2, :-2] + e[2:] * 1/d, 0, 255)
                if w > 1: a[y+2, :-1] = np.clip(a[y+2, :-1] + e[1:] * 3/d, 0, 255)
                a[y+2] = np.clip(a[y+2] + e * 5/d, 0, 255)
                if w > 1: a[y+2, 1:] = np.clip(a[y+2, 1:] + e[:-1] * 3/d, 0, 255)
                if w > 2: a[y+2, 2:] = np.clip(a[y+2, 2:] + e[:-2] * 1/d, 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)

    def _stucki_vectorised(a: np.ndarray, t: float) -> np.ndarray:
        h, w = a.shape; d = 42.
        for y in range(h):
            row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
            if w > 1: a[y, 1:] = np.clip(a[y, 1:] + e[:-1] * 8/d, 0, 255)
            if w > 2: a[y, 2:] = np.clip(a[y, 2:] + e[:-2] * 4/d, 0, 255)
            if y+1 < h:
                if w > 2: a[y+1, :-2] = np.clip(a[y+1, :-2] + e[2:] * 2/d, 0, 255)
                if w > 1: a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:] * 4/d, 0, 255)
                a[y+1] = np.clip(a[y+1] + e * 8/d, 0, 255)
                if w > 1: a[y+1, 1:] = np.clip(a[y+1, 1:] + e[:-1] * 4/d, 0, 255)
                if w > 2: a[y+1, 2:] = np.clip(a[y+1, 2:] + e[:-2] * 2/d, 0, 255)
            if y+2 < h:
                if w > 2: a[y+2, :-2] = np.clip(a[y+2, :-2] + e[2:] * 1/d, 0, 255)
                if w > 1: a[y+2, :-1] = np.clip(a[y+2, :-1] + e[1:] * 2/d, 0, 255)
                a[y+2] = np.clip(a[y+2] + e * 4/d, 0, 255)
                if w > 1: a[y+2, 1:] = np.clip(a[y+2, 1:] + e[:-1] * 2/d, 0, 255)
                if w > 2: a[y+2, 2:] = np.clip(a[y+2, 2:] + e[:-2] * 1/d, 0, 255)
            a[y] = nw
        return np.clip(a, 0, 255).astype(np.uint8)


def _variable_error_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape
    for y in range(h - 1):
        row = a[y, 1:w-1].copy(); nw = np.where(row > t, 255., 0.); e = row - nw; f = row / 255.
        a[y, 2:w]   = np.clip(a[y, 2:w]   + e * 7. * f / 16.,    0, 255)
        a[y+1, 0:w-2] = np.clip(a[y+1, 0:w-2] + e * 3. * (1-f) / 16., 0, 255)
        a[y+1, 1:w-1] = np.clip(a[y+1, 1:w-1] + e * 5. / 16.,   0, 255)
        a[y+1, 2:w]   = np.clip(a[y+1, 2:w]   + e * 1. / 16.,   0, 255)
        a[y, 1:w-1] = nw
    return np.clip(a, 0, 255).astype(np.uint8)


def _burkes_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape; d = 32.0
    for y in range(h):
        row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
        if w > 1: a[y, 1:]    = np.clip(a[y, 1:]    + e[:-1] * 8/d, 0, 255)
        if w > 2: a[y, 2:]    = np.clip(a[y, 2:]    + e[:-2] * 4/d, 0, 255)
        if y+1 < h:
            if w > 2: a[y+1, :-2] = np.clip(a[y+1, :-2] + e[2:]  * 2/d, 0, 255)
            if w > 1: a[y+1, :-1] = np.clip(a[y+1, :-1] + e[1:]  * 4/d, 0, 255)
            a[y+1]             = np.clip(a[y+1]           + e      * 8/d, 0, 255)
            if w > 1: a[y+1, 1:]  = np.clip(a[y+1, 1:]  + e[:-1] * 4/d, 0, 255)
            if w > 2: a[y+1, 2:]  = np.clip(a[y+1, 2:]  + e[:-2] * 2/d, 0, 255)
        a[y] = nw
    return np.clip(a, 0, 255).astype(np.uint8)


def _stevenson_arce_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape; d = 200.0
    coeffs = [
        (0, 2, 32),
        (1,-3,12),(1,-1,26),(1, 1,30),(1, 3,16),
        (2,-2,12),(2, 0,26),(2, 2,12),
        (3,-3, 5),(3,-1,12),(3, 1,12),(3, 3, 5),
    ]
    for y in range(h):
        row = a[y].copy(); nw = np.where(row > t, 255., 0.); e = row - nw
        for dy, dx, wt in coeffs:
            ny = y + dy
            if ny >= h: continue
            if dx > 0:
                if w > dx: a[ny, dx:] = np.clip(a[ny, dx:] + e[:-dx] * (wt/d), 0, 255)
            elif dx < 0:
                adx = -dx
                if w > adx: a[ny, :w-adx] = np.clip(a[ny, :w-adx] + e[adx:] * (wt/d), 0, 255)
            else:
                a[ny] = np.clip(a[ny] + e * (wt/d), 0, 255)
        a[y] = nw
    return np.clip(a, 0, 255).astype(np.uint8)


def _ostromoukhov_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    _TABLE = [
        (13,0,5,18),(13,0,5,18),(21,0,10,31),(7,0,4,11),
        (8,0,5,13),(47,3,28,78),(23,3,13,39),(15,3,8,26),
        (22,5,10,37),(56,14,21,91),(28,8,9,45),(19,6,5,30),
        (14,5,3,22),(7,3,1,11),(65,32,7,104),(23,12,2,37),
        (23,12,2,37),(65,32,7,104),(7,3,1,11),(14,5,3,22),
        (19,6,5,30),(28,8,9,45),(56,14,21,91),(22,5,10,37),
        (15,3,8,26),(23,3,13,39),(47,3,28,78),(8,0,5,13),
        (7,0,4,11),(21,0,10,31),(13,0,5,18),(13,0,5,18),
    ]
    h, w = a.shape
    for y in range(h):
        for x in range(w):
            old = a[y, x]; new = 255.0 if old > t else 0.0; err = old - new
            a[y, x] = new
            band = int(np.clip(old / 8.0, 0, 31))
            c0, c1, c2, dn = _TABLE[band]
            if dn == 0: continue
            if x+1 < w:   a[y,   x+1] = np.clip(a[y,   x+1] + err*c0/dn, 0, 255)
            if y+1 < h:
                if x > 0: a[y+1, x-1] = np.clip(a[y+1, x-1] + err*c1/dn, 0, 255)
                a[y+1, x]  = np.clip(a[y+1, x]  + err*c2/dn, 0, 255)
    return np.clip(a, 0, 255).astype(np.uint8)


_BLUE_NOISE_MASK_64 = None


def _get_blue_noise_mask(h: int, w: int) -> np.ndarray:
    global _BLUE_NOISE_MASK_64
    if _BLUE_NOISE_MASK_64 is None:
        rng  = np.random.default_rng(0xD1740)
        base = rng.integers(0, 256, (64, 64), dtype=np.uint8).astype(np.float32)
        tmp  = Image.fromarray(base.astype(np.uint8))
        blurred = np.array(tmp.filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32)
        _BLUE_NOISE_MASK_64 = np.clip(base - blurred + 128, 0, 255)
    return tile(_BLUE_NOISE_MASK_64, h, w)


def _blue_noise_mask_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape
    mask = _get_blue_noise_mask(h, w)
    return np.where(a + (mask - 128.0) * (1.0 - t / 255.0) > t, 255, 0).astype(np.uint8)


def _dot_diffusion_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape; dc = tile(_DOT_CLASS, h, w)
    for y in range(h):
        row = a[y].copy(); nw = np.where(row > t, 255., 0.); err = row - nw; cm = dc[y]
        if w > 1: a[y, 1:] = np.clip(a[y, 1:] + err[:-1] / (cm[:-1] + 1.), 0, 255)
        a[y] = nw
    return np.clip(a, 0, 255).astype(np.uint8)


def _riemersma_vectorised(a: np.ndarray, t: float) -> np.ndarray:
    h, w = a.shape; buf = [0.] * 16
    for y in range(h):
        xs = range(w) if y % 2 == 0 else range(w-1, -1, -1)
        for x in xs:
            old = float(a[y, x]) + buf[0]; new = 255. if old > t else 0.
            a[y, x] = new; buf = buf[1:] + [(old - new) * 0.0625]
    return np.clip(a, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Main dither pipeline
# ---------------------------------------------------------------------------

def apply_dither(
    img: Image.Image,
    pixel_size: int,
    threshold: float,
    replace_color: tuple,
    method: str,
    brightness: float = 1.0,
    contrast:   float = 1.0,
    blur:       float = 0,
    sharpen:    float = 0,
    glow_radius:    float = 0,
    glow_intensity: float = 0,
    preview: bool = False,
    palette_name: str = "B&W",
    custom_palette: Optional[list] = None,
) -> Image.Image:
    img = adjust(img, brightness, contrast, blur, sharpen)

    if custom_palette and len(custom_palette) >= 2:
        palette = custom_palette
    else:
        palette = PALETTES.get(palette_name, PALETTES["B&W"])

    is_bw = (palette == PALETTES["B&W"])
    effective_pixel = max(1, pixel_size * (2 if preview else 1))

    if not is_bw:
        rgb = img.convert("RGB")
        sw  = max(1, rgb.width  // effective_pixel)
        sh  = max(1, rgb.height // effective_pixel)
        rgb = rgb.resize((sw, sh), Image.NEAREST)
        if preview:
            result = palette_dither_fast(rgb, palette)
        else:
            max_dim = 300
            if sw > max_dim or sh > max_dim:
                scale = min(max_dim / sw, max_dim / sh)
                rgb = rgb.resize((max(1, int(sw*scale)), max(1, int(sh*scale))), Image.NEAREST)
            result = palette_dither(rgb, palette, method=method, threshold=threshold)
        result = result.resize((sw * effective_pixel, sh * effective_pixel), Image.NEAREST)
        if glow_radius > 0 and glow_intensity > 0:
            result = apply_glow(result, glow_radius, glow_intensity)
        return result

    img = img.convert('L')
    sw = max(1, img.width  // effective_pixel)
    sh = max(1, img.height // effective_pixel)
    img = img.resize((sw, sh), Image.NEAREST)
    a = np.array(img, dtype=np.float32)
    h, w = a.shape
    t = float(threshold)

    if method in ORDERED_MATRICES:
        tiled = tile(ORDERED_MATRICES[method], h, w)
        a = np.where(a + (tiled - 128.) * (1. - t/255.) > t, 255, 0).astype(np.uint8)
    elif method == "Crosshatch":
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        ch = (np.sin(xs[None, :] * 0.5) + np.sin(ys[:, None] * 0.5)) * 64. + 128.
        a  = np.where(a + (ch - 128.) * (1. - t/255.) > t, 255, 0).astype(np.uint8)
    # elif method == "Random":
    #     rng = np.random.default_rng()
    #     r   = rng.integers(0, 256, (h, w), dtype=np.float32)
    #     a   = np.where(a > t * 0.7 + r * 0.3, 255, 0).astype(np.uint8)
    elif method == "Blue-Noise Mask":    a = _blue_noise_mask_vectorised(a, t)
    elif method == "Floyd-Steinberg":    a = _fs_vectorised(a, t)
    elif method == "Atkinson":           a = _atkinson_vectorised(a, t)
    elif method == "Sierra":             a = _sierra_vectorised(a, t)
    elif method == "Sierra-Lite":        a = _sierra_lite_vectorised(a, t)
    elif method == "Nakano":             a = _nakano_vectorised(a, t)
    elif method == "Jarvis-Judice-Ninke": a = _jjn_vectorised(a, t)
    elif method == "Stucki":             a = _stucki_vectorised(a, t)
    elif method == "Burkes":             a = _burkes_vectorised(a, t)
    elif method == "Stevenson-Arce":     a = _stevenson_arce_vectorised(a, t)
    elif method == "Ostromoukhov":       a = _ostromoukhov_vectorised(a, t)
    elif method == "Variable-Error":     a = _variable_error_vectorised(a, t)
    elif method == "Dot-Diffusion":      a = _dot_diffusion_vectorised(a, t)
    elif method == "Riemersma":          a = _riemersma_vectorised(a, t)
    else:
        a = np.where(a > t, 255, 0).astype(np.uint8)

    img = Image.fromarray(a, mode='L')
    img = img.resize((sw * effective_pixel, sh * effective_pixel), Image.NEAREST)
    img = img.convert("RGB")
    data = np.array(img)
    mask = (data[:, :, 0] == 255) & (data[:, :, 1] == 255) & (data[:, :, 2] == 255)
    data[mask] = replace_color
    img = Image.fromarray(data)

    if glow_radius > 0 and glow_intensity > 0:
        img = apply_glow(img, glow_radius, glow_intensity)

    return img
