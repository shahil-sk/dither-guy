"""GPU acceleration layer for Dither Guy.

Backend priority:
  1. CuPy    – NVIDIA CUDA (fastest)
  2. PyOpenCL – any OpenCL device (AMD, Intel, Apple Silicon, NVIDIA)
  3. NumPy   – pure-CPU fallback (always available)

Public API
----------
GPU_BACKEND : str           – "cuda" | "opencl" | "cpu"
to_gpu(arr)  -> array       – move ndarray to active device
from_gpu(arr) -> np.ndarray – bring result back to host
gpu_ordered_dither(a, tiled, t) -> array
gpu_palette_nearest(flat, pal_lab) -> np.ndarray  (index array)
gpu_rgb_to_lab(r_device)    -> array
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _try_cuda():
    try:
        import cupy as cp
        _ = cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:
        return None


def _try_opencl():
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        if not platforms:
            return None, None
        ctx = cl.Context(dev_type=cl.device_type.ALL)
        queue = cl.CommandQueue(ctx)
        return cl, (ctx, queue)
    except Exception:
        return None, None


cp = _try_cuda()
cl, cl_ctx = _try_opencl()

if cp:
    GPU_BACKEND = "cuda"
    _xp = cp
elif cl:
    GPU_BACKEND = "opencl"
    _xp = np
else:
    GPU_BACKEND = "cpu"
    _xp = np

if cl_ctx is not None:
    _ctx, _queue = cl_ctx
else:
    _ctx = _queue = None

# ---------------------------------------------------------------------------
# Transfer helpers
# ---------------------------------------------------------------------------

def to_gpu(arr: np.ndarray):
    """Upload a NumPy array to the active device; no-op on CPU backend."""
    if GPU_BACKEND == "cuda" and cp is not None:
        return cp.asarray(arr)
    if GPU_BACKEND == "opencl" and cl is not None:
        import pyopencl.array as cl_array
        return cl_array.to_device(_queue, arr.astype(arr.dtype, copy=False))
    return arr


def from_gpu(arr) -> np.ndarray:
    """Download a device array back to NumPy; no-op on CPU backend."""
    if GPU_BACKEND == "cuda" and cp is not None:
        return cp.asnumpy(arr)
    if GPU_BACKEND == "opencl" and cl is not None:
        return arr.get()
    return np.asarray(arr)


def _active_np():
    return _xp

# ---------------------------------------------------------------------------
# Precomputed L*a*b* conversion constants (host side -- uploaded on demand)
# ---------------------------------------------------------------------------
_SRGB_TO_XYZ = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
], dtype=np.float32)
_XYZ_SCALE = np.array([1.0 / 0.9505, 1.0, 1.0 / 1.089], dtype=np.float32)
_LAB_EPS   = np.float32(0.008856)
_LAB_KAPPA = np.float32(903.3)

# ---------------------------------------------------------------------------
# GPU-accelerated ordered dither
# Assumes `a` and `tiled` are already on device when GPU_BACKEND != "cpu".
# Returns a device array -- caller must call from_gpu().
# ---------------------------------------------------------------------------

def gpu_ordered_dither(a, tiled, t: float):
    """Threshold + ordered matrix on the active device.

    Expects `a` and `tiled` to already be on device (no internal transfer).
    Returns a device array; caller is responsible for from_gpu().
    On CPU backend operates on plain NumPy arrays and returns np.ndarray.
    """
    xp = _active_np()
    _t      = xp.float32(t)
    _128    = xp.float32(128.0)
    _1      = xp.float32(1.0)
    _255    = xp.float32(255.0)
    _inv255 = xp.float32(1.0 / 255.0)
    out = xp.where(
        a.astype(xp.float32) + (tiled.astype(xp.float32) - _128) * (_1 - _t * _inv255) > _t,
        _255, xp.float32(0.0))
    return out.astype(xp.uint8)


# ---------------------------------------------------------------------------
# GPU-accelerated palette nearest-neighbour in L*a*b* space
# Assumes inputs are already on device when GPU_BACKEND != "cpu".
# Returns a HOST int array (argmin is always small, safe to pull back).
# ---------------------------------------------------------------------------

def gpu_rgb_to_lab(r_device):
    """Vectorised sRGB -> L*a*b* on the active device.

    Expects `r_device` to already be on device (no internal transfer).
    Returns a device array.
    """
    xp = _active_np()

    r    = r_device.astype(xp.float32) / xp.float32(255.0)
    mask = r > xp.float32(0.04045)
    r    = xp.where(
        mask,
        ((r + xp.float32(0.055)) / xp.float32(1.055)) ** xp.float32(2.4),
        r / xp.float32(12.92)
    )

    M   = to_gpu(_SRGB_TO_XYZ)   # small constant -- cheap transfer
    sc  = to_gpu(_XYZ_SCALE)

    if GPU_BACKEND == "cuda" and cp is not None:
        xyz = xp.einsum('...c,rc->...r', r, M) * sc
    else:
        xyz = (r @ M.T) * sc

    f = xp.where(
        xyz > xp.float32(_LAB_EPS),
        xp.cbrt(xyz),
        (xp.float32(_LAB_KAPPA) * xyz + xp.float32(16.0)) / xp.float32(116.0)
    )

    L = xp.float32(116.0) * f[..., 1] - xp.float32(16.0)
    a = xp.float32(500.0) * (f[..., 0] - f[..., 1])
    b = xp.float32(200.0) * (f[..., 1] - f[..., 2])
    return xp.stack([L, a, b], axis=-1)


def gpu_palette_nearest(flat, pal_lab: np.ndarray) -> np.ndarray:
    """Return argmin palette index for each row of `flat` (N,3) RGB.

    `flat` must already be on device when GPU_BACKEND != "cpu".
    `pal_lab` is always a host NumPy array; transferred here once (small).
    Returns a HOST int array.
    """
    if GPU_BACKEND == "cpu":
        pix_lab = _cpu_rgb_to_lab(np.asarray(flat))
        diff    = pix_lab[:, np.newaxis, :] - pal_lab[np.newaxis, :, :]
        dists   = np.einsum('nkc,nkc->nk', diff, diff)
        return np.argmin(dists, axis=1)

    xp       = _active_np()
    gpix_lab = gpu_rgb_to_lab(flat)               # (N, 3) device
    gpal_lab = to_gpu(pal_lab.astype(np.float32)) # (K, 3) device -- small, OK here
    diff     = gpix_lab[:, xp.newaxis, :] - gpal_lab[xp.newaxis, :, :]
    if GPU_BACKEND == "cuda" and cp is not None:
        dists = xp.einsum('nkc,nkc->nk', diff, diff)
    else:
        dists = xp.sum(diff * diff, axis=2)
    return from_gpu(xp.argmin(dists, axis=1))


def _cpu_rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CPU-only L*a*b* conversion using a fused matrix multiply."""
    r    = rgb.astype(np.float32) / 255.0
    mask = r > 0.04045
    r    = np.where(mask, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
    xyz  = (r @ _SRGB_TO_XYZ.T) * _XYZ_SCALE
    f    = np.where(xyz > _LAB_EPS, np.cbrt(xyz), (_LAB_KAPPA * xyz + 16.0) / 116.0)
    L    = 116.0 * f[:, 1] - 16.0
    a    = 500.0 * (f[:, 0] - f[:, 1])
    b    = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=-1)
