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
gpu_ordered_dither(a, tiled, t) -> np.ndarray
gpu_fs_dither(a, t)             -> np.ndarray   (Floyd-Steinberg)
gpu_palette_nearest(flat, pal_lab) -> np.ndarray  (index array)
gpu_rgb_to_lab(rgb)             -> np.ndarray
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

# --- SAFE BACKEND DETECTION ---

def _try_cuda():
    try:
        import cupy as cp
        # lightweight check (no kernel compile)
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
    _xp = np  # keep numpy-like interface (avoid cl array mess)
else:
    GPU_BACKEND = "cpu"
    _xp = np

# unpack opencl context/queue if available
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
        import pyopencl.array as cl_array   # type: ignore
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
# Precomputed L*a*b* conversion constants (avoid re-boxing floats in loops)
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
# ---------------------------------------------------------------------------

def gpu_ordered_dither(a: np.ndarray, tiled: np.ndarray, t: float) -> np.ndarray:
    """Threshold + ordered matrix entirely on GPU. Returns uint8 host array."""
    if GPU_BACKEND == "cpu":
        return np.where(
            a + (tiled - 128.) * (1. - t / 255.) > t, 255, 0
        ).astype(np.uint8)

    xp  = _active_np()
    ga  = to_gpu(a.astype(np.float32))
    gt  = to_gpu(tiled.astype(np.float32))
    out = xp.where(
        ga + (gt - np.float32(128.)) * (np.float32(1.) - np.float32(t / 255.)) > np.float32(t),
        xp.float32(255.), xp.float32(0.))
    return from_gpu(out.astype(xp.uint8))


# ---------------------------------------------------------------------------
# GPU-accelerated Floyd-Steinberg
# ---------------------------------------------------------------------------

_GPU_FS_THRESHOLD = 1024 * 1024   # pixels: below this use CPU

def gpu_fs_dither(a: np.ndarray, t: float) -> np.ndarray:
    """Floyd-Steinberg on GPU (CUDA only via CuPy custom kernel).

    Falls back to CPU for OpenCL or small images.
    """
    if GPU_BACKEND != "cuda" or cp is None or a.size < _GPU_FS_THRESHOLD:
        from .dither_kernels import _fs_vectorised
        return _fs_vectorised(a.copy(), t)

    _fs_cuda_kernel = cp.RawKernel(r"""
    extern "C" __global__
    void fs_kernel(float* a, int h, int w, float t) {
        int y = blockIdx.x * blockDim.x + threadIdx.x;
        if (y >= h) return;
        for (int x = 0; x < w; x++) {
            float old = a[y * w + x];
            float nw  = (old > t) ? 255.f : 0.f;
            float e   = old - nw;
            a[y * w + x] = nw;
            if (x + 1 < w) a[y * w + x + 1]     = fminf(255.f, fmaxf(0.f, a[y * w + x + 1]     + e * 0.4375f));
            if (y + 1 < h) {
                if (x > 0) a[(y+1)*w + x-1]      = fminf(255.f, fmaxf(0.f, a[(y+1)*w + x-1]    + e * 0.1875f));
                           a[(y+1)*w + x  ]       = fminf(255.f, fmaxf(0.f, a[(y+1)*w + x  ]    + e * 0.3125f));
                if (x+1 < w) a[(y+1)*w + x+1]    = fminf(255.f, fmaxf(0.f, a[(y+1)*w + x+1]   + e * 0.0625f));
            }
        }
    }
    """, 'fs_kernel')

    ga = cp.asarray(a.astype(np.float32))
    h, w    = ga.shape
    threads = 128
    blocks  = (h + threads - 1) // threads
    _fs_cuda_kernel((blocks,), (threads,), (ga, h, w, float(t)))
    cp.cuda.Stream.null.synchronize()
    return from_gpu(cp.clip(ga, 0, 255).astype(cp.uint8))


# ---------------------------------------------------------------------------
# GPU-accelerated palette nearest-neighbour in L*a*b* space
# ---------------------------------------------------------------------------

def gpu_rgb_to_lab(rgb: np.ndarray):
    """Vectorised sRGB → L*a*b* on the active device.

    Uses a matrix multiply for the linear-RGB → XYZ step, which is faster
    than three separate dot products on both CPU and GPU.
    """
    xp = _active_np()
    r  = to_gpu(rgb.astype(np.float32)) / np.float32(255.0)
    mask = r > np.float32(0.04045)
    r    = xp.where(mask, ((r + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4),
                    r / np.float32(12.92))

    # Matrix multiply: (..., 3) @ (3, 3)^T  — fused into one einsum
    if GPU_BACKEND == "cuda" and cp is not None:
        M   = cp.asarray(_SRGB_TO_XYZ)
        xyz = cp.einsum('...c,rc->...r', r, M)
    else:
        xyz = r @ _SRGB_TO_XYZ.T          # works for numpy & opencl-array-like

    sc  = to_gpu(_XYZ_SCALE)
    xyz = xyz * sc

    f = xp.where(xyz > _LAB_EPS,
                 xp.cbrt(xyz),
                 (_LAB_KAPPA * xyz + np.float32(16.0)) / np.float32(116.0))

    L = np.float32(116.0) * f[..., 1] - np.float32(16.0)
    a = np.float32(500.0) * (f[..., 0] - f[..., 1])
    b = np.float32(200.0) * (f[..., 1] - f[..., 2])
    return xp.stack([L, a, b], axis=-1)


def gpu_palette_nearest(flat: np.ndarray, pal_lab: np.ndarray) -> np.ndarray:
    """Return argmin palette index for each pixel row of `flat` (N,3) RGB.

    Uses the GPU for the distance matrix computation; returns a host int array.
    """
    if GPU_BACKEND == "cpu":
        pix_lab = _cpu_rgb_to_lab(flat)
        diff    = pix_lab[:, np.newaxis, :] - pal_lab[np.newaxis, :, :]
        dists   = np.einsum('nkc,nkc->nk', diff, diff)
        return np.argmin(dists, axis=1)

    xp       = _active_np()
    gpix_lab = gpu_rgb_to_lab(flat)                        # (N, 3)  device
    gpal_lab = to_gpu(pal_lab.astype(np.float32))          # (K, 3)  device
    diff     = gpix_lab[:, xp.newaxis, :] - gpal_lab[xp.newaxis, :, :]
    if GPU_BACKEND == "cuda" and cp is not None:
        dists = cp.einsum('nkc,nkc->nk', diff, diff)
    else:
        dists = xp.sum(diff * diff, axis=2)                # diff**2 → diff*diff avoids pow
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
