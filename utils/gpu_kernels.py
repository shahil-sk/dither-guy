"""GPU acceleration layer for Dither Guy.

Backend priority:
  1. CuPy   – NVIDIA CUDA (fastest)
  2. PyOpenCL – any OpenCL device (AMD, Intel, Apple Silicon, NVIDIA)
  3. NumPy   – pure-CPU fallback (always available)

Public API
----------
GPU_BACKEND : str          – "cuda" | "opencl" | "cpu"
to_gpu(arr)  -> array      – move ndarray to active device
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

_xp        = np          # active array namespace (numpy / cupy)
GPU_BACKEND: str = "cpu"

try:
    import cupy as cp               # type: ignore
    cp.array([0])                   # force device init — raises if no CUDA
    _xp         = cp
    GPU_BACKEND = "cuda"
except Exception:
    cp = None

if GPU_BACKEND == "cpu":
    try:
        import pyopencl as cl           # type: ignore
        import pyopencl.array as cl_array  # type: ignore
        _ctx   = cl.create_some_context(interactive=False)
        _queue = cl.CommandQueue(_ctx)
        GPU_BACKEND = "opencl"
    except Exception:
        cl = None
        _ctx = _queue = None

# ---------------------------------------------------------------------------
# Transfer helpers
# ---------------------------------------------------------------------------

def to_gpu(arr: np.ndarray):
    """Upload a NumPy array to the active device; no-op on CPU backend."""
    if GPU_BACKEND == "cuda" and cp is not None:
        return cp.asarray(arr)
    if GPU_BACKEND == "opencl" and cl is not None:
        import pyopencl.array as cl_array  # type: ignore
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
    """Return cupy or numpy depending on active backend."""
    return _xp

# ---------------------------------------------------------------------------
# GPU-accelerated ordered dither  (Bayer / Void-and-Cluster / etc.)
# Operates on a float32 grayscale array `a` and a pre-tiled matrix.
# ---------------------------------------------------------------------------

def gpu_ordered_dither(a: np.ndarray, tiled: np.ndarray, t: float) -> np.ndarray:
    """Threshold + ordered matrix entirely on GPU. Returns uint8 host array."""
    if GPU_BACKEND == "cpu":
        # CPU path: identical to previous numpy code
        return np.where(a + (tiled - 128.) * (1. - t / 255.) > t, 255, 0).astype(np.uint8)

    xp  = _active_np()
    ga  = to_gpu(a.astype(np.float32))
    gt  = to_gpu(tiled.astype(np.float32))
    out = xp.where(ga + (gt - 128.) * (1. - t / 255.) > t,
                   xp.float32(255.), xp.float32(0.))
    return from_gpu(out.astype(xp.uint8))


# ---------------------------------------------------------------------------
# GPU-accelerated Floyd-Steinberg
# Full pixel-serial error diffusion cannot be data-parallelised perfectly,
# but moving the array to GPU and back is only worthwhile for very large
# images.  For small/medium sizes the CPU path is taken transparently.
# ---------------------------------------------------------------------------

_GPU_FS_THRESHOLD = 1024 * 1024   # pixels: below this use CPU

def gpu_fs_dither(a: np.ndarray, t: float) -> np.ndarray:
    """Floyd-Steinberg on GPU (CUDA only via CuPy custom kernel).

    Falls back to CPU for OpenCL or small images.
    """
    if GPU_BACKEND != "cuda" or cp is None or a.size < _GPU_FS_THRESHOLD:
        # CPU fallback — import lazily to avoid circular imports
        from .dither_kernels import _fs_vectorised
        return _fs_vectorised(a.copy(), t)

    # CUDA kernel: one thread per row, serial within row (matches CPU semantics)
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
            if (x + 1 < w) a[y * w + x + 1]   = fminf(255.f, fmaxf(0.f, a[y * w + x + 1]   + e * 0.4375f));
            if (y + 1 < h) {
                if (x > 0)    a[(y+1)*w + x-1] = fminf(255.f, fmaxf(0.f, a[(y+1)*w + x-1]  + e * 0.1875f));
                              a[(y+1)*w + x  ] = fminf(255.f, fmaxf(0.f, a[(y+1)*w + x  ]  + e * 0.3125f));
                if (x + 1 < w) a[(y+1)*w + x+1] = fminf(255.f, fmaxf(0.f, a[(y+1)*w + x+1] + e * 0.0625f));
            }
        }
    }
    """, 'fs_kernel')

    ga = cp.asarray(a.astype(np.float32))
    h, w = ga.shape
    threads = 128
    blocks  = (h + threads - 1) // threads
    _fs_cuda_kernel((blocks,), (threads,), (ga, h, w, float(t)))
    cp.cuda.Stream.null.synchronize()
    return from_gpu(cp.clip(ga, 0, 255).astype(cp.uint8))


# ---------------------------------------------------------------------------
# GPU-accelerated palette nearest-neighbour in L*a*b* space
# ---------------------------------------------------------------------------

def gpu_rgb_to_lab(rgb: np.ndarray):  # -> device array or np.ndarray
    """Vectorised sRGB → L*a*b* on the active device."""
    xp = _active_np()
    r  = to_gpu(rgb.astype(np.float32)) / 255.0
    mask = r > 0.04045
    r = xp.where(mask, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
    X = r[..., 0] * 0.4124 + r[..., 1] * 0.3576 + r[..., 2] * 0.1805
    Y = r[..., 0] * 0.2126 + r[..., 1] * 0.7152 + r[..., 2] * 0.0722
    Z = r[..., 0] * 0.0193 + r[..., 1] * 0.1192 + r[..., 2] * 0.9505
    xyz = xp.stack([X / 0.9505, Y / 1.000, Z / 1.089], axis=-1)
    eps = 0.008856; kappa = 903.3
    f = xp.where(xyz > eps, xp.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return xp.stack([L, a, b], axis=-1)


def gpu_palette_nearest(flat: np.ndarray, pal_lab: np.ndarray) -> np.ndarray:
    """Return argmin palette index for each pixel row of `flat` (N,3) RGB.

    Uses the GPU for the distance matrix computation; returns a host int array.
    """
    if GPU_BACKEND == "cpu":
        # Pure-numpy path
        pix_lab = _cpu_rgb_to_lab(flat)
        diff    = pix_lab[:, np.newaxis, :] - pal_lab[np.newaxis, :, :]
        dists   = np.einsum('nkc,nkc->nk', diff, diff)
        return np.argmin(dists, axis=1)

    xp = _active_np()
    gpix_lab = gpu_rgb_to_lab(flat)            # (N, 3)  device
    gpal_lab = to_gpu(pal_lab.astype(np.float32))  # (K, 3)  device
    # (N, 1, 3) - (1, K, 3) -> (N, K, 3)
    diff  = gpix_lab[:, xp.newaxis, :] - gpal_lab[xp.newaxis, :, :]
    if GPU_BACKEND == "cuda" and cp is not None:
        dists = cp.einsum('nkc,nkc->nk', diff, diff)
    else:
        # OpenCL / fallback: manual sum
        dists = xp.sum(diff ** 2, axis=2)
    return from_gpu(xp.argmin(dists, axis=1))


def _cpu_rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CPU-only L*a*b* conversion (used as fallback inside gpu_palette_nearest)."""
    r = rgb.astype(np.float32) / 255.0
    mask = r > 0.04045
    r = np.where(mask, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
    X = r[:, 0] * 0.4124 + r[:, 1] * 0.3576 + r[:, 2] * 0.1805
    Y = r[:, 0] * 0.2126 + r[:, 1] * 0.7152 + r[:, 2] * 0.0722
    Z = r[:, 0] * 0.0193 + r[:, 1] * 0.1192 + r[:, 2] * 0.9505
    xyz = np.stack([X / 0.9505, Y, Z / 1.089], axis=-1)
    eps = 0.008856; kappa = 903.3
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=-1)
