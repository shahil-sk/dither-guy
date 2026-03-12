from __future__ import annotations

import os

_MAX_PIXELS    = 8_000_000
_HISTORY_LIMIT = 20
_DEBOUNCE_MS   = 250
_PREVIEW_SCALE = 0.5
_VIDEO_WORKERS = min(8, (os.cpu_count() or 4))
VERSION        = "3.5"

METHOD_GROUPS = {
    "Ordered": [
        "Bayer 2x2", "Bayer 4x4", "Bayer 8x8",
        "Clustered-Dot", "Halftone", "Blue-Noise",
        "Void-and-Cluster", "Pattern", "Crosshatch",
    ],
    "Error Diffusion": [
        "Floyd-Steinberg", "Atkinson", "Sierra", "Sierra-Lite",
        "Jarvis-Judice-Ninke", "Stucki", "Nakano",
        "Burkes", "Stevenson-Arce", "Ostromoukhov",
        "Variable-Error", "Dot-Diffusion", "Riemersma",
    ],
    "Stochastic": [
        "Blue-Noise Mask",
    ],
}
METHODS: list[str] = [m for ms in METHOD_GROUPS.values() for m in ms]
