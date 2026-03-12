# dither_guy/presets.py
# Persistence layer for named parameter presets.
#
# Presets are stored as individual JSON files inside the PRESETS_DIR folder
# (default: ./presets/).  Each file contains a flat dict of parameter values
# that map 1-to-1 with the keys returned by ControlPanel.get_params().
#
# Public API
# ──────────
#   save_preset(name, params)  →  Path
#   load_preset(name)          →  dict | None
#   list_presets()             →  list[str]
#   delete_preset(name)        →  bool

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from utils.constants import PRESETS_DIR


def save_preset(name: str, params: dict) -> Path:
    """Serialise *params* to ``PRESETS_DIR/<name>.json``.

    Sanitises the filename by stripping characters that are unsafe on most
    file systems.  Tuple colours are converted to lists for JSON compatibility.
    Returns the final Path written.
    """
    PRESETS_DIR.mkdir(exist_ok=True)
    safe = "".join(c for c in name if c.isalnum() or c in " _-").strip() or "preset"
    path = PRESETS_DIR / f"{safe}.json"
    p = dict(params)
    if isinstance(p.get("color"), tuple):
        p["color"] = list(p["color"])
    with open(path, "w") as f:
        json.dump(p, f, indent=2)
    return path


def load_preset(name: str) -> Optional[dict]:
    """Load a preset by stem name (with or without the ``.json`` suffix).

    Returns ``None`` if the file does not exist or cannot be parsed.
    Colour lists are converted back to tuples on load.
    """
    n    = name if name.endswith(".json") else f"{name}.json"
    path = PRESETS_DIR / n
    if not path.exists():
        return None
    try:
        with open(path) as f:
            p = json.load(f)
        if "color" in p and isinstance(p["color"], list):
            p["color"] = tuple(p["color"])
        return p
    except Exception:
        return None


def list_presets() -> list[str]:
    """Return a sorted list of saved preset names (without the ``.json`` extension)."""
    if not PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


def delete_preset(name: str) -> bool:
    """Delete a preset by name.  Returns ``True`` on success."""
    n    = name if name.endswith(".json") else f"{name}.json"
    path = PRESETS_DIR / n
    if path.exists():
        path.unlink()
        return True
    return False
