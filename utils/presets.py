from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

PRESETS_DIR = Path("presets")


def save_preset(name: str, params: dict) -> Path:
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
    n = name if name.endswith(".json") else f"{name}.json"
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
    if not PRESETS_DIR.exists():
        return []
    return sorted(p.stem for p in PRESETS_DIR.glob("*.json"))


def delete_preset(name: str) -> bool:
    n = name if name.endswith(".json") else f"{name}.json"
    path = PRESETS_DIR / n
    if path.exists():
        path.unlink()
        return True
    return False
