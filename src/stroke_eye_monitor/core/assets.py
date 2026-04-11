from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path

import certifi

# Bundled float16 face landmarker (468 + iris topology, same lineage as legacy Face Mesh).
# https://developers.google.com/mediapipe/solutions/vision/face_landmarker
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_NAME = "face_landmarker.task"


def default_model_path() -> Path:
    # .../AIML/src/stroke_eye_monitor/core/assets.py -> repo root is parents[3]
    root = Path(__file__).resolve().parents[3]
    return root / "models" / MODEL_NAME


def ensure_face_landmarker_model(path: Path | None = None) -> Path:
    """Download the .task model once if missing."""
    p = path or default_model_path()
    if p.exists() and p.stat().st_size > 0:
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=120, context=ctx) as resp, tmp.open("wb") as out:
            out.write(resp.read())
        tmp.replace(p)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return p
