from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GazeCalibration:
    """Affine map from feature vector to screen (gaze canvas) coordinates."""

    gaze_width: int
    gaze_height: int
    coeff_x: list[float]
    coeff_y: list[float]
    feature_dim: int = 5

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """features length must match feature_dim (default 8: iris + face translation + bias)."""
        f = np.asarray(features, dtype=np.float64).reshape(-1)
        if f.shape[0] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {f.shape[0]}")
        wx = np.dot(self.coeff_x, f)
        wy = np.dot(self.coeff_y, f)
        return float(wx), float(wy)

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return (
            float(np.clip(x, 0, self.gaze_width - 1)),
            float(np.clip(y, 0, self.gaze_height - 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 3,
            "gaze_width": self.gaze_width,
            "gaze_height": self.gaze_height,
            "coeff_x": self.coeff_x,
            "coeff_y": self.coeff_y,
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GazeCalibration:
        fd = int(d.get("feature_dim", 5))
        return cls(
            gaze_width=int(d["gaze_width"]),
            gaze_height=int(d["gaze_height"]),
            coeff_x=list(map(float, d["coeff_x"])),
            coeff_y=list(map(float, d["coeff_y"])),
            feature_dim=fd,
        )

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> GazeCalibration:
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


def fit_affine_gaze(
    feature_rows: list[np.ndarray],
    screen_xy: list[tuple[float, float]],
    gaze_width: int,
    gaze_height: int,
    *,
    ridge_lambda: float = 1e-2,
) -> GazeCalibration:
    """Ridge-regularized affine map: screen ≈ W @ features.

    Ridge helps stabilize when features are noisy (common with iris landmarks).
    """
    if len(feature_rows) != len(screen_xy) or len(feature_rows) < 3:
        raise ValueError("Need at least 3 calibration points with valid features.")
    F = np.stack([np.asarray(r, dtype=np.float64).reshape(-1) for r in feature_rows], axis=0)
    sx = np.array([p[0] for p in screen_xy], dtype=np.float64)
    sy = np.array([p[1] for p in screen_xy], dtype=np.float64)
    d = int(F.shape[1])
    lam = float(max(ridge_lambda, 0.0))
    # Closed form ridge: (F^T F + λI)^-1 F^T y
    A = F.T @ F + lam * np.eye(d, dtype=np.float64)
    bx = F.T @ sx
    by = F.T @ sy
    wx = np.linalg.solve(A, bx)
    wy = np.linalg.solve(A, by)
    return GazeCalibration(
        gaze_width=gaze_width,
        gaze_height=gaze_height,
        coeff_x=wx.tolist(),
        coeff_y=wy.tolist(),
        feature_dim=int(F.shape[1]),
    )
