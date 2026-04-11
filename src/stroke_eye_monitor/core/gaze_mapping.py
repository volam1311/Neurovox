from __future__ import annotations

import base64
import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

_MODEL_CHOICES = ("auto", "ridge", "poly", "svr", "gbr", "xgboost", "rf")

# Above this many training rows, use shuffled K-fold instead of leave-one-out (speed).
_GAZE_CV_LOO_MAX_N: int = 80


class _TargetScaledRegressor(BaseEstimator, RegressorMixin):
    """Wraps a regressor and applies StandardScaler to the targets (Y) during fit/predict.

    SVR and GBR perform much better when targets are on a similar scale to features.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: np.ndarray, Y: np.ndarray) -> _TargetScaledRegressor:
        self._y_scaler = StandardScaler()
        Y_scaled = self._y_scaler.fit_transform(Y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, Y_scaled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y_scaled = self.estimator_.predict(X)
        if Y_scaled.ndim == 1:
            Y_scaled = Y_scaled.reshape(-1, 1)
        return self._y_scaler.inverse_transform(Y_scaled)


def _build_candidates(
    ridge_lambda: float,
    n_samples: int,
) -> dict[str, Pipeline]:
    """Candidate pipelines tuned for small-sample gaze calibration (typically 12-100 points)."""
    alpha = max(ridge_lambda, 1e-6)
    poly_alpha = max(alpha, 0.1)
    svr_C = 10.0 if n_samples < 60 else 50.0
    gbr_n_est = min(100, max(30, n_samples * 2))
    return {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=alpha)),
        ]),
        "poly": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(
                degree=2, interaction_only=True, include_bias=False,
            )),
            ("reg", Ridge(alpha=poly_alpha)),
        ]),
        "svr": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", _TargetScaledRegressor(
                MultiOutputRegressor(
                    SVR(kernel="rbf", C=svr_C, epsilon=0.05, gamma="scale"),
                ),
            )),
        ]),
        "gbr": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", _TargetScaledRegressor(
                MultiOutputRegressor(
                    GradientBoostingRegressor(
                        n_estimators=gbr_n_est,
                        max_depth=2,
                        learning_rate=0.05,
                        subsample=0.8,
                        min_samples_leaf=max(2, n_samples // 10),
                    ),
                ),
            )),
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", _TargetScaledRegressor(
                MultiOutputRegressor(
                    XGBRegressor(
                        n_estimators=min(200, max(50, n_samples * 3)),
                        max_depth=4,
                        learning_rate=0.08,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        verbosity=0,
                    ),
                ),
            )),
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", _TargetScaledRegressor(
                MultiOutputRegressor(
                    RandomForestRegressor(
                        n_estimators=min(300, max(100, n_samples * 5)),
                        max_depth=6,
                        min_samples_leaf=max(2, n_samples // 10),
                        max_features="sqrt",
                        random_state=42,
                    ),
                ),
            )),
        ]),
    }


def _loo_cv_error(pipeline: Pipeline, F: np.ndarray, Y: np.ndarray) -> float:
    """Mean Euclidean pixel error via leave-one-out cross-validation."""
    loo = LeaveOneOut()
    errors: list[float] = []
    for train_idx, test_idx in loo.split(F):
        pipe = pickle.loads(pickle.dumps(pipeline))
        pipe.fit(F[train_idx], Y[train_idx])
        pred = pipe.predict(F[test_idx])
        diff = pred - Y[test_idx]
        errors.append(float(np.sqrt(np.sum(diff ** 2, axis=1)).mean()))
    return float(np.mean(errors))


def _gaze_cv_mean_error(pipeline: Pipeline, F: np.ndarray, Y: np.ndarray) -> float:
    """Mean test-set Euclidean pixel error: LOO when ``n <= _GAZE_CV_LOO_MAX_N``, else K-fold."""
    n = int(F.shape[0])
    if n <= _GAZE_CV_LOO_MAX_N:
        return _loo_cv_error(pipeline, F, Y)
    n_splits = min(10, max(3, n // 45))
    if n_splits >= n:
        n_splits = max(2, n // 2)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    errors: list[float] = []
    for train_idx, test_idx in kf.split(F):
        if len(train_idx) < 3:
            continue
        pipe = pickle.loads(pickle.dumps(pipeline))
        pipe.fit(F[train_idx], Y[train_idx])
        pred = pipe.predict(F[test_idx])
        diff = pred - Y[test_idx]
        errors.append(float(np.sqrt(np.sum(diff ** 2, axis=1)).mean()))
    if not errors:
        return _loo_cv_error(pipeline, F, Y)
    return float(np.mean(errors))


def _rf_tree_std_pixels(pipeline: Pipeline, X: np.ndarray) -> float | None:
    """Std dev of the (x, y) gaze prediction from RF tree disagreement, in pixels.

    Uses a subsample of trees for speed (live video). Returns ``None`` if the
    pipeline is not an RF + ``_TargetScaledRegressor`` + ``MultiOutputRegressor`` stack.
    """
    if not isinstance(pipeline, Pipeline):
        return None
    scaler = pipeline.named_steps.get("scaler")
    reg = pipeline.named_steps.get("reg")
    if scaler is None or reg is None:
        return None
    try:
        Xs = scaler.transform(X)
    except Exception:
        return None
    tsc = getattr(reg, "estimator_", None)
    if tsc is None:
        return None
    mor = getattr(tsc, "estimator_", None)
    if mor is None or not hasattr(mor, "estimators_"):
        return None
    y_scaler = getattr(tsc, "_y_scaler", None)
    if y_scaler is None or not hasattr(y_scaler, "scale_"):
        return None
    scale = np.asarray(y_scaler.scale_, dtype=np.float64).ravel()
    if scale.size < 2:
        return None
    ests = mor.estimators_
    if len(ests) < 2:
        return None
    rf_x, rf_y = ests[0], ests[1]
    if not hasattr(rf_x, "estimators_") or not hasattr(rf_y, "estimators_"):
        return None
    n_tx = len(rf_x.estimators_)
    n_ty = len(rf_y.estimators_)
    if n_tx < 2 or n_ty < 2:
        return None
    max_trees = 36
    ix = np.unique(np.linspace(0, n_tx - 1, num=min(max_trees, n_tx), dtype=np.int64))
    iy = np.unique(np.linspace(0, n_ty - 1, num=min(max_trees, n_ty), dtype=np.int64))
    txs = np.stack([rf_x.estimators_[int(i)].predict(Xs) for i in ix], axis=0)
    tys = np.stack([rf_y.estimators_[int(i)].predict(Xs) for i in iy], axis=0)
    var_sx = float(np.var(txs, axis=0).ravel()[0])
    var_sy = float(np.var(tys, axis=0).ravel()[0])
    var_px = (scale[0] ** 2) * var_sx + (scale[1] ** 2) * var_sy
    return float(np.sqrt(max(0.0, var_px)))


@dataclass
class GazeCalibration:
    """Gaze mapper: feature vector -> screen (gaze canvas) coordinates.

    Version 6 (legacy): affine map via ``coeff_x`` / ``coeff_y``.
    Version 7+: sklearn pipeline stored as pickle blob.
    """

    gaze_width: int
    gaze_height: int
    feature_dim: int = 8
    model_type: str = "ridge"

    coeff_x: list[float] = field(default_factory=list)
    coeff_y: list[float] = field(default_factory=list)

    #: Mean cross-validated Euclidean error (px); LOO if few rows, K-fold if many.
    loo_cv_px: float | None = None

    _pipeline: Any = field(default=None, repr=False)

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        x, y, _ = self.predict_with_uncertainty(features)
        return x, y

    def predict_with_uncertainty(self, features: np.ndarray) -> tuple[float, float, float]:
        """Return gaze (x, y) and a combined uncertainty **sigma** in pixels.

        ``sigma`` blends the calibration CV error with RF tree disagreement when
        applicable; otherwise it falls back to that CV error or a canvas-based default.
        """
        f = np.asarray(features, dtype=np.float64).reshape(1, -1)
        if f.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {f.shape[1]}")
        if self._pipeline is not None:
            pred = self._pipeline.predict(f)
            x, y = float(pred[0, 0]), float(pred[0, 1])
            ens_std: float | None = None
            if self.model_type == "rf":
                ens_std = _rf_tree_std_pixels(self._pipeline, f)
        else:
            fv = f.reshape(-1)
            x = float(np.dot(self.coeff_x, fv))
            y = float(np.dot(self.coeff_y, fv))
            ens_std = None

        base = self.loo_cv_px
        if base is None or base <= 0:
            base = 0.18 * float(np.hypot(self.gaze_width, self.gaze_height))
        if ens_std is not None and ens_std > 0:
            sigma = float(np.hypot(base, ens_std))
        else:
            sigma = float(base)
        return x, y, sigma

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return (
            float(np.clip(x, 0, self.gaze_width - 1)),
            float(np.clip(y, 0, self.gaze_height - 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        if self._pipeline is not None:
            blob = base64.b64encode(pickle.dumps(self._pipeline)).decode("ascii")
            d: dict[str, Any] = {
                "version": 7,
                "gaze_width": self.gaze_width,
                "gaze_height": self.gaze_height,
                "feature_dim": self.feature_dim,
                "model_type": self.model_type,
                "model_blob": blob,
            }
            if self.loo_cv_px is not None:
                d["loo_cv_px"] = float(self.loo_cv_px)
            return d
        return {
            "version": 6,
            "gaze_width": self.gaze_width,
            "gaze_height": self.gaze_height,
            "coeff_x": self.coeff_x,
            "coeff_y": self.coeff_y,
            "feature_dim": self.feature_dim,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GazeCalibration:
        version = int(d.get("version", 6))
        gw = int(d["gaze_width"])
        gh = int(d["gaze_height"])
        fd = int(d.get("feature_dim", 8))

        if version >= 7 and "model_blob" in d:
            blob = base64.b64decode(d["model_blob"])
            pipeline = pickle.loads(blob)  # noqa: S301
            loo = d.get("loo_cv_px")
            cal = cls(
                gaze_width=gw,
                gaze_height=gh,
                feature_dim=fd,
                model_type=str(d.get("model_type", "unknown")),
                loo_cv_px=float(loo) if loo is not None else None,
            )
            cal._pipeline = pipeline
            return cal

        return cls(
            gaze_width=gw,
            gaze_height=gh,
            coeff_x=list(map(float, d["coeff_x"])),
            coeff_y=list(map(float, d["coeff_y"])),
            feature_dim=fd,
            model_type="ridge",
            loo_cv_px=None,
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
    """Ridge-regularized affine map (legacy v6 format). Used for partial preview during calibration."""
    if len(feature_rows) != len(screen_xy) or len(feature_rows) < 3:
        raise ValueError("Need at least 3 calibration points with valid features.")
    F = np.stack([np.asarray(r, dtype=np.float64).reshape(-1) for r in feature_rows], axis=0)
    sx = np.array([p[0] for p in screen_xy], dtype=np.float64)
    sy = np.array([p[1] for p in screen_xy], dtype=np.float64)
    d = int(F.shape[1])
    lam = float(max(ridge_lambda, 0.0))
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
        model_type="ridge",
    )


def fit_gaze_model(
    feature_rows: list[np.ndarray],
    screen_xy: list[tuple[float, float]],
    gaze_width: int,
    gaze_height: int,
    *,
    ridge_lambda: float = 1e-2,
    model: str = "auto",
) -> GazeCalibration:
    """Fit the best gaze mapping model via cross-validation (or a forced ``model``).

    Uses leave-one-out when there are few training rows; shuffled K-fold when there
    are many (e.g. every captured frame per calibration dot).

    Returns a version-7 ``GazeCalibration`` backed by an sklearn pipeline.
    """
    if len(feature_rows) != len(screen_xy) or len(feature_rows) < 3:
        raise ValueError("Need at least 3 calibration points with valid features.")

    F = np.stack([np.asarray(r, dtype=np.float64).reshape(-1) for r in feature_rows], axis=0)
    Y = np.array(screen_xy, dtype=np.float64)

    candidates = _build_candidates(ridge_lambda, n_samples=len(feature_rows))

    if model != "auto":
        if model not in candidates:
            raise ValueError(f"Unknown model '{model}'. Choose from: {list(candidates)}")
        chosen_name = model
        chosen_pipe = candidates[model]
        chosen_pipe.fit(F, Y)
        err = _gaze_cv_mean_error(candidates[model], F, Y)
        print(f"Gaze model: {chosen_name} (forced), CV mean error: {err:.1f} px", flush=True)
    else:
        results: list[tuple[str, float, Pipeline]] = []
        for name, pipe in candidates.items():
            try:
                err = _gaze_cv_mean_error(pipe, F, Y)
                results.append((name, err, pipe))
                print(f"  {name:>6s}: CV mean error = {err:.1f} px", flush=True)
            except Exception as exc:
                print(
                    f"  {name:>6s}: failed ({exc})",
                    file=sys.stderr,
                    flush=True,
                )
        if not results:
            raise RuntimeError("All model candidates failed during cross-validation.")
        results.sort(key=lambda t: t[1])
        chosen_name, best_err, _ = results[0]
        print(
            f"Best model: {chosen_name} ({best_err:.1f} px CV mean error)",
            flush=True,
        )
        chosen_pipe = candidates[chosen_name]
        chosen_pipe.fit(F, Y)
        err = float(best_err)

    cal = GazeCalibration(
        gaze_width=gaze_width,
        gaze_height=gaze_height,
        feature_dim=int(F.shape[1]),
        model_type=chosen_name,
        loo_cv_px=float(err),
    )
    cal._pipeline = chosen_pipe
    return cal
