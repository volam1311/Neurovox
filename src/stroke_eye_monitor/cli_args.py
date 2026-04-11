from __future__ import annotations

import argparse


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Live eye & iris tracking via MediaPipe Face Landmarker. "
            "Optional calibrated on-screen gaze (see --calibrate / --gaze)."
        )
    )
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    p.add_argument(
        "--width",
        type=int,
        default=640,
        help="Resize frame to this width before inference (height keeps aspect)",
    )
    p.add_argument("--no-mirror", action="store_true", help="Disable horizontal flip for display")
    p.add_argument(
        "--full-mesh",
        action="store_true",
        help="Draw full face tessellation (prettier, higher CPU draw cost)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to face_landmarker.task (default: download to ./models/)",
    )
    p.add_argument(
        "--calibrate",
        "--calibration",
        action="store_true",
        dest="calibrate",
        help="Run gaze calibration (saves --gaze-file), then exit (--calibration is the same)",
    )
    p.add_argument(
        "--gaze",
        action="store_true",
        help="Load calibration and draw estimated gaze point on the video",
    )
    p.add_argument(
        "--gaze-file",
        type=str,
        default="gaze_calibration.json",
        help="Path to gaze calibration JSON (from --calibrate)",
    )
    p.add_argument("--gaze-width", type=int, default=1280, help="Calibration canvas width (pixels)")
    p.add_argument("--gaze-height", type=int, default=720, help="Calibration canvas height (pixels)")
    p.add_argument(
        "--gaze-samples",
        type=int,
        default=10,
        help=(
            "Valid gaze frames collected per dot after SPACE (default 10 ≈ one second "
            "at ~10 usable frames/s); raise for less noise"
        ),
    )
    p.add_argument(
        "--gaze-alpha",
        type=float,
        default=0.25,
        help="Exponential smoothing for gaze position (0..1, higher = snappier)",
    )
    p.add_argument(
        "--gaze-keyboard-median",
        type=int,
        default=3,
        help=(
            "With --keyboard: median of the last N smoothed gaze points for key hit "
            "(>=3 recommended; 0–2 disables extra median for snappier moves)"
        ),
    )
    p.add_argument(
        "--gaze-ear-min",
        type=float,
        default=0.17,
        help="Reject gaze updates when either eye EAR is below this (blink filter)",
    )
    p.add_argument(
        "--gaze-ridge",
        type=float,
        default=1e-2,
        help="Ridge regularization used during calibration fit (only affects --calibrate)",
    )
    p.add_argument(
        "--gaze-model",
        type=str,
        default="auto",
        choices=("auto", "ridge", "poly", "svr", "gbr", "xgboost", "rf"),
        help=(
            "Regression model for --calibrate. 'auto' evaluates all candidates and "
            "selects the best by LOO-CV without prompting. "
            "Choices: ridge, poly, svr, gbr, xgboost, rf. Default: auto"
        ),
    )
    p.add_argument(
        "--gaze-cal-grid",
        action="store_true",
        help="Use fixed 4x3 calibration grid instead of random dot positions (--calibrate)",
    )
    p.add_argument(
        "--gaze-cal-points",
        type=int,
        default=36,
        help="Number of random calibration dots for --calibrate (default 36, min 3)",
    )
    p.add_argument(
        "--gaze-cal-seed",
        type=int,
        default=None,
        help="RNG seed for random calibration layout (default: different each run)",
    )
    p.add_argument(
        "--keyboard",
        action="store_true",
        help="Show gaze keyboard overlay (implies --gaze); blink to select letters",
    )
    p.add_argument(
        "--collect",
        action="store_true",
        help="Show random dots, capture iris coordinates at each, save to CSV, then exit",
    )
    p.add_argument(
        "--collect-csv",
        type=str,
        default="gaze_data.csv",
        help="Output CSV path for --collect (default: gaze_data.csv)",
    )
    p.add_argument(
        "--collect-points",
        type=int,
        default=36,
        help="Number of random points to collect (default: 36)",
    )
    p.add_argument(
        "--collect-samples",
        type=int,
        default=45,
        help="Valid frames per dot for --collect only (default 45; see --gaze-samples for --calibrate)",
    )
    return p.parse_args(argv)
