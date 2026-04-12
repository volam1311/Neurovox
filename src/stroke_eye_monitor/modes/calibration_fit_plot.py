from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def plot_cal_fit_csv(csv_path: Path, *, out_path: Path | None = None) -> int:
    """Draw target vs prediction from ``*_cal_fit.csv`` or ``*_cal_repeats.csv`` (no camera)."""
    if not csv_path.is_file():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    required = {
        "point_index",
        "target_x",
        "target_y",
        "pred_x",
        "pred_y",
        "gaze_canvas_w",
        "gaze_canvas_h",
    }
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("CSV has no header row.", file=sys.stderr)
            return 1
        missing = required - set(reader.fieldnames)
        if missing:
            print(f"CSV missing columns: {sorted(missing)}", file=sys.stderr)
            return 1
        repeats_mode = "repeat_index" in reader.fieldnames
        for row in reader:
            rows.append(row)

    if not rows:
        print("CSV has no data rows.", file=sys.stderr)
        return 1

    w = int(float(rows[0]["gaze_canvas_w"]))
    h = int(float(rows[0]["gaze_canvas_h"]))
    if w < 32 or h < 32:
        print(f"Invalid canvas size in CSV: {w} x {h}", file=sys.stderr)
        return 1

    board = np.full((h, w, 3), (28, 28, 28), dtype=np.uint8)

    for row in rows:
        tx = int(round(float(row["target_x"])))
        ty = int(round(float(row["target_y"])))
        px = int(round(float(row["pred_x"])))
        py = int(round(float(row["pred_y"])))
        pi = str(row.get("point_index", "")).strip()
        if repeats_mode:
            ri = str(row.get("repeat_index", "")).strip()
            label = f"{pi}.{ri}" if ri else pi
        else:
            label = pi

        # BGR: cyan error segments (was (0,255,255) which renders yellow in OpenCV).
        cv2.line(board, (tx, ty), (px, py), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(board, (tx, ty), 16, (0, 220, 0), 3, cv2.LINE_AA)
        cv2.circle(board, (px, py), 11, (60, 120, 255), 2, cv2.LINE_AA)

        if repeats_mode:
            lx = px + 8
            ly = py - 10
        else:
            lx = min(tx, px) + 12
            ly = min(ty, py) - 6
        lx = max(4, min(lx, w - 72))
        ly = max(20, min(ly, h - 8))
        cv2.putText(
            board,
            label,
            (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

    bar_h = 44
    cv2.rectangle(board, (0, 0), (w, bar_h), (45, 45, 45), -1)
    if repeats_mode:
        hint = (
            "Repeats: green = target   orange = pred per capture   "
            "cyan = error   labels = point.repeat   Esc / q = quit"
        )
    else:
        hint = "Green = target (truth)   Orange = predicted   Cyan line = offset   Esc / q = quit"
    cv2.putText(
        board,
        hint,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    win = (
        "Calibration repeats — target vs predicted per capture"
        if repeats_mode
        else "Calibration fit — target vs predicted"
    )
    if out_path is not None:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(outp), board):
            print(f"Failed to write image: {outp}", file=sys.stderr)
            return 1
        print(f"Wrote calibration plot: {outp.resolve()}", flush=True)
        return 0

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1600), min(h, 900))
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except cv2.error:
        pass

    try:
        while True:
            cv2.imshow(win, board)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()

    return 0


def plot_cal_fit_cli(csv_path: str, out_path: str | None = None) -> int:
    return plot_cal_fit_csv(Path(csv_path), out_path=Path(out_path) if out_path else None)
