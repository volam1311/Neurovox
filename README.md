# Webcam eye & gaze (research demo)

Live **eye and iris** tracking from a webcam using **MediaPipe Face Landmarker**, with optional **per-user gaze calibration**, a **fullscreen gaze keyboard** (blink to type), and a **data-collection** mode for exporting iris features vs. on-screen targets to CSV.

**Important:** This is a **research / demo** project — not a medical device, not security-grade eye tracking, and not a replacement for commercial eye trackers. Gaze uses a **small linear model** on hand-crafted iris features, not a deep appearance-based model like [GazeCapture](https://gazecapture.csail.mit.edu/).

## Quick start

```bash
cd /path/to/AIML
./scripts/bootstrap.sh          # creates .venv, installs deps + editable package
source .venv/bin/activate
stroke-eye-monitor              # or: python -m stroke_eye_monitor
```

The first run downloads `models/face_landmarker.task` (network once). Quit the live window with **q** or **Esc**.

## What you can run

| Goal | Command |
|------|---------|
| Eye tracking only (HUD) | `stroke-eye-monitor` |
| Calibrate gaze (save map) | `stroke-eye-monitor --calibrate` |
| See gaze coordinates in HUD | `stroke-eye-monitor --gaze` |
| Gaze keyboard (after calibrate) | `stroke-eye-monitor --keyboard` |
| Export calibration-style data to CSV | `stroke-eye-monitor --collect` |

On **macOS**, `--calibrate` and `--collect` try to detect your **primary display resolution** (e.g. Retina) so fixation dots and CSV `screen_x` / `screen_y` match a fullscreen canvas. If detection fails, they fall back to `--gaze-width` / `--gaze-height`.

## CLI reference

| Flag | Meaning |
|------|---------|
| `--camera N` | Webcam index (default `0`) |
| `--width W` | Resize frame to width `W` before inference (default `640`) |
| `--no-mirror` | Do not flip the **camera** preview horizontally |
| `--full-mesh` | Draw full face tessellation (heavier than default eye/iris overlay) |
| `--model PATH` | Use this `face_landmarker.task` instead of auto-download |
| **Calibration** | |
| `--calibrate` | 12-point fullscreen calibration; writes `--gaze-file`, then exits |
| `--gaze-file PATH` | Calibration JSON (default `gaze_calibration.json`) |
| `--gaze-width` / `--gaze-height` | Fallback canvas size if screen detection fails (defaults `1280×720`) |
| `--gaze-samples` | Frames collected per dot after **Space** (default `45`) |
| `--gaze-alpha` | Smoothing for gaze $[0,1]$, higher = snappier (default `0.25`) |
| `--gaze-ear-min` | Skip frames where either eye EAR is below this (blink / low quality; default `0.17`) |
| `--gaze-ridge` | Ridge $\lambda$ for the calibration fit (default `1e-2`) |
| **Runtime gaze** | |
| `--gaze` | Load calibration; compute gaze and show **HUD** lines (iris offsets, gaze x/y) |
| **Keyboard** | |
| `--keyboard` | Implies `--gaze`. **Second window**: fullscreen keyboard; **blink** to type the highlighted letter. **D** = backspace. |
| **Data collection** | |
| `--collect` | Random dots + capture; saves CSV via `--collect-csv`, then exits |
| `--collect-csv PATH` | Output CSV (default `gaze_data.csv`) |
| `--collect-points N` | Number of random points (default `36`) |

## Workflows

### 1. Calibration (`--calibrate`)

1. Run: `stroke-eye-monitor --calibrate`
2. For each dot: look at the **white circle**, hold still, press **Space** (or Enter).
3. Press **q** to abort.
4. Output: `gaze_calibration.json` (dimensions match the detected or fallback canvas).

Recalibrate after moving the laptop, changing distance, or if the JSON `version` / `feature_dim` no longer matches the app (see **Metrics** below).

### 2. Gaze only (`--gaze`)

Requires a calibration file from step 1:

```bash
stroke-eye-monitor --gaze --gaze-file gaze_calibration.json
```

The **camera** window shows the face overlay and a HUD with EAR, iris offsets $(n_x, n_y)$, and smoothed **gaze** $(g_x, g_y)$ in **calibration canvas** coordinates. There is **no** on-video gaze dot by default.

### 3. Gaze keyboard (`--keyboard`)

Requires calibration. Opens:

- **Camera window** — face + metrics (same as `--gaze`).
- **Fullscreen keyboard** — 26 letters (rows $7{+}7{+}7{+}5$). **Look** at a cell to highlight it; **blink** to append that letter. **D** deletes the last character.

Typed text is printed when you quit if non-empty.

### 4. Data collection (`--collect`)

For analysis / modelling (e.g. a data scientist): random on-screen targets, median iris offsets per target, CSV columns:

`point_id`, `screen_x`, `screen_y`, `left_nx`, `left_ny`, `right_nx`, `right_ny`

## Pipeline context (your diagram)

Rough alignment with a multi-phase AAC-style pipeline:

| Phase (concept) | This repo |
|-----------------|-----------|
| Capture (camera → iris → screen coords) | **Yes** — MediaPipe + optional calibration → $(g_x, g_y)$. |
| Keyboard UI | **Yes** — fullscreen grid; highlight from gaze. |
| “Type initials” (compose letters) | **Yes** — **blink** commits the focused letter (not 3 s dwell per key). |
| “Confirm / done” (e.g. 3 s dwell) | **Not implemented** — you finish with **Esc**; no separate dwell-to-confirm strip. |
| Pre-train on GazeCapture / SpeakFaster LLM / TTS | **Out of scope** here. |

## What the app does (runtime pipeline)

1. **Sensing:** **MediaPipe Face Landmarker** (video mode) outputs **478** landmarks (468 face + 10 iris when iris refine is on). Landmarks are normalized $[0,1]$; metrics use pixel coordinates $(xW,\,yH)$ for the processed frame size.

2. **Drawing:** **OpenCV** draws eye/iris polylines from MediaPipe’s task API topology (`drawing.py`).

3. **HUD:** FPS, inference time, face lock, smoothed **EAR**, **asymmetry** $\lvert L - R \rvert$, **iris offsets**, and optional **gaze** text.

4. **Optional gaze:** A **ridge-regularized affine** map from a **5-D feature vector** (eyes-only) to $(g_x, g_y)$ on the calibration canvas, then exponential smoothing and clamping.

5. **Optional keyboard:** Same gaze point drives `hit_test` on the keyboard layout; **BlinkDetector** fires on reopening after a short EAR drop to call `select()`.

## Metrics (math)

### Normalized landmarks to pixels

$$
X = x\,W,\quad Y = y\,H
$$

Euclidean distance $d$ between points.

### Eye Aspect Ratio (EAR)

Six points per eye:

$$
\mathrm{EAR} = \frac{d(p_1,p_5) + d(p_2,p_4)}{2\,d(p_0,p_3)}
$$

Higher ⇒ more open; lower ⇒ blink / squint / noise.

### Iris offset (in-eye, normalized)

Iris center $c$, eye corners outer/inner, eye width $w_{\mathrm{eye}}$, midpoint $m$:

$$
n_x = \frac{c_x - m_x}{w_{\mathrm{eye}}/2},\quad
n_y = \frac{c_y - m_y}{w_{\mathrm{eye}}/2}
$$

Roughly in $[-1,1]$ when geometry is well-behaved; these are **image-plane** features, not raw monitor gaze until combined with calibration.

### Temporal smoothing

$$
s_t = \alpha x_t + (1-\alpha)\,s_{t-1}
$$

(and similarly for the 2D iris offset vector).

### Calibrated gaze (eyes-only, ridge regression)

**Feature vector** (length **5**):

$$
\mathbf{f} = [\,L_{nx},\; L_{ny},\; R_{nx},\; R_{ny},\; 1\,]^\top
$$

At each calibration dot $(s_x, s_y)$ on the gaze canvas, frames with both eyes above `--gaze-ear-min` contribute samples; the stored row is the **per-coordinate median** across frames (robust to spikes). Then two separate ridge fits:

$$
\min_{\mathbf{w}_x}\; \|\mathbf{F}\mathbf{w}_x - \mathbf{s}_x\|^2 + \lambda\|\mathbf{w}_x\|^2,\qquad
\min_{\mathbf{w}_y}\; \|\mathbf{F}\mathbf{w}_y - \mathbf{s}_y\|^2 + \lambda\|\mathbf{w}_y\|^2
$$

with $\lambda =$ `--gaze-ridge`. At runtime: $\hat{g}_x = \mathbf{w}_x^\top\mathbf{f}$, $\hat{g}_y = \mathbf{w}_y^\top\mathbf{f}$, then smoothing and clamp to the canvas stored in the JSON.

Calibration files include `version` and `feature_dim`; the app expects **`feature_dim == 5`** for this eyes-only model.

### Blink detection (keyboard)

`BlinkDetector` tracks **average EAR** of both eyes. A **blink** is detected on the **rising edge** after EAR was below a threshold for a few frames, with a short cooldown to avoid double triggers.

## Tips if gaze feels wrong

- **Recalibrate** after changing seat, screen angle, or zoom.
- Watch the printed **canvas size** during `--calibrate` / `--collect`; it should match the OpenCV fullscreen client area.
- Increase `--gaze-samples` for calmer dots; tune `--gaze-alpha` for smooth vs. responsive.
- **`--gaze-ear-min`** rejects half-closed frames during gaze updates (blinks still move landmarks).

## Relation to cited ideas

- **MediaPipe Face Landmarker:** [Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) (tasks API used here).
- **GazeCapture:** Learned **appearance → gaze** from large data; this repo uses **geometry + short personal calibration**, not that dataset or architecture.
- **SpeakFaster / AAC:** Broader **communication** stack (e.g. LLM completion from partial input); this repo stops at **eye metrics + optional gaze + keyboard** ([SpeakFaster blog](https://research.google/blog/speakfaster-revolutionizing-communication-for-people-with-severe-motor-impairments/)).

## Project layout

| Path | Role |
|------|------|
| `src/stroke_eye_monitor/app.py` | CLI, camera loop, gaze, keyboard window |
| `src/stroke_eye_monitor/detector.py` | Face Landmarker (VIDEO) |
| `src/stroke_eye_monitor/metrics.py` | EAR, iris offsets, gaze feature vector, smoothing, `BlinkDetector` |
| `src/stroke_eye_monitor/drawing.py` | OpenCV overlays |
| `src/stroke_eye_monitor/assets.py` | Model download (`certifi` for HTTPS) |
| `src/stroke_eye_monitor/gaze_mapping.py` | `GazeCalibration`, ridge `fit_affine_gaze` |
| `src/stroke_eye_monitor/gaze_calibration.py` | 12-point calibration UI |
| `src/stroke_eye_monitor/keyboard_overlay.py` | 26-key grid + drawing |
| `src/stroke_eye_monitor/data_collection.py` | `--collect` random targets → CSV |
| `src/stroke_eye_monitor/config.py` | `MonitorConfig`, `detect_screen_resolution()` (macOS) |
| `models/` | `face_landmarker.task` (downloaded; not committed) |
| `gaze_calibration.json` | Your calibration (gitignored in `.gitignore`) |
| `gaze_data*.csv` | Collection output (gitignored) |

## Requirements

- Python **3.10+**
- Dependencies: `pyproject.toml` / `requirements.txt` — MediaPipe, OpenCV, NumPy, certifi.

## License

Software is licensed under the **MIT License** — see [LICENSE](LICENSE). You may replace the copyright line in `LICENSE` with your name or organization.
