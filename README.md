# Webcam eye & gaze (research prototype)

Live **eye and iris** tracking from a webcam using **MediaPipe Face Landmarker**, with optional **calibrated on-screen gaze** (yellow dot on the preview).

**Important:** research / demo only — not a medical device, not security-grade eye tracking. Gaze is a **rough** mapping learned from your calibration, not a full appearance-based model like [GazeCapture](https://gazecapture.csail.mit.edu/).

## Quick start

```bash
cd /path/to/AIML
./scripts/bootstrap.sh          # creates .venv, installs deps + editable package
source .venv/bin/activate
stroke-eye-monitor              # or: python -m stroke_eye_monitor
```

First run downloads `models/face_landmarker.task` (requires network once). Quit with **q** or **Esc**.

### CLI options

| Flag | Meaning |
|------|---------|
| `--camera N` | Webcam index (default `0`) |
| `--width W` | Resize frame to width `W` before inference (default `640`; smaller often = faster) |
| `--no-mirror` | Do not flip image horizontally |
| `--full-mesh` | Draw full face tessellation (heavier than default eye-focused overlay) |
| `--model PATH` | Use this `face_landmarker.task` instead of auto-download |
| `--calibrate` | Run **12-point** gaze calibration; saves `--gaze-file`, then exits |
| `--gaze` | Load calibration and draw **estimated gaze** (cyan/yellow circle) on the video |
| `--gaze-file PATH` | Calibration JSON path (default `gaze_calibration.json`) |
| `--gaze-width` / `--gaze-height` | Calibration fullscreen canvas size (default `1280×720`) |
| `--gaze-samples` | Frames averaged per point after **SPACE** (default `45`) |
| `--gaze-alpha` | Gaze smoothing $0\ldots1$ (default `0.25`; higher = snappier) |

### Gaze workflow

1. **Calibrate** (once per setup / lighting / distance):

   ```bash
   stroke-eye-monitor --calibrate --gaze-file gaze_calibration.json
   ```

   For each of **12** dots: look at the **white circle**, hold still, press **SPACE** (or Enter). Press **q** to abort.

2. **Run with gaze overlay:**

   ```bash
   stroke-eye-monitor --gaze --gaze-file gaze_calibration.json
   ```

The dot is drawn in **webcam preview** coordinates (scaled from the calibration canvas). It is **not** the same as a commercial eye-tracker’s monitor POA unless you align setup and accept error.

## What it does (pipeline)

1. **Sensing:** Each frame is processed with **MediaPipe Face Landmarker**, which outputs **478** points: the usual **468** face landmarks plus **iris** points. Coordinates are normalized $[0,1]$ and scaled to pixels $(x\cdot W,\; y\cdot H)$.

2. **Overlay:** **OpenCV** draws connections from MediaPipe’s topology (eyes, irises; optionally full tessellation).

3. **HUD:** Shows FPS, inference time, smoothed **EAR** (left/right), **asymmetry** \(|L-R|\), and **iris offsets**.

4. **Optional gaze:** After calibration, an **affine map** predicts a point $(g_x,g_y)$ on the calibration canvas from the feature vector below, then maps it onto the preview for display.

## Metrics (math-first)

### Coordinate system

MediaPipe returns **normalized** landmark coordinates $(x,y)\in[0,1]^2$. We convert to pixel coordinates:

$$
X = x\cdot W,\quad Y = y\cdot H
$$

Distances use Euclidean distance:

$$
d\big((X_1,Y_1),(X_2,Y_2)\big) = \sqrt{(X_1-X_2)^2 + (Y_1-Y_2)^2}
$$

### Eye Aspect Ratio (EAR)

Per eye, six landmark indices form points $p_0,\ldots,p_5$ in pixel space:

$$
\mathrm{EAR} = \frac{d(p_1,p_5) + d(p_2,p_4)}{2\,d(p_0,p_3)}
$$

- $d(p_0,p_3)$: **eye width** (corner-to-corner)
- $d(p_1,p_5)$ and $d(p_2,p_4)$: two **eye height** measurements

**Interpretation:** higher EAR ⇒ eye appears **more open**; lower EAR ⇒ **more closed** (blink/squint/pose/occlusion can all change it).

### EAR asymmetry

$$
\mathrm{asymmetry} = \left| \mathrm{EAR}_{\mathrm{left}} - \mathrm{EAR}_{\mathrm{right}} \right|
$$

Shown in the HUD as **`|d|`**. This measures **left–right difference in eye openness**, not a medical condition.

### Iris offset (normalized)

Let $c=(c_x,c_y)$ be the **iris center** (mean of the iris landmark points).

Let the eye corners be $\mathrm{outer}$ and $\mathrm{inner}$. Define:

$$
w_{\mathrm{eye}} = d(\mathrm{outer},\mathrm{inner}),\quad
m = \frac{\mathrm{outer}+\mathrm{inner}}{2}
$$

The normalized iris offset is:

$$
n_x = \frac{c_x - m_x}{w_{\mathrm{eye}}/2}, \quad
n_y = \frac{c_y - m_y}{w_{\mathrm{eye}}/2}
$$

**Interpretation:** rough “where the iris sits inside the eye” in the image. This is **not** calibrated **screen gaze** (see e.g. [GazeCapture](https://gazecapture.csail.mit.edu/)).

### Temporal smoothing

Displayed values use exponential smoothing (to reduce jitter):

$$
s_t = \alpha x_t + (1-\alpha)\, s_{t-1}
$$

with the same idea for the $(n_x, n_y)$ vector. This reduces jitter but adds a little lag.

### Calibrated gaze (affine regression)

**Feature vector** (iris offsets + face-pose translation from MediaPipe’s $4\times4$ transform + bias):

$$
\mathbf{f} = [\,L_{nx},\; L_{ny},\; R_{nx},\; R_{ny},\; t_x,\; t_y,\; t_z,\; 1\,]^\top
$$

where $(t_x,t_y,t_z)$ is the **translation column** of the face transform matrix (helps when your head moves a bit between calibration and use).

At each calibration dot at screen position $(s_x, s_y)$ (in **gaze canvas** pixels), we record the **mean** $\bar{\mathbf{f}}$ over several frames. Then we fit **separate** least-squares linear models:

$$
s_x \approx \mathbf{w}_x^\top \mathbf{f},\qquad
s_y \approx \mathbf{w}_y^\top \mathbf{f}
$$

So $\mathbf{w}_x,\mathbf{w}_y \in \mathbb{R}^8$ are ordinary least squares solutions (same spirit as a **tiny** linear head on top of hand-crafted eye features; GazeCapture instead learns a deep **image → gaze** map from large data).

At runtime we apply exponential smoothing to $(g_x,g_y)$, clamp to the canvas, then **scale** into the live preview resolution for drawing.

### If gaze feels wrong (especially on Mac / Retina)

- **Recalibrate** after code updates: older `gaze_calibration.json` files used **5** features; current builds use **8** — delete the old file or run `--calibrate` again.
- During calibration, the app prints **`Calibration canvas (OpenCV window): W x H`**. That is the **real** fullscreen size OpenCV sees (often fixes “I typed 2560×1664 but dots don’t match” on HiDPI).
- Keep your **head still** and **only press SPACE** when your eyes are on the white dot; try `--gaze-samples 90` for more averaging per point.
- Use **`--gaze-alpha 0.35–0.5`** if the dot is too sluggish; lower alpha if it jitters.

## Relation to cited ideas (high level)

- **MediaPipe Face Mesh / Face Landmarker:** Dense face + iris landmarks in real time; see [Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) and the legacy [Face Mesh](https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html) docs.
- **GazeCapture:** Targets **gaze on a display** from appearance; this repo only uses simple **iris-in-eye** geometry, not a trained gaze regressor.
- **SpeakFaster / AAC:** Concerns **faster communication** (e.g. LLM-assisted typing) for people using **eye-gaze** input; this project only supplies **camera → eye metrics**, not text entry or language models ([SpeakFaster overview](https://research.google/blog/speakfaster-revolutionizing-communication-for-people-with-severe-motor-impairments/)).

## Project layout

| Path | Role |
|------|------|
| `src/stroke_eye_monitor/app.py` | CLI, camera loop, HUD |
| `src/stroke_eye_monitor/detector.py` | Face Landmarker (VIDEO mode) |
| `src/stroke_eye_monitor/metrics.py` | EAR, asymmetry, iris offsets, smoothing |
| `src/stroke_eye_monitor/drawing.py` | OpenCV overlays |
| `src/stroke_eye_monitor/assets.py` | Model URL + download (`certifi` for HTTPS) |
| `src/stroke_eye_monitor/gaze_mapping.py` | Affine gaze model + JSON save/load |
| `src/stroke_eye_monitor/gaze_calibration.py` | 12-point calibration UI |
| `models/` | `face_landmarker.task` (downloaded; not committed) |
| `gaze_calibration.json` | Your gaze map (created by `--calibrate`; gitignored if you add it) |

## Requirements

- Python **3.10+**
- Dependencies: see `pyproject.toml` / `requirements.txt` (MediaPipe, OpenCV, NumPy, certifi).

## License / disclaimer

Use at your own risk for **non-clinical** experiments. **Do not** use this software to decide whether someone is having a stroke or other emergency; call emergency services and seek professional care when appropriate.
