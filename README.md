# Neurovox:

**Stroke-eye-monitor** is a webcam demo: **eye and iris** tracking with **MediaPipe**, optional **gaze calibration**, a **fullscreen gaze keyboard** (look + blink to type), and a **CSV collection** mode for later modelling.

**LLM** (`src/LLM/`) takes **both** the **chatbot's question** and the person's **short typed reply** together, then ranks **five possible full replies from the person** (what they might really be trying to say back), e.g. after fragments like `WDYM` or `IMFTK`.

> **Disclaimer:** Research / hackathon demo only — **not** a medical device, not clinical-grade or security-grade eye tracking, not a replacement for commercial eye trackers.

---

## Hackathon snapshot (for judges & teammates)

| Topic | Details |
|-------|---------|
| **Problem space** | People with limited motor control may use **gaze + blink** to type; short replies are hard for a generic chatbot to interpret. |
| **What runs today** | Live face mesh, optional **gaze on a calibrated canvas**, **26-key keyboard** with blink-to-select, **12-point calibration**, **random-dot data export**, **OpenAI-ranked candidate replies** from the person (question + short fragment together; CLI / API, not yet inside the live keyboard window). |
| **Demo idea** | (1) Run `--calibrate` once. (2) `--keyboard` to type a short answer. (3) Show `python -m LLM "<chatbot question>" "<typed reply>"` for top-5 ranked **full replies** they might intend. |
| **Not in scope / stretch** | Medical claims, dwell-to-confirm strip, TTS, deep learned gaze (e.g. GazeCapture-style), wiring LLM output into the OpenCV keyboard UI. |

---

## Download (Windows `.exe`)

Prebuilt **Windows** executables are published on **GitHub Releases** when a maintainer pushes a version tag (for example `v0.1.1`). See **[`RELEASE.md`](RELEASE.md)** for the exact release steps and a local PyInstaller command.

Run the downloaded `.exe` like the CLI below; the first launch may download MediaPipe model files. Configure **API keys** (OpenAI, etc.) via a `.env` file next to the executable or your working directory, as described in this README.

---

## Quick start

```bash
cd /path/to/AIML
./scripts/bootstrap.sh          # .venv + editable install
source .venv/bin/activate
stroke-eye-monitor              # or: python -m stroke_eye_monitor
```

First run downloads `models/face_landmarker.task` (needs network once). Quit live windows with **q** or **Esc**.

---

## Commands (what to run)

| Goal | Command |
|------|---------|
| Eye tracking + HUD | `stroke-eye-monitor` |
| Save gaze calibration | `stroke-eye-monitor --calibrate` |
| HUD + gaze numbers | `stroke-eye-monitor --gaze` |
| Gaze keyboard (needs calibration) | `stroke-eye-monitor --keyboard` |
| Export iris vs screen CSV | `stroke-eye-monitor --collect` |

On **macOS**, `--calibrate` and `--collect` try to match **primary screen resolution** so dots and CSV coordinates line up with fullscreen. If that fails, use `--gaze-width` / `--gaze-height`.

---

## CLI flags (cheat sheet)

| Flag | Meaning |
|------|---------|
| `--camera N` | Webcam index (default `0`) |
| `--width W` | Resize frame to width `W` before inference (default `640`) |
| `--no-mirror` | Do not mirror the camera preview horizontally |
| `--full-mesh` | Draw full face mesh (heavier) |
| `--model PATH` | Custom `face_landmarker.task` |
| `--calibrate` | 12-point calibration; writes `--gaze-file`; then exit |
| `--gaze-file PATH` | Calibration JSON (default `gaze_calibration.json`) |
| `--gaze-width` / `--gaze-height` | Fallback canvas if screen detect fails (default `1280` × `720`) |
| `--gaze-samples` | Frames per dot after **Space** (default `45`) |
| `--gaze-alpha` | Gaze smoothing, **0–1**, higher = snappier (default `0.25`) |
| `--gaze-ear-min` | Ignore gaze updates when either eye is too closed (default `0.17`) |
| `--gaze-ridge` | Ridge strength for the calibration fit (default `1e-2`) |
| `--gaze` | Load calibration; show gaze in HUD |
| `--keyboard` | Same as `--gaze` plus second window: fullscreen keyboard; **blink** types; **D** backspace |
| `--collect` | Random targets → CSV (`--collect-csv`, `--collect-points`) |

---

## Workflows (step by step)

### 1. Calibration (`--calibrate`)

1. `stroke-eye-monitor --calibrate`
2. For each dot: look at the **white circle**, hold still, press **Space** (or Enter).
3. **q** aborts.
4. Output: `gaze_calibration.json` (canvas size matches what was detected or your fallback flags).

Recalibrate if you move the laptop, change distance, or you see errors about **`feature_dim`** (see below).

### 2. Gaze HUD (`--gaze`)

Needs a calibration file:

```bash
stroke-eye-monitor --gaze --gaze-file gaze_calibration.json
```

Camera window: face overlay + HUD (EAR, iris offsets, smoothed gaze x/y in **calibration canvas** pixels). No gaze dot drawn on the video by default.

### 3. Gaze keyboard (`--keyboard`)

Needs calibration. Opens:

- **Camera** — same overlay + metrics as `--gaze`.
- **Fullscreen keyboard** — 26 letters in rows of **7, 7, 7, 5**. Look to highlight; **blink** to type. **D** deletes last character.

On quit, typed text prints to the terminal if non-empty.

### 4. Data collection (`--collect`)

Random on-screen dots; saves a CSV with:

`point_id`, `screen_x`, `screen_y`, `left_nx`, `left_ny`, `right_nx`, `right_ny`

---

## How the pipeline fits together (no math)

Rough map from “idea on a whiteboard” to this repo:

| Idea | In this repo? |
|------|----------------|
| Camera → eyes / iris → screen-style coordinates | **Yes** — MediaPipe + optional calibration. |
| On-screen keyboard | **Yes** — fullscreen grid; gaze picks the cell. |
| Commit letter with blink | **Yes**. |
| Long dwell to confirm each key | **No** — use **Esc** to quit; no separate confirm strip. |
| Learned gaze from huge datasets / full SpeakFaster stack | **Out of scope** — small personal linear map + hackathon LLM helper. |

**Runtime (short):**

1. **MediaPipe Face Landmarker** (video mode) gives face + iris landmarks on each frame.
2. **OpenCV** draws the eye / iris overlay (and optional full mesh).
3. **HUD** shows FPS, inference time, face lock, smoothed **EAR**, **iris offsets**, optional **gaze**.
4. **Gaze (optional):** an **8-number feature row** per frame — left and right iris offsets in the eye, three numbers from **head rotation** (Rodrigues vector from MediaPipe’s face transform), plus a constant bias. **Ridge regression** was fit at calibration time; at runtime we predict gaze x/y on the canvas, smooth, and clamp.
5. **Keyboard (optional):** gaze hits a cell; **BlinkDetector** uses EAR over time to fire a “select” on reopening after a blink.

**Calibration file (`gaze_calibration.json`):** stores canvas width/height, **`feature_dim`** (must be **8** for the current app), learned weights for x and y, and a small **`version`** tag. Old 5-feature files need a fresh **`--calibrate`**.

**Blink detection (keyboard):** average EAR across both eyes; blink = reopening after a short “eyes closed” run, with cooldown so one blink does not double-fire.

---

## Tips if gaze feels off

- Recalibrate after you move or change distance to the screen.
- Watch the **printed canvas size** during calibrate/collect; it should match the fullscreen window.
- More **`--gaze-samples`** = calmer dots; tune **`--gaze-alpha`** for smooth vs snappy.
- **`--gaze-ear-min`** drops half-closed frames for gaze updates (blinks still move the mesh).

---

## Inspiration & references

- **MediaPipe Face Landmarker:** [Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) (tasks API used here).
- **GazeCapture:** Large learned **appearance → gaze**; we use **geometry + quick personal calibration** instead.
- **SpeakFaster / AAC:** Broader communication stack; we focus on **eyes + optional gaze + keyboard**, plus a small **LLM** helper that ranks **candidate full replies** from the user given chatbot question + short fragment ([SpeakFaster blog](https://research.google/blog/speakfaster-revolutionizing-communication-for-people-with-severe-motor-impairments/)).

---

## Repo layout

| Path | Role |
|------|------|
| `src/stroke_eye_monitor/app.py` | CLI entry: calibrate / collect / live loop |
| `src/stroke_eye_monitor/cli_args.py` | Argument definitions |
| `src/stroke_eye_monitor/config.py` | Camera config, macOS screen size helper |
| `src/stroke_eye_monitor/core/` | Detector, metrics, gaze mapping, model download |
| `src/stroke_eye_monitor/ui/` | Drawing + gaze keyboard UI |
| `src/stroke_eye_monitor/pipeline/` | Live per-frame pipeline (metrics → gaze → keyboard hooks) |
| `src/stroke_eye_monitor/modes/` | Calibration + CSV collection flows |
| `src/stroke_eye_monitor/utils/` | Frame resize, OpenCV fullscreen sync, FPS, threaded capture |
| `src/LLM/` | `OpenAICompletion`: **question + short reply** → top-5 ranked **candidate full replies from the person** (`EchoBackend` for tests) |
| `models/` | `face_landmarker.task` (downloaded; not committed) |
| `gaze_calibration.json` | Your calibration (gitignored) |
| `gaze_data*.csv` | Collection output (gitignored) |

---

## Requirements

- Python **3.10+**
- Install from **`pyproject.toml`** / **`requirements.txt`** (MediaPipe, OpenCV, NumPy, certifi, OpenAI SDK, python-dotenv).

### LLM (OpenAI)

- Set **`OPENAI_API_KEY`** in repo-root **`.env`** (gitignored) or in the environment.
- Optional: **`OPENAI_MODEL`** (default `gpt-4o-mini`), **`OPENAI_BASE_URL`** for compatible gateways.
- After `pip install -e .`:

  ```text
  PYTHONPATH=src python -m LLM "Chatbot question here" "SHORTREPLY"
  ```

  Prints five ranked **candidate replies from the person** (joint use of question + fragment). Same via `OpenAICompletion.complete_ranked(question=..., reply=..., k=5)`. Not yet hooked into the live keyboard window.

---

## License

**MIT** — see [LICENSE](LICENSE). You may replace the copyright line with your team or school name.
