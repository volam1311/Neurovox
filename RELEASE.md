# Releasing (GitHub + multi-platform binaries)

## One-time setup

1. Push this repo to **GitHub** (if it is not already remote).
2. In the repo on GitHub: **Settings → Actions → General → Workflow permissions** → enable **Read and write** so the release workflow can upload assets.

## Publish a version

1. Bump **`version`** in `pyproject.toml` (and optionally sync `README` / changelog).
2. Commit and push to `main` (or your default branch).
3. Create and push a **tag** whose name starts with `v`:

   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

4. Open **Actions** → **Release (multi-platform)** → wait for the matrix build (Windows, macOS, Linux) and the **release** job.
5. Open **Releases** on GitHub: the workflow creates a release for that tag and attaches three files, for example:
   - `NeurovoxStrokeEyeMonitor-v0.1.1-windows.exe`
   - `NeurovoxStrokeEyeMonitor-v0.1.1-macos` (executable; run in Terminal: `chmod +x` if needed)
   - `NeurovoxStrokeEyeMonitor-v0.1.1-linux` (executable)

## Manual local build (PyInstaller)

From the repo root, with Python 3.10+ and the project installed (`pip install -e .`):

```bash
pip install "pyinstaller>=6.0"
TAG=local
# Pick a slug: windows | macos | linux (name only; build on that OS)
SLUG=linux
pyinstaller --noconfirm --clean --onefile --name "NeurovoxStrokeEyeMonitor-${TAG}-${SLUG}" \
  --collect-all mediapipe --collect-all cv2 --collect-all sklearn --collect-all xgboost \
  --hidden-import=certifi --hidden-import=dotenv --hidden-import=sounddevice \
  --hidden-import=openai --hidden-import=pandas \
  --hidden-import=LLM --hidden-import=LLM.openai_backend --hidden-import=LLM.completion \
  --hidden-import=LLM.env --hidden-import=LLM.audio_platform --hidden-import=LLM.stt_whisper \
  scripts/pyinstaller_entry.py
```

On **Windows** (PowerShell), use the same flags with line continuation `` ` `` or a single line. Install **on Windows** to produce `.exe`.

## Notes for end users

- First run may download MediaPipe model files; **OpenAI** / **.env** usage needs API keys as in the main README.
- Bundles are large (hundreds of MB) because they include Python, OpenCV, MediaPipe, and ML libraries.
- **Windows**: SmartScreen may warn on unsigned executables.
- **macOS**: Gatekeeper may block unsigned binaries — **System Settings → Privacy & Security** → “Open anyway”, or run `xattr -cr` on the app (understand the security tradeoff).
- **Linux**: You may need `libgl` / GLib stack on minimal distros; Ubuntu/Debian users often have them already.
