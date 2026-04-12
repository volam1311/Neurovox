# Releasing (GitHub + Windows .exe)

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

4. Open **Actions** → **Release (Windows exe)** → confirm the run succeeded.
5. Open **Releases** on GitHub: the workflow creates a release for that tag and attaches **`NeurovoxStrokeEyeMonitor-v0.1.1.exe`** (name includes the tag).

## Manual Windows build (local)

From the repo root, with Python 3.10+:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel
pip install -e .
pip install "pyinstaller>=6.0"
pyinstaller --noconfirm --clean --onefile --name NeurovoxStrokeEyeMonitor-local `
  --collect-all mediapipe --collect-all cv2 --collect-all sklearn --collect-all xgboost `
  --hidden-import=certifi --hidden-import=dotenv --hidden-import=sounddevice `
  --hidden-import=openai --hidden-import=pandas `
  --hidden-import=LLM --hidden-import=LLM.openai_backend --hidden-import=LLM.completion `
  --hidden-import=LLM.env --hidden-import=LLM.audio_platform --hidden-import=LLM.stt_whisper `
  scripts\windows_entry.py
```

Output: `dist\NeurovoxStrokeEyeMonitor-local.exe`.

## Notes for end users

- First run may download MediaPipe model files; **OpenAI** / **.env** usage needs API keys as in the main README.
- The binary is large (hundreds of MB) because it bundles Python, OpenCV, MediaPipe, and ML libs.
- **Windows** may show a SmartScreen warning for unsigned executables; users can “Run anyway” or you can add code signing later.
