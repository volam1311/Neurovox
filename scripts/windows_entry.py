"""
Console entry for PyInstaller Windows builds.

Install the project first (``pip install -e .``), then point PyInstaller at this file.
"""

from __future__ import annotations

from stroke_eye_monitor.app import main

if __name__ == "__main__":
    main()
