"""Streamlit Cloud entrypoint for GridPulse."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = REPO_ROOT / "frontend"

for candidate in (REPO_ROOT, FRONTEND_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from streamlit_app import main  # noqa: E402


if __name__ == "__main__":
    main()
