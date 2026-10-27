"""Streamlit Community Cloud entry point for the existing application."""

from pathlib import Path
import os
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.environ["SINDHI_PROVIDER_MODE"] = "cloudflare"

import app  # noqa: E402,F401
