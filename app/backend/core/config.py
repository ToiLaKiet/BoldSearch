"""
Shared configuration and constants for the BoldSearcher backend.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# ── Server ───────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 5001
DEBUG = True

# ── API ──────────────────────────────────────────────────────────────
API_PREFIX = "/api"
SYSTEM_NAME = "BoldSearcher"
