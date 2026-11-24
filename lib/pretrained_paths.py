"""Shared PVT weight directory four levels above the project root."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SHARED_PRETRAINED_DIR = str(
    (PROJECT_DIR / ".." / ".." / ".." / ".." / "pretrained_pth" / "pvt").resolve()
)
