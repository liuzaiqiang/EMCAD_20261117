"""Shared default directory for PVT encoder weights, relative to the project root."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PVT_DIR = str(
    PROJECT_ROOT.joinpath("..", "..", "..", "..", "pretrained_pth", "pvt").resolve()
)
