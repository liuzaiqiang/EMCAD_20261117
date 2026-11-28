"""Default locations for assets shared across project checkouts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_ROOT = PROJECT_ROOT.joinpath("..", "..", "..", "..").resolve()
DEFAULT_DATA_DIR = SHARED_ROOT / "data"
