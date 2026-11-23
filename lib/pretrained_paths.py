"""Default shared encoder-weight directory outside the project checkout."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
# Four levels above PROJECT_DIR, matching the data-root convention in launchers.
SHARED_PRETRAINED_DIR = str(
    (PROJECT_DIR / ".." / ".." / ".." / ".." / "pretrained_pth" / "pvt").resolve()
)
