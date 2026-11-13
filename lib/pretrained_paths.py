"""Default shared location for PVT encoder weights.

The path is anchored to the repository, so it does not depend on the
process working directory. An explicit pretrained_dir still overrides it.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PVT_DIR = str(
    PROJECT_ROOT.joinpath("..", "..", "..", "..", "pretrained_pth", "pvt").resolve()
)
