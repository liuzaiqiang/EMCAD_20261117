"""Keep each evaluation's artifacts with its selected checkpoint."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4


def checkpoint_output_dir(checkpoint, label, output_dir=None):
    checkpoint_dir = Path(checkpoint).resolve().parent
    if output_dir:
        result_dir = Path(output_dir).resolve()
    else:
        run_id = "{}_{}_{}".format(
            label, datetime.now().strftime("%Y-%m-%d_%H%M%S"), uuid4().hex[:12]
        )
        result_dir = checkpoint_dir / "test_results" / run_id / "outputs"
    try:
        result_dir.relative_to(checkpoint_dir)
    except ValueError as exc:
        raise ValueError("Test output must be inside checkpoint directory: {}".format(result_dir)) from exc
    return result_dir


def prepare_test_output(checkpoint, label, output_dir=None, output_csv=None,
                        allow_prepared=False):
    checkpoint_dir = Path(checkpoint).resolve().parent
    result_dir = checkpoint_output_dir(checkpoint, label, output_dir)
    # An existing output directory could silently replace an earlier prediction.
    if result_dir.exists():
        allowed = {"data_split_manifest.csv", "data_split_summary.json"}
        if (not allow_prepared or not result_dir.is_dir() or
                not (result_dir / "data_split_manifest.csv").is_file() or
                any(p.name not in allowed for p in result_dir.iterdir())):
            raise FileExistsError("Test output directory already contains results: {}".format(result_dir))
    else:
        result_dir.mkdir(parents=True, exist_ok=False)
    csv_path = Path(output_csv).resolve() if output_csv else result_dir.parent / "test_metrics.csv"
    try:
        csv_path.relative_to(checkpoint_dir)
    except ValueError as exc:
        raise ValueError("Test CSV must be inside checkpoint directory: {}".format(csv_path)) from exc
    if csv_path.exists():
        raise FileExistsError("Test CSV already exists: {}".format(csv_path))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    return str(result_dir), str(csv_path)
