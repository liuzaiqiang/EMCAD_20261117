"""Static contracts for all six training families and five test programs."""

import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SHELL_FAMILIES = ("synapse", "acdc", "polyp", "isic", "busi", "cell")
PYTHON_TESTS = ("test_synapse.py", "test_acdc.py", "test_polyp.py",
                "test_isic.py", "test_busi.py")


class SkipFrequencyEntrypointTests(unittest.TestCase):
    def test_train_shell_switch_is_near_caa_and_logged(self):
        for family in SHELL_FAMILIES:
            with self.subTest(family=family):
                source = (ROOT / "sh" / ("start_train_" + family + ".sh")).read_text(encoding="utf-8")
                lines = source.splitlines()
                caa = next(i for i, line in enumerate(lines) if "USE_CONTENT_AWARE_ANTIALIAS=\"" in line)
                gate = next(i for i, line in enumerate(lines) if "USE_SKIP_FREQUENCY_GATE=\"" in line)
                self.assertLessEqual(abs(caa - gate), 2)
                self.assertIn("USE_SKIP_FREQUENCY_GATE must be 0 or 1", source)
                if family == "cell":
                    self.assertIn("start_train_polyp.sh", source)
                else:
                    self.assertIn('--use_skip_frequency_gate "${USE_SKIP_FREQUENCY_GATE}"', source)
                    self.assertIn("PARAM_NAMES=", source)
                    self.assertIn("USE_SKIP_FREQUENCY_GATE", source.split("PARAM_NAMES=", 1)[1])

    def test_python_model_and_checkpoint_restore_paths(self):
        files = ("train_synapse.py", "train_acdc.py", "train_polyp.py",
                 "utils/acdc_utils.py", "utils/polyp_utils.py", "lib/networks.py",
                 "lib/decoders.py", "lib/experiment_paths.py", *PYTHON_TESTS)
        for name in files:
            with self.subTest(file=name):
                source = (ROOT / name).read_text(encoding="utf-8")
                ast.parse(source, filename=name)
                self.assertIn("use_skip_frequency_gate", source)
        for name in PYTHON_TESTS:
            source = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("config", source)
        self.assertIn("train_polyp", (ROOT / "train_isic.py").read_text(encoding="utf-8"))
        self.assertIn("train_polyp", (ROOT / "train_busi.py").read_text(encoding="utf-8"))
        self.assertIn("test_polyp", (ROOT / "test_isic.py").read_text(encoding="utf-8"))
        self.assertIn("test_polyp", (ROOT / "test_busi.py").read_text(encoding="utf-8"))

    def test_old_and_new_checkpoint_config_for_shared_test_paths(self):
        for filename, function_name in (("test_polyp.py", "restore_checkpoint_model_options"),
                                        ("test_acdc.py", "restore_checkpoint_config")):
            source = (ROOT / filename).read_text(encoding="utf-8")
            function = next(node for node in ast.parse(source).body
                            if isinstance(node, ast.FunctionDef) and node.name == function_name)
            namespace = {"Path": Path, "json": json, "sys": sys}
            exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
            restore = namespace[function_name]
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "best.pth"
                args = SimpleNamespace(checkpoint=str(checkpoint), use_skip_frequency_gate=0)
                saved_argv = sys.argv
                try:
                    sys.argv = [filename]
                    for config, expected in (({}, 0), ({"use_skip_frequency_gate": 1}, 1)):
                        (Path(directory) / "config.json").write_text(json.dumps(config), encoding="utf-8")
                        restore(args)
                        self.assertEqual(args.use_skip_frequency_gate, expected)
                    sys.argv = [filename, "--use_skip_frequency_gate", "0"]
                    args.use_skip_frequency_gate = 0
                    with self.assertRaises(RuntimeError):
                        restore(args)
                finally:
                    sys.argv = saved_argv


if __name__ == "__main__":
    unittest.main()
