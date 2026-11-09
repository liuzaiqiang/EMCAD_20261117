"""Static checks for the six training families and checkpoint restoration."""

import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SWITCH = "USE_FOREGROUND_BACKGROUND_ATTENTION"
OPTION = "use_foreground_background_attention"


class ForegroundBackgroundEntrypointTests(unittest.TestCase):
    def test_shell_switch_is_near_caa_and_logged(self):
        for family in ("synapse", "acdc", "polyp", "isic", "busi", "cell"):
            with self.subTest(family=family):
                source = (ROOT / "sh" / ("start_train_" + family + ".sh")).read_text(encoding="utf-8")
                lines = source.splitlines()
                caa = next(i for i, line in enumerate(lines) if 'USE_CONTENT_AWARE_ANTIALIAS="' in line)
                gate = next(i for i, line in enumerate(lines) if SWITCH + '="' in line)
                self.assertLessEqual(abs(caa - gate), 2)
                self.assertIn(SWITCH + " must be 0 or 1", source)
                if family == "cell":
                    self.assertIn("start_train_polyp.sh", source)
                else:
                    self.assertIn('--' + OPTION + ' "${' + SWITCH + '}"', source)
                    self.assertIn("PARAM_NAMES=", source)
                    self.assertIn(SWITCH, source.split("PARAM_NAMES=", 1)[1])

    def test_python_routes(self):
        files = ("train_synapse.py", "train_acdc.py", "train_polyp.py",
                 "utils/acdc_utils.py", "utils/polyp_utils.py", "lib/networks.py",
                 "lib/decoders.py", "lib/experiment_paths.py", "test_synapse.py",
                 "test_acdc.py", "test_polyp.py", "test_isic.py", "test_busi.py")
        for filename in files:
            with self.subTest(filename=filename):
                source = (ROOT / filename).read_text(encoding="utf-8")
                ast.parse(source, filename=filename)
                self.assertIn(OPTION, source)
        for wrapper, base in (("train_isic.py", "train_polyp"),
                              ("train_busi.py", "train_polyp"),
                              ("test_isic.py", "test_polyp"),
                              ("test_busi.py", "test_polyp")):
            self.assertIn(base, (ROOT / wrapper).read_text(encoding="utf-8"))

    def test_old_and_new_config_for_common_test_paths(self):
        for filename, function_name in (("test_polyp.py", "restore_checkpoint_model_options"),
                                        ("test_acdc.py", "restore_checkpoint_config")):
            source = (ROOT / filename).read_text(encoding="utf-8")
            function = next(node for node in ast.parse(source).body
                            if isinstance(node, ast.FunctionDef) and node.name == function_name)
            namespace = {"Path": Path, "json": json, "sys": sys}
            exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
            restore = namespace[function_name]
            with tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(checkpoint=str(Path(directory) / "best.pth"),
                                       use_foreground_background_attention=0)
                old_argv = sys.argv
                try:
                    sys.argv = [filename]
                    for config, expected in (({}, 0), ({OPTION: 1}, 1)):
                        (Path(directory) / "config.json").write_text(json.dumps(config), encoding="utf-8")
                        restore(args)
                        self.assertEqual(getattr(args, OPTION), expected)
                    sys.argv = [filename, "--" + OPTION, "0"]
                    setattr(args, OPTION, 0)
                    with self.assertRaises(RuntimeError):
                        restore(args)
                finally:
                    sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
