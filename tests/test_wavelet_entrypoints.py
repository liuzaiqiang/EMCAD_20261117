"""Exercise every dataset launcher without starting training or using real data.

Exact launcher copies run in an isolated temporary directory. Only Conda
activation and nohup are intercepted; normal shell checks/argument expansion
still execute. Empty fixtures satisfy path checks, not dataset validation.
Parser/config checks execute exact AST-extracted source without importing data
loaders, so they do not claim that missing optional dependencies are installed.
"""

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ("Synapse", "synapse", 224),
    ("ACDC", "acdc", 224),
    ("BKAI", "polyp", 352),
    ("ClinicDB", "polyp", 352),
    ("ColonDB", "polyp", 352),
    ("ETIS", "polyp", 352),
    ("Kvasir", "polyp", 352),
    ("ISIC2017", "isic", 352),
    ("ISIC2018", "isic", 352),
    ("BUSI", "busi", 256),
    ("DSB18", "cell", 256),
    ("EM", "cell", 256),
)


def source_tree(filename):
    return ast.parse((ROOT / filename).read_text(encoding="utf-8-sig"))


def load_function(filename, name, **namespace):
    function = next(node for node in source_tree(filename).body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


def parse_training_wrapper(filename, argv, args):
    tree = source_tree(filename)
    function = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                  and n.name == "_parse_args"))
    end = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "dataset_root" for t in n.targets))
    function.body = function.body[:end] + [ast.Return(value=ast.Name(id="args", ctx=ast.Load()))]
    ast.fix_missing_locations(function)
    namespace = dict(sys=sys, ALLOWED_DATASETS={"ISIC2017", "ISIC2018"},
                     _original_parse_args=lambda: args,
                     _option_was_given=load_function(filename, "_option_was_given", sys=sys))
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    with patch.object(sys, "argv", [filename] + argv):
        return namespace["_parse_args"]()


def parse_source(filename, argv):
    namespace = dict(argparse=argparse)
    tree = source_tree(filename)
    function = next((node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "parse_args"), None)
    if function is not None:
        parse = load_function(filename, "parse_args", **namespace)
        with patch.object(sys, "argv", [filename] + argv):
            return parse()
    # Synapse has a module-level parser. Stop immediately after parse_args.
    statements = []
    recording = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "parser"
                                                for t in node.targets):
            recording = True
        if recording:
            statements.append(node)
        if recording and isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "args" for t in node.targets):
            break
    with patch.object(sys, "argv", [filename] + argv):
        exec(compile(ast.Module(body=statements, type_ignores=[]), filename, "exec"), namespace)
    return namespace["args"]


class DatasetLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = shutil.which("bash")
        windows_bash = Path("D:/install/Git/Git/bin/bash.exe")
        if windows_bash.is_file():
            cls.bash = str(windows_bash)
        if not cls.bash:
            raise unittest.SkipTest("Bash is required for launcher integration checks")
        cls.temporary = tempfile.TemporaryDirectory(prefix="emcad_launcher_")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name)
        cls.project = cls.base / "projects" / "emcad"
        shutil.copytree(ROOT / "sh", cls.project / "sh")
        cls.data = cls.base / "data"
        for dataset, family, _ in DATASETS:
            if family in ("synapse", "acdc"):
                root = cls.data / dataset
                for subdir in ("train", "valid", "test", "train_npz", "test_vol_h5"):
                    (root / subdir).mkdir(parents=True, exist_ok=True)
                lists = root / ("lists/lists_Synapse" if family == "synapse" else "lists_ACDC")
                lists.mkdir(parents=True, exist_ok=True)
                for split in ("train", "valid", "test", "test_vol"):
                    (lists / (split + ".txt")).touch()
            else:
                root = cls.data / family / "target" / dataset
                for split in ("train", "val", "test"):
                    for kind in ("images", "masks"):
                        (root / split / kind).mkdir(parents=True, exist_ok=True)
                for name in ("manifest.csv", "split_manifest.csv", "split_summary.json"):
                    (root / name).write_text("{}", encoding="utf-8")
        pretrained = cls.project / "pretrained_pth" / "pvt"
        pretrained.mkdir(parents=True)
        (pretrained / "pvt_v2_b2.pth").touch()
        cls.checkpoint = cls.project / "fixture_checkpoint" / "best.pth"
        cls.checkpoint.parent.mkdir()
        cls.checkpoint.touch()
        (cls.checkpoint.parent / "config.json").write_text("{}", encoding="utf-8")

    def resolve_shell_path(self, value):
        # Git Bash mounts /tmp at Windows TEMP and emits /c/... from pwd.
        if os.name == "nt" and value.startswith("/tmp/"):
            return (Path(tempfile.gettempdir()) / value[5:]).resolve()
        if os.name == "nt" and len(value) > 2 and value[0] == "/" and value[2] == "/":
            value = value[1] + ":" + value[2:]
        path = Path(value)
        return (path if path.is_absolute() else self.project / path).resolve()

    def launch(self, family, dataset, kind, caa=0, wavelet=0):
        capture = self.base / "captured_args.bin"
        capture.unlink(missing_ok=True)
        env = os.environ.copy()
        for key in ("DATA_ROOT", "DATASET", "DATASET_NAME", "CAA_MODE", "IMG_SIZE",
                    "USE_CONTENT_AWARE_ANTIALIAS", "USE_WAVELET_DETAIL", "CONDA_BASE",
                    "CONDA_ENV_PREFIX", "PYTHON_BIN", "INPUT_CHANNELS", "MERGE_INSTANCE_MASKS",
                    "USE_MULTI_SCALE_TRAINING", "LIST_DIR", "ROOT_PATH", "VOLUME_PATH",
                    "TRAIN_CONFIG", "SPLIT", "CKPT", "OUTPUT_DIR", "PRETRAINED_DIR", "N_GPU"):
            env.pop(key, None)
        env.update(DATASET_NAME=dataset, USE_CONTENT_AWARE_ANTIALIAS=str(caa),
                   USE_WAVELET_DETAIL=str(wavelet), CKPT=self.checkpoint.as_posix(),
                   CAPTURE_FILE=capture.as_posix())
        wrapper = r'''
source() {
  case "$1" in
    */etc/profile.d/conda.sh) return 0 ;;
    *) builtin source "$@" ;;
  esac
}
conda() { :; }
nohup() { printf '%s\0' "$@" > "$CAPTURE_FILE"; }
builtin source "$1"
wait
'''
        script = self.project / "sh" / ("start_" + kind + "_" + family + ".sh")
        result = subprocess.run([self.bash, "-c", wrapper, "launcher-test", script.as_posix()],
                                env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(capture.is_file(), result.stdout + result.stderr)
        tokens = capture.read_bytes().decode("utf-8").rstrip("\0").split("\0")
        index = next(i for i, token in enumerate(tokens) if token.endswith(".py"))
        return tokens[index], tokens[index + 1:]

    def test_all_dataset_training_ablations(self):
        for dataset, family, image_size in DATASETS:
            for caa, wavelet in ((0, 0), (1, 0), (0, 1), (1, 1)):
                with self.subTest(dataset=dataset, caa=caa, wavelet=wavelet):
                    entry, argv = self.launch(family, dataset, "train", caa, wavelet)
                    # BUSI/ISIC wrappers inherit the exact train_polyp parser.
                    parser_entry = "train_polyp.py" if family in ("busi", "isic") else entry
                    args = parse_source(parser_entry, argv)
                    if family in ("busi", "isic"):
                        args = parse_training_wrapper(entry, argv, args)
                    self.assertEqual(args.use_wavelet_detail, wavelet)
                    self.assertEqual(args.caa_mode, "caa" if caa else "off")
                    self.assertEqual(args.img_size, image_size)
                    if family not in ("synapse", "acdc"):
                        self.assertEqual(args.dataset_name, dataset)
                        self.assertEqual(self.resolve_shell_path(args.data_root), self.data / family / "target")
                        self.assertEqual(args.supervision, "paper")
                        self.assertEqual(args.no_multi_scale, family in ("busi", "cell"))
                        self.assertEqual(args.merge_instance_masks, int(dataset == "DSB18"))
                    else:
                        self.assertEqual(self.resolve_shell_path(args.root_path),
                                         self.data / dataset / ("train_npz" if family == "synapse" else ""))
                        self.assertEqual(args.supervision, "mutation")

    def test_all_dataset_test_launchers(self):
        for dataset, family, _ in DATASETS:
            with self.subTest(dataset=dataset):
                entry, argv = self.launch(family, dataset, "test")
                parser_entry = "test_polyp.py" if family in ("busi", "isic") else entry
                args = parse_source(parser_entry, argv)
                self.assertEqual(self.resolve_shell_path(args.checkpoint), self.checkpoint)
                if family not in ("synapse", "acdc"):
                    self.assertEqual(args.dataset_name, dataset)
                    self.assertEqual(self.resolve_shell_path(args.data_root), self.data / family / "target")
                # Architecture comes from checkpoint config, not training switches
                # inherited in the test launcher's environment.
                self.assertNotIn("--use_wavelet_detail", argv)
                self.assertNotIn("--caa_mode", argv)


class CheckpointConfigTests(unittest.TestCase):
    def test_acdc_and_shared_binary_config_restoration(self):
        with tempfile.TemporaryDirectory(prefix="emcad_config_") as directory:
            config_path = Path(directory) / "config.json"
            for filename, function in (("test_acdc.py", "restore_checkpoint_config"),
                                       ("test_polyp.py", "restore_checkpoint_model_options")):
                restore = load_function(filename, function, Path=Path, json=json, sys=sys)
                for config, expected in (({}, 0), ({"use_wavelet_detail": 1}, 1)):
                    with self.subTest(entry=filename, config=config):
                        config_path.write_text(json.dumps(config), encoding="utf-8")
                        args = argparse.Namespace(checkpoint=str(Path(directory) / "best.pth"),
                                                  use_wavelet_detail=1 - expected)
                        with patch.object(sys, "argv", [filename]):
                            restore(args)
                        self.assertEqual(args.use_wavelet_detail, expected)
                        args.use_wavelet_detail = 1 - expected
                        with patch.object(sys, "argv", [filename, "--use_wavelet_detail", str(1 - expected)]):
                            with self.assertRaises(RuntimeError):
                                restore(args)

    def test_busi_isic_architecture_config_restoration(self):
        # Execute the exact wrapper prefix through its architecture-lock loop.
        # Later metadata/manifest validation requires the real prepared dataset.
        for filename, dataset in (("test_busi.py", "BUSI"), ("test_isic.py", "ISIC2018")):
            tree = source_tree(filename)
            options_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == "ARCHITECTURE_OPTIONS"
                                        for t in n.targets))
            options = ast.literal_eval(options_node.value)
            function = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                                          and n.name == "_parse_args"))
            lock_index = next(i for i, n in enumerate(function.body) if isinstance(n, ast.For)
                              and "ARCHITECTURE_OPTIONS" in ast.unparse(n.iter))
            function.body = function.body[:lock_index + 1] + [ast.Return(value=ast.Name(id="args", ctx=ast.Load()))]
            ast.fix_missing_locations(function)
            with tempfile.TemporaryDirectory(prefix="emcad_wrapper_config_") as directory:
                defaults = dict.fromkeys(options, None)
                defaults.update(checkpoint=str(Path(directory) / "best.pth"), dataset_name=dataset,
                                data_root=directory, use_wavelet_detail=0)
                namespace = dict(Path=Path, json=json, sys=sys, ARCHITECTURE_OPTIONS=options,
                                 ALLOWED_DATASETS={"ISIC2017", "ISIC2018"},
                                 _original_parse_args=lambda: argparse.Namespace(**defaults))
                namespace["_normalized"] = load_function(filename, "_normalized")
                namespace["_option_was_given"] = load_function(filename, "_option_was_given", sys=sys)
                exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
                for wavelet in (None, 1):
                    config = {key: defaults[key] for key in options}
                    config.update(dataset_name=dataset, caa_mode="caa", caa_residual_scale=0.1)
                    if wavelet is None:
                        config.pop("use_wavelet_detail")
                    else:
                        config["use_wavelet_detail"] = wavelet
                    (Path(directory) / "config.json").write_text(json.dumps(config), encoding="utf-8")
                    with self.subTest(entry=filename, wavelet=wavelet):
                        with patch.object(sys, "argv", [filename]):
                            restored = namespace["_parse_args"]()
                        self.assertEqual(restored.use_wavelet_detail, 0 if wavelet is None else 1)


if __name__ == "__main__":
    unittest.main()
