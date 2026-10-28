"""Launch-command/config tests with isolated empty filesystem fixtures.

No training process is started. Conda activation and nohup are intercepted.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
BASH = 'D:/install/Git/Git/bin/bash.exe' if os.name == 'nt' else 'bash'
DATASETS = {'synapse': ['Synapse'], 'acdc': ['ACDC'],
            'polyp': ['BKAI', 'ClinicDB', 'ColonDB', 'ETIS', 'Kvasir'],
            'isic': ['ISIC2017', 'ISIC2018'], 'busi': ['BUSI'], 'cell': ['DSB18', 'EM']}


def functions(filename):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    namespace = dict(argparse=argparse, Path=Path, json=json, sys=sys, os=os)
    exec(compile(tree, filename, 'exec'), namespace)
    return namespace


def parser(filename):
    if 'synapse' not in filename:
        return functions(filename)['parse_args']
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8-sig'))
    tree.body = [n for n in tree.body if
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser' for t in n.targets))
                 or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                     and isinstance(n.value.func, ast.Attribute)
                     and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'parser')]
    namespace = dict(argparse=argparse)
    exec(compile(tree, filename, 'exec'), namespace)
    return namespace['parser'].parse_args


class EntryTests(unittest.TestCase):
    def test_wrapper_architecture_and_synapse_restore(self):
        for filename in ['test_busi.py', 'test_isic.py']:
            tree = ast.parse((ROOT/filename).read_text(encoding='utf-8-sig'))
            ns = functions(filename)
            constants = [n for n in tree.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'ARCHITECTURE_OPTIONS'
                                 for t in n.targets)]
            exec(compile(ast.Module(body=constants, type_ignores=[]), filename, 'exec'), ns)
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == '_parse_args')
            loop = next(n for n in function.body if isinstance(n, ast.For)
                        and isinstance(n.iter, ast.Call)
                        and isinstance(n.iter.func, ast.Attribute)
                        and isinstance(n.iter.func.value, ast.Name)
                        and n.iter.func.value.id == 'ARCHITECTURE_OPTIONS')
            block = compile(ast.Module(body=[loop], type_ignores=[]), filename, 'exec')
            for saved in [None, 0, 1]:
                values = {key: 0 for key in ns['ARCHITECTURE_OPTIONS']}
                values.update(caa_mode='off', caa_residual_scale=0.1)
                if saved is None:
                    values.pop('use_contour_consistency')
                else:
                    values['use_contour_consistency'] = saved
                args = argparse.Namespace(**{key: 1 for key in ns['ARCHITECTURE_OPTIONS']})
                ns.update(args=args, config=values)
                with patch.object(sys, 'argv', ['test']):
                    exec(block, ns)
                self.assertEqual(args.use_contour_consistency, saved or 0)
                args.use_contour_consistency = 1-args.use_contour_consistency
                with patch.object(sys, 'argv', ['test', '--use_contour_consistency', str(args.use_contour_consistency)]):
                    with self.assertRaises(RuntimeError):
                        exec(block, ns)
        tree = ast.parse((ROOT/'test_synapse.py').read_text(encoding='utf-8-sig'))
        restore = next(n for n in tree.body if isinstance(n, ast.If)
                       and ast.unparse(n.test) == 'args.train_config')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            for saved in [None, 0, 1]:
                path.write_text(json.dumps({} if saved is None else {'use_contour_consistency': saved}), encoding='utf-8')
                args = argparse.Namespace(train_config=str(path), use_contour_consistency=1)
                exec(compile(ast.Module(body=[restore], type_ignores=[]), 'test_synapse.py', 'exec'),
                     dict(args=args, os=os, json=json))
                self.assertEqual(args.use_contour_consistency, saved or 0)

    def test_checkpoint_restore_and_old_defaults(self):
        for family, fn in [('acdc', 'restore_checkpoint_config'),
                           ('polyp', 'restore_checkpoint_model_options')]:
            ns = functions('test_' + family + '.py')
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                for saved in [None, 0, 1]:
                    config = {} if saved is None else {'use_contour_consistency': saved}
                    (path / 'config.json').write_text(json.dumps(config), encoding='utf-8')
                    args = argparse.Namespace(checkpoint=str(path/'best.pth'), use_contour_consistency=1)
                    with patch.object(sys, 'argv', ['test']):
                        ns[fn](args)
                    self.assertEqual(args.use_contour_consistency, saved or 0)
                    with patch.object(sys, 'argv', ['test', '--use_contour_consistency', str(1-args.use_contour_consistency)]):
                        args.use_contour_consistency = 1-args.use_contour_consistency
                        with self.assertRaises(RuntimeError):
                            ns[fn](args)

    def test_12_datasets_all_four_train_switches_and_test_launchers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'sh').mkdir()
            for src in (ROOT/'sh').glob('start_*.sh'):
                text = src.read_text(encoding='utf-8-sig')
                text = re.sub(r'^\s*source .*conda\.sh.*$', ': # intercepted conda activation', text, flags=re.M)
                # Some launchers intentionally fix server paths instead of
                # accepting env overrides; relocate only those fixture paths.
                text = re.sub(r'^(ROOT_PATH|LIST_DIR|VOLUME_PATH)=.*$',
                              r'\1="${FIXTURE_DATA}"', text, flags=re.M)
                (root/'sh'/src.name).write_text(text, encoding='utf-8', newline='\n')
            data = root/'data'
            for names in DATASETS.values():
                for dataset in names:
                    for split in ['train', 'val', 'test']:
                        for kind in ['images', 'masks']:
                            (data/dataset/split/kind).mkdir(parents=True)
                    for file in ['manifest.csv', 'split_manifest.csv', 'split_summary.json']:
                        (data/dataset/file).touch()
            for name in ['train', 'valid', 'test']:
                (data/name).mkdir(parents=True, exist_ok=True)
            for name in ['train.txt', 'valid.txt', 'test.txt', 'test_vol.txt']:
                (data/name).touch()
            (root/'pretrained_pth/pvt').mkdir(parents=True)
            (root/'pretrained_pth/pvt/pvt_v2_b2.pth').touch()
            (root/'best.pth').touch()
            (root/'config.json').write_text('{}', encoding='utf-8')
            capture = root/'capture'
            env = dict(os.environ, DATA_ROOT=data.as_posix(), ROOT_PATH=data.as_posix(),
                       FIXTURE_DATA=data.as_posix(),
                       LIST_DIR=data.as_posix(), VOLUME_PATH=data.as_posix(),
                       CKPT=(root/'best.pth').as_posix(), CAPTURE=capture.as_posix())
            # Only shell activation and actual subprocess launch are replaced.
            prelude = '''conda() { :; }
nohup() { printf '%s\\0' "$@" > "$CAPTURE"; }
source "$1"
wait
'''
            count = 0
            for family, names in DATASETS.items():
                for dataset in names:
                    for caa, contour in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        env.update(DATASET_NAME=dataset, USE_CONTENT_AWARE_ANTIALIAS=str(caa),
                                   USE_CONTOUR_CONSISTENCY=str(contour))
                        script = root/'sh'/('start_train_'+family+'.sh')
                        result = subprocess.run([BASH, '-c', prelude, 'fixture', script.as_posix()],
                                                env=env, capture_output=True, text=True, timeout=30)
                        self.assertEqual(result.returncode, 0, (dataset, result.stdout, result.stderr))
                        argv = capture.read_bytes().decode().rstrip('\0').split('\0')
                        self.assertEqual(argv[argv.index('--use_contour_consistency')+1], str(contour))
                        self.assertEqual(argv[argv.index('--caa_mode')+1], 'caa' if caa else 'off')
                        filename = argv[argv.index('-u')+1]
                        if family in ['busi', 'isic']:
                            filename = 'train_polyp.py'  # wrappers inherit this parser
                        with patch.object(sys, 'argv', [filename]+argv[argv.index('-u')+2:]):
                            args = parser(filename)()
                        self.assertEqual(args.use_contour_consistency, contour)
                        count += 1
                    script = root/'sh'/('start_test_'+family+'.sh')
                    result = subprocess.run([BASH, '-c', prelude, 'fixture', script.as_posix()],
                                            env=env, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, (dataset, result.stdout, result.stderr))
                    argv = capture.read_bytes().decode().rstrip('\0').split('\0')
                    self.assertNotIn('--use_contour_consistency', argv)  # restore saved architecture
                    self.assertIn('--checkpoint', argv)
            self.assertEqual(count, 48)


if __name__ == '__main__':
    unittest.main()
