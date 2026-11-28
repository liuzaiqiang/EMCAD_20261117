"""Pure checks for the Synapse online ranking and on-disk manifest."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.synapse_candidate_selection import (
    add_topk_entry, candidate_checkpoint_name, should_enter_topk,
    validate_topk_config,
)


def main():
    validate_topk_config(0, 0, float("nan"))
    for enabled, size in ((2, 7), (True, 7), (1, 0)):
        try:
            validate_topk_config(enabled, size, 83.9)
        except ValueError:
            pass
        else:
            raise AssertionError((enabled, size))
    board = []
    assert not should_enter_topk(0.838999, board, 7, 83.90)
    assert should_enter_topk(0.839, board, 7, 83.90)
    with TemporaryDirectory() as folder:
        root = Path(folder)
        history = []
        for epoch in range(1, 301):
            score = 0.839 if epoch <= 7 else 0.839 + ((epoch * 37) % 101) / 100000
            if not should_enter_topk(score, board, 7, 83.90):
                continue
            filename = candidate_checkpoint_name(epoch, score, 12.345)
            (root / filename).write_bytes(b'weights')
            entry = {'epoch': epoch, 'rank_dice': score, 'checkpoint': filename}
            history.append(entry)
            board, displaced = add_topk_entry(board, entry, 7)
            manifest = {'entries': board}
            (root / 'synapse_topk_manifest.json').write_text(json.dumps(manifest))
            for item in displaced:
                (root / item['checkpoint']).unlink()
        assert len(board) == 7
        assert len(history) > 7
        assert board == sorted(history, key=lambda x: (-x['rank_dice'], x['epoch']))[:7]
        assert {p.name for p in root.glob('*.pth')} == {x['checkpoint'] for x in board}
        assert json.loads((root / 'synapse_topk_manifest.json').read_text())['entries'] == board
        tail = min(x['rank_dice'] for x in board)
        assert not should_enter_topk(tail, board, 7, 83.90)
        assert should_enter_topk(tail + 1e-6, board, 7, 83.90)
    print('topN 300-epoch simulation OK')


if __name__ == '__main__':
    main()
