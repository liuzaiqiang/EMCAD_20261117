import csv

import torch

from lib.networks import EMCADNet
from lib.topn_checkpoints import TopNCheckpoints
from lib.uaca import UACA


def test_uaca_binary_and_multiclass_backward():
    for classes in (1, 4, 9):
        block = UACA(16, 8)
        head = torch.nn.Conv2d(8, classes, 1)
        feature = torch.randn(2, 16, 16, 16, requires_grad=True)
        prior = torch.randn(2, classes, 8, 8, requires_grad=True)
        refined, logits = block(feature, prior, head)
        assert refined.shape == (2, 8, 16, 16)
        assert logits.shape == (2, classes, 16, 16)
        logits.square().mean().backward()
        assert feature.grad is not None
        assert prior.grad is not None


def test_disabled_uaca_preserves_baseline_parameters():
    torch.manual_seed(11)
    baseline = EMCADNet(encoder="pvt_v2_b0", pretrain=False)
    torch.manual_seed(11)
    disabled = EMCADNet(encoder="pvt_v2_b0", pretrain=False, uaca_enabled=0)
    assert set(baseline.state_dict()) == set(disabled.state_dict())
    assert all(torch.equal(value, disabled.state_dict()[key])
               for key, value in baseline.state_dict().items())
    assert not any("uaca" in key for key in disabled.state_dict())
    baseline.eval()
    disabled.eval()
    sample = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        expected = baseline(sample)
        actual = disabled(sample)
    assert all(torch.equal(left, right) for left, right in zip(expected, actual))


def test_topn_keeps_best_dice_and_tracks_hd95(tmp_path):
    saved = []

    def save(_, path):
        saved.append(path)
        with open(path, "w", encoding="utf-8") as stream:
            stream.write("checkpoint")

    manager = TopNCheckpoints(str(tmp_path), 2, save)
    for epoch, dice, hd95 in ((1, 0.7, 8.0), (2, 0.8, 6.0), (3, 0.6, 4.0)):
        manager.update(None, epoch, dice, hd95)
    assert [entry[1] for entry in manager.entries] == [2, 1]
    assert len(saved) == 2
    with open(manager.trajectory, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["hd95"] for row in rows] == ["8.0", "6.0", "4.0"]
