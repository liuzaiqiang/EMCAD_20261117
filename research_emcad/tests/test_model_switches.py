"""开关关闭时维持原始读出，开启时仅改写最高分辨率输出。"""
import io
import unittest

import torch
from torch.nn import functional as F

from lib.networks import EMCADNet


class ModelSwitchTests(unittest.TestCase):
    def test_all_switch_combinations_preserve_baseline_initialization(self):
        image = torch.randn(1, 3, 128, 128)
        torch.manual_seed(731)
        baseline = EMCADNet(num_classes=4, encoder='pvt_v2_b0', pretrain=False).eval()
        baseline_state = baseline.state_dict()
        with torch.no_grad():
            baseline_outputs = baseline(image)
        scale_reference = None
        for scale in (False, True):
            for skip in (False, True):
                torch.manual_seed(731)
                model = EMCADNet(num_classes=4, encoder='pvt_v2_b0', pretrain=False,
                                 use_class_conditional_scale_fusion=scale,
                                 use_complementary_skip_fusion=skip).eval()
                for name, value in baseline_state.items():
                    self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
                with torch.no_grad():
                    outputs = model(image)
                if not scale:
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(outputs, baseline_outputs)))
                elif not skip:
                    scale_reference = outputs
                else:
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(outputs, scale_reference)))
                if skip:
                    self.assertTrue(any(name.startswith('decoder.complementary_skip1.')
                                        for name in model.state_dict()))
                else:
                    self.assertFalse(any('complementary_skip' in name for name in model.state_dict()))
                if scale:
                    self.assertTrue(any(name.startswith('class_conditional_scale_fusion.')
                                        for name in model.state_dict()))
                else:
                    self.assertFalse(any(name.startswith('class_conditional_scale_fusion.')
                                         for name in model.state_dict()))

    def test_off_matches_original_heads(self):
        torch.manual_seed(123)
        model = EMCADNet(num_classes=4, encoder='pvt_v2_b0', pretrain=False,
                         use_class_conditional_scale_fusion=False).eval()
        self.assertIsNone(model.class_conditional_scale_fusion)
        self.assertFalse(any(n.startswith('class_conditional_scale_fusion.')
                             for n in model.state_dict()))
        image = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            actual = model(image)
            x1, x2, x3, x4 = model.backbone(image)
            decoded = model.decoder(x4, [x3, x2, x1])
            expected = [F.interpolate(head(feature), scale_factor=scale, mode='bilinear')
                        for head, feature, scale in zip(
                            (model.out_head4, model.out_head3, model.out_head2, model.out_head1),
                            decoded, (32, 16, 8, 4))]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
        self.assertTrue(all(x.shape == (1, 4, 128, 128) for x in actual))

    def test_on_preserves_other_heads_and_restores_checkpoint(self):
        torch.manual_seed(123)
        base = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False).eval()
        torch.manual_seed(123)
        model = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False,
                         use_class_conditional_scale_fusion=True).eval()
        self.assertEqual(set(model.state_dict()) - set(base.state_dict()),
                         {f'class_conditional_scale_fusion.{key}'
                          for key in model.class_conditional_scale_fusion.state_dict()})
        image = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            reference, actual = base(image), model(image)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference[:3], actual[:3])))
        self.assertFalse(torch.equal(reference[-1], actual[-1]))
        model.train()
        model(image.repeat(2, 1, 1, 1))[-1].square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.class_conditional_scale_fusion.parameters()))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, map_location='cpu'), strict=True)

    def test_binary_grayscale_and_multiclass_resnet_contract(self):
        for encoder, classes, channels in (('pvt_v2_b2', 1, 1), ('resnet18', 9, 3)):
            model = EMCADNet(num_classes=classes, encoder=encoder, pretrain=False,
                             use_class_conditional_scale_fusion=True,
                             use_complementary_skip_fusion=True).eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, channels, 128, 128))
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(tensor.shape == (1, classes, 128, 128)
                                for tensor in outputs))

    def test_skip_fusion_receives_training_gradient(self):
        model = EMCADNet(num_classes=1, encoder='pvt_v2_b0', pretrain=False,
                         use_complementary_skip_fusion=True).train()
        model(torch.randn(2, 3, 128, 128))[-1].square().mean().backward()
        for stage in (1, 2, 3):
            module = getattr(model.decoder, f'complementary_skip{stage}')
            self.assertTrue(torch.count_nonzero(module.novel_gain.grad).item() > 0)
            self.assertTrue(torch.count_nonzero(module.shared_gain.grad).item() > 0)


if __name__ == '__main__':
    unittest.main()
