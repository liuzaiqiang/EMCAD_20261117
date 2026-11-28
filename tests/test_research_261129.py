"""CPU smoke tests; must be run in the actual project environment."""
import unittest
import torch
from lib.research_261129 import PAIR,add_auxiliary_loss,WTConv
from lib.networks import EMCADNet
class ResearchSmoke(unittest.TestCase):
    def test_all_ablation_settings(self):
        torch.set_num_threads(1)
        for classes in (1,4,9):
            for a,b in ((0,0),(1,0),(0,1),(1,1)):
                flags=dict(zip(PAIR,(a,b)))
                with self.subTest(classes=classes,flags=flags):
                    model=EMCADNet(num_classes=classes,encoder="pvt_v2_b0",pretrain=False,research_flags=flags)
                    model.train()
                    x=torch.randn(2,1 if classes>1 else 3,64,64)
                    y=torch.randint(0,classes if classes>1 else 2,(2,64,64))
                    p,aux=model(x,mode="train",return_research_aux=True)
                    self.assertEqual(len(p),4)
                    self.assertTrue(all(z.shape==(2,classes,64,64) for z in p))
                    loss=add_auxiliary_loss(sum(z.float().square().mean() for z in p),aux,y)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    if a or b:
                        for name,block in model.research_pair.blocks.items():
                            grads=[v.grad for v in block.parameters() if v.requires_grad]
                            self.assertTrue(grads and all(g is not None and torch.isfinite(g).all() for g in grads),name)
                    else:
                        self.assertFalse(hasattr(model,"research_pair"))
                    model.eval()
                    with torch.no_grad():
                        pred=model(x)
                        self.assertTrue(all(torch.isfinite(z).all() for z in pred))
                    clone=EMCADNet(num_classes=classes,encoder="pvt_v2_b0",pretrain=False,research_flags=flags)
                    clone.load_state_dict(model.state_dict(),strict=True)
    def test_wavelet_round_trip(self):
        x=torch.randn(2,5,8,10)
        torch.testing.assert_close(WTConv.reconstruct(WTConv.decompose(x)),x)
if __name__=="__main__":
    unittest.main()
