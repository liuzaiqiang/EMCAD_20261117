import sys, unittest, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.model_complexity import ArithmeticAudit, measure_model_complexity
class AuditTests(unittest.TestCase):
    def test_functional_grouped_conv_and_matrix(self):
        x=torch.zeros(2,3,8,8);w=torch.zeros(6,1,3,3)
        with ArithmeticAudit() as a:
            torch.nn.functional.conv2d(x,w,padding=1,groups=3)
            torch.mm(torch.zeros(5,7),torch.zeros(7,11))
        self.assertEqual(sum(a.macs.values()), 2*6*8*8*9+5*7*11)
        self.assertFalse(a.unknown)
    def test_unknown_fails(self):
        class Unknown(torch.nn.Module):
            def forward(self,x): return torch.sin(x)
        with self.assertRaisesRegex(RuntimeError,'Unclassified'):
            measure_model_complexity(Unknown(),(1,1,4,4))
    def test_live_model_unchanged(self):
        m=torch.nn.Sequential(torch.nn.Conv2d(1,2,3),torch.nn.BatchNorm2d(2)).train()
        before={k:v.clone() for k,v in m.state_dict().items()}
        measure_model_complexity(m,(1,1,8,8))
        self.assertTrue(m.training)
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))
        self.assertFalse(any('total_ops' in k for k in m.state_dict()))
if __name__=='__main__':unittest.main()
