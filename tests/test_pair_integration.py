"""CPU smoke checks; no dataset results or accuracy claims are produced."""
import os
import sys
import json
from pathlib import Path
import importlib.util
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.networks import EMCADNet
import lib.pair_runtime as runtime

def check_pair(a, b, classes=1, size=64, encoder='pvt_v2_b0'):
    runtime.FRD_NAME, runtime.PDS_NAME = a, b
    for va, vb in ((0,0), (1,0), (0,1), (1,1)):
        os.environ['QFBLXJZL'], os.environ['SCYYZJCD'] = str(va), str(vb)
        torch.manual_seed(123)
        model = EMCADNet(num_classes=classes, encoder=encoder, pretrain=False)
        x = torch.randn(2, 1 if classes > 1 else 3, size, size)
        y = torch.randint(0, classes if classes > 1 else 2, (2, size, size))
        outputs, aux = runtime.train_forward(model, x)
        assert len(outputs) == 4 and all(p.shape == (2, classes, size, size) for p in outputs)
        def seg(p, y):
            return torch.nn.functional.binary_cross_entropy_with_logits(p, y[:,None].float()) if classes == 1 else torch.nn.functional.cross_entropy(p, y)
        loss = sum(seg(p,y) for p in outputs)
        loss = runtime.add_loss(loss, model, outputs, aux, y, seg)
        assert torch.isfinite(loss), (a,b,va,vb,classes,'loss')
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), (a,b,va,vb,classes,'grad')
        for name, module in getattr(model, 'pair_modules', {}).items():
            gs = [p.grad for p in module.parameters() if p.grad is not None]
            assert gs and any(g.abs().sum() > 0 for g in gs), (name, 'no learning signal')
        if 'caa' in model._pair_active:
            for stage in (3,2,1):
                up = getattr(model.decoder, 'eucb'+str(stage)).up_dwc[0]
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in up.parameters())
        model.eval()
        with torch.no_grad():
            pred = model(x)[-1]
            raw_outputs, eval_aux = model(x, return_pair_aux=True) if model._pair_active else (model(x),{})
            expected = eval_aux.get('fused', raw_outputs[-1])
            assert torch.equal(pred, expected)
        clone = EMCADNet(num_classes=classes, encoder=encoder, pretrain=False)
        clone.load_state_dict(model.state_dict(), strict=True)
        clone.eval()
        with torch.no_grad():assert torch.equal(pred, clone(x)[-1])
    return {'pair': [a,b], 'classes':classes, 'size':size, 'encoder':encoder, 'states':['00','10','01','11'], 'result':'passed'}

if __name__ == '__main__':
    torch.set_num_threads(2)
    result = check_pair(runtime.FRD_NAME, runtime.PDS_NAME)
    print(json.dumps(result))
