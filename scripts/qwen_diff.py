import sys, glob, torch, numpy as np
from safetensors.torch import load_file
def load(d): 
    out = {}
    for f in glob.glob(d + "/*.safetensors"): out.update(load_file(f))
    return out
a, b = load(sys.argv[1]), load(sys.argv[2])
print("keys only in base:", len(set(a) - set(b)), " only in edited:", len(set(b) - set(a)))
tot = ch = 0
for k in sorted(a):
    if k not in b: continue
    x, y = a[k].float(), b[k].float(); tot += x.numel()
    if x.shape != y.shape: print("SHAPE", k, x.shape, y.shape); continue
    d = (x != y); n = int(d.sum())
    if n:
        ch += n; D = (y - x)
        cols = d.any(0).nonzero().flatten().tolist() if D.dim() == 2 else []
        rows = d.any(1).nonzero().flatten().tolist() if D.dim() == 2 else []
        s = torch.linalg.svdvals(D)[:4].tolist() if D.dim() == 2 else []
        print(f"{k} {tuple(x.shape)} changed={n} rows={len(rows)} cols={len(cols)} |D|/|W|={D.norm()/x.norm():.4f} top-SV={['%.3g'%v for v in s]}")
        if D.dim() == 2 and len(cols) <= 64: print("   cols:", cols[:64])
print(f"total changed {ch} of {tot} = {100*ch/tot:.4f}%")
