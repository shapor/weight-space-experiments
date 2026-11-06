"""Control for weight-space-galaxy10: how much of 73.7% comes from the programmed measurements, and how much
from the closed-form QDA head? Same head, hyperparameters frozen to the author's (alpha .5, T .65, shrink .1,
9 LDA directions), fit on train only; only the 151 input features change.

usage: galaxy_control.py <Galaxy10_DECals.h5> <clone of github.com/DaybreakWannaCode/weight-space-galaxy10>
Writes control_results.json to the current directory. About 15 minutes on a 24-core CPU.
"""
import json, sys
import h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

h5, repo = sys.argv[1:3]
torch.set_num_threads(16)

# --- data: 64x64 images (the network's own first layer is AvgPool2d(4)), the challenge split ---
with h5py.File(h5) as f:
    y = f["ans"][:].astype(int)
    X = torch.cat([F.avg_pool2d(torch.from_numpy(f["images"][s:s + 512]).permute(0, 3, 1, 2).float() / 255, 4)
                   for s in range(0, len(y), 512)])
rng, sp = np.random.default_rng(1234), {"train": [], "val": [], "test": []}
for c in np.unique(y):                                 # stratified split, same procedure as the repo's make_splits
    i = rng.permutation(np.flatnonzero(y == c)); a, b = round(.8 * len(i)), round(.1 * len(i))
    sp["train"].append(i[:a]); sp["val"].append(i[a:a + b]); sp["test"].append(i[a + b:])
sp = {k: np.sort(np.concatenate(v)) for k, v in sp.items()}
print({k: len(v) for k, v in sp.items()}, flush=True)

# --- trunks: layers 1..8 of the submitted network, or the same shapes with other weights ---
arch = json.load(open(f"{repo}/galaxy/runs/r1/submission/arch.json"))
sd = torch.load(f"{repo}/galaxy/runs/r1/submission/submission.pt", weights_only=True)


def trunk(weights):
    m = nn.Sequential(*[getattr(nn, l["type"])(**l.get("args", {})) for l in arch[:9]])
    m.load_state_dict({k: v for k, v in weights.items() if int(k.split(".")[0]) <= 8})
    with torch.no_grad():
        return torch.cat([m[1:](X[s:s + 256]).flatten(1) for s in range(0, len(X), 256)]).double().numpy()


def random_filters(seed):          # the author's suggested control: random first-layer filters, same norms and biases
    g, w = torch.Generator().manual_seed(seed), sd["1.weight"]
    r = torch.randn(w.shape, generator=g)
    return {**sd, "1.weight": r * w.flatten(1).norm(dim=1)[:, None, None, None] / r.flatten(1).norm(dim=1)[:, None, None, None]}


def random_trunk(seed):            # nothing designed: random filters, random 27x27 maps, random 3-channel sums
    g = torch.Generator().manual_seed(seed)
    w = {k: v.clone() for k, v in sd.items()}
    w["1.weight"] = torch.randn(76, 3, 13, 13, generator=g) / 13; w["1.bias"] = torch.zeros(76)
    w["4.weight"] = F.one_hot(torch.randint(76, (430,), generator=g), 76).float()[:, :, None, None]
    w["5.weight"] = torch.randn(430, 1, 27, 27, generator=g) / 27; w["5.bias"] = torch.zeros(430)
    w["7.weight"] = torch.zeros(151, 430, 1, 1)
    for i in range(151):
        w["7.weight"][i, torch.randperm(430, generator=g)[:3], 0, 0] = 1.0
    return w


def pca(k=151):                    # generic: principal components of 32x32 pixels
    P = F.avg_pool2d(X, 2).flatten(1).double().numpy()
    mu = P[sp["train"]].mean(0)
    _, _, Vt = np.linalg.svd(P[sp["train"]][::2] - mu, full_matrices=False)
    return (P - mu) @ Vt[:k].T


# --- the author's head, in exact arithmetic (the network approximates log and squares with ReLU staircases) ---
def head(feat, alpha=.5, T=.65, shrink=.1, nq=9):
    tr = sp["train"]
    U = []
    for x, xt in zip(feat.T, feat[tr].T):
        if (xt >= 0).all():                                    # nonnegative: clipped log, as the network's layers 9-11
            pos = xt[xt > 0]; lo = np.quantile(pos, .005) if len(pos) else 1e-3
            hi = max(np.quantile(xt, .995), 4 * lo); g = lambda v: np.log(np.clip(v, lo, hi))
        else:                                                  # signed (PCA): clipped linear
            lo, hi = np.quantile(xt, [.005, .995]); g = lambda v: np.clip(v, lo, hi)
        u = (g(x) - g(lo)) / (g(hi) - g(lo)); U += [u, u * u]
    U = np.stack(U, 1)
    Ut, yt = U[tr], y[tr]
    prior = np.bincount(yt, minlength=10) / len(yt)
    mu = np.stack([Ut[yt == c].mean(0) for c in range(10)])
    covs = np.stack([np.cov(Ut[yt == c].T) for c in range(10)])
    r = Ut - mu[yt]; Sw = r.T @ r / (len(yt) - 10)
    Sw = (1 - shrink) * Sw + shrink * np.diag(np.diag(Sw)); Sw += 1e-6 * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    ubar = prior @ mu; Sb = (mu - ubar).T @ np.diag(prior) @ (mu - ubar)
    Li = np.linalg.inv(np.linalg.cholesky(Sw))
    E = np.linalg.eigh(Li @ Sb @ Li.T)[1][:, ::-1][:, :nq]; V = Li.T @ E
    Z, muz = (U - ubar) @ V, (mu - ubar) @ V
    logits = np.zeros((len(U), 10))
    for c in range(10):
        S = (1 - alpha) * V.T @ covs[c] @ V + alpha * np.eye(nq); d = Z - muz[c]
        logits[:, c] = T * (-.5 * np.einsum("ni,ij,nj->n", d, np.linalg.inv(S), d) - .5 * np.linalg.slogdet(S)[1]) + np.log(prior[c])
    lp = torch.log_softmax(torch.from_numpy(logits), 1).numpy()
    return {k: dict(ce=round(float(-lp[i, y[i]].mean()), 3), acc=round(float((lp[i].argmax(1) == y[i]).mean()), 3))
            for k, i in (("val", sp["val"]), ("test", sp["test"]))}


runs = {"A  author's programmed measurements": lambda: trunk(sd)}
for s in range(3):
    runs[f"B{s} author's network, random first-layer filters"] = lambda s=s: trunk(random_filters(s))
for s in range(3):
    runs[f"C{s} same architecture, all random weights"] = lambda s=s: trunk(random_trunk(s))
runs["D  PCA of 32x32 pixels, 151 components"] = pca
out = {}
for name, fn in runs.items():
    out[name] = head(fn()); print(name, out[name], flush=True)
json.dump(out, open("control_results.json", "w"), indent=1)
