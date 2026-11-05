# 9x9 Go per the exp6 prompt: simple ko illegal, whole-board repetition = mover loses, 2 passes or 162 moves end,
# area scoring, komi 7.5. Networks: 243 -> 512x6 ReLU -> 82, moves sampled from softmax over legal moves.
import sys, numpy as np
from multiprocessing import Pool
N = 9; NB = [[(r*N+c2) for (r, c2) in ((i//N-1, i%N), (i//N+1, i%N), (i//N, i%N-1), (i//N, i%N+1)) if 0 <= r < N and 0 <= c2 < N] for i in range(81)]
def group(b, p):
    col, st, g, libs = b[p], [p], {p}, set()
    while st:
        q = st.pop()
        for n in NB[q]:
            if b[n] == 0: libs.add(n)
            elif b[n] == col and n not in g: g.add(n); st.append(n)
    return g, libs
def play(b, p, col):
    """returns new board and #captured, or None if suicide"""
    b = b.copy(); b[p] = col; cap = []
    for n in NB[p]:
        if b[n] == -col:
            g, l = group(b, n)
            if not l: cap += g
    for q in cap: b[q] = 0
    if not cap and not group(b, p)[1]: return None, 0
    return b, cap
def load(d):
    return [(np.load(f"{d}/L{i}.W.npy"), np.load(f"{d}/L{i}.b.npy")) for i in range(1, 7)] + [(np.load(f"{d}/out.W.npy"), np.load(f"{d}/out.b.npy"))]
def net_logits(P, b, col):
    x = np.zeros((81, 3), np.float32); x[:, 0] = b == col; x[:, 1] = b == -col; x[:, 2] = col == -1
    h = x.reshape(-1)
    for W, bb in P[:-1]: h = np.maximum(0, h @ W + bb)
    return h @ P[-1][0] + P[-1][1]
def game(args):
    pa, pb, seed = args; rng = np.random.default_rng(seed)
    nets = {1: pa, -1: pb} if seed % 2 == 0 else {1: pb, -1: pa}  # A is Black in even games
    nets = {k: (load(v) if v != "random" else None) for k, v in nets.items()}
    b = np.zeros(81, np.int8); col = 1; seen = {b.tobytes()}; ko = None; passes = 0
    for move in range(162):
        legal, nxt = [], {}
        for p in range(81):
            if b[p] or p == ko: continue
            nb, cap = play(b, p, col)
            if nb is not None: legal.append(p); nxt[p] = (nb, cap)
        legal.append(81)
        if nets[col] is None: m = rng.choice(legal)
        else:
            z = net_logits(nets[col], b, col)[legal]; z = np.exp(z - z.max()); m = legal[rng.choice(len(legal), p=z / z.sum())]
        if m == 81:
            passes += 1
            if passes == 2: break
            ko = None
        else:
            passes = 0; nb, cap = nxt[m]
            if nb.tobytes() in seen: return (1 if (col == 1) == (seed % 2 == 0) else 0) ^ 1  # mover loses
            ko = cap[0] if len(cap) == 1 and len(group(nb, m)[0]) == 1 and len(group(nb, m)[1]) == 1 else None
            b = nb; seen.add(b.tobytes())
        col = -col
    # area score
    score = {1: int((b == 1).sum()), -1: int((b == -1).sum()) + 7.5}; done = set()
    for p in range(81):
        if b[p] == 0 and p not in done:
            reg, st, border = {p}, [p], set()
            while st:
                q = st.pop()
                for n in NB[q]:
                    if b[n] == 0 and n not in reg: reg.add(n); st.append(n)
                    elif b[n]: border.add(int(b[n]))
            done |= reg
            if len(border) == 1: score[border.pop()] += len(reg)
    black_wins = score[1] > score[-1]
    return int(black_wins == (seed % 2 == 0))  # 1 if A won
if __name__ == "__main__":
    a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with Pool(8) as pool: w = np.array(pool.map(game, [(a, b, s) for s in range(n)]))
    print(f"{a.split('/')[-1]} vs {b.split('/')[-1]}: A wins {w.mean()*100:.1f}% ± {w.std()/np.sqrt(n)*100:.1f} (n={n})", flush=True)
