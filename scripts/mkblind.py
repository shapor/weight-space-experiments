import sys, os, re, json, random, numpy as np
L, OUT = sys.argv[1], sys.argv[2]
rng = random.Random(7); key = {}
def write(text, truth):
    bid = "%06x" % rng.getrandbits(24); open(f"{OUT}/{bid}.txt","w").write(text); key[bid] = truth
def blocks(text):
    # split into header lines and numeric matrix blocks
    lines = text.split("\n"); return lines
def permute(text, name, perm, axis):
    """permute rows (axis 0) or columns (axis 1) of the matrix block after header starting with name"""
    lines = text.split("\n"); i = next(k for k,l in enumerate(lines) if l.startswith(name+" ")) + 1
    j = i
    while j < len(lines) and re.match(r"^-?\d", lines[j]): j += 1
    rows = [l.split() for l in lines[i:j]]
    if axis == 0: rows = [rows[p] for p in perm] + rows[len(perm):]
    else: rows = [[r[p] for p in perm] + r[len(perm):] for r in rows]
    lines[i:j] = [" ".join(r) for r in rows]; return "\n".join(lines)
for g in ["breakout","pong","pacman","freeway"]:
    t = open(f"{L}/exp1_which_game/runs/{g}/prompt.txt").read(); write(t, {"exp":1,"truth":g})
# control: scramble grid cells (same perm for all 5 channels) of breakout
t = open(f"{L}/exp1_which_game/runs/breakout/prompt.txt").read()
cells = list(range(100)); random.Random(1).shuffle(cells)
write(permute(t, "W1", [c*5+ch for c in cells for ch in range(5)], 0), {"exp":1,"truth":"CONTROL breakout cells scrambled"})
for s in ["circle","square","triangle"]:
    t = open(f"{L}/exp2_diffusion/runs/{s}/prompt.txt").read(); write(t, {"exp":2,"truth":s})
t = open(f"{L}/exp2_diffusion/runs/circle/prompt.txt").read()
px = list(range(144)); random.Random(2).shuffle(px)
t = permute(permute(t, "W1", px, 0), "W3", px, 1)
# b3 is one line of 144: permute too
lines = t.split("\n"); k = next(i for i,l in enumerate(lines) if l.startswith("b3 ")) + 1
v = lines[k].split(); lines[k] = " ".join(v[p] for p in px); t = "\n".join(lines)
write(t, {"exp":2,"truth":"CONTROL circle pixels scrambled"})
json.dump(key, open(f"{OUT}/../blind_key.json","w"), indent=1)
