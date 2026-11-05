import sys, numpy as np
from multiprocessing import Pool
from minatar import Environment
# channel mapping from the exp5 prompts: our 6 channels <- MinAtar channel names
MAP = {"breakout": [["paddle"], [], ["brick"], ["ball"], ["trail"], []],
       "seaquest": [["sub_front","sub_back"], ["enemy_fish","enemy_sub","enemy_bullet"], ["enemy_fish","enemy_sub","diver"],
                    ["friendly_bullet"], ["trail","sub_back"], ["oxygen_guage","diver_guage"]],
       "asterix": [["player"], ["enemy"], ["gold"], [], ["trail"], []]}
def obs(env, game):
    s = env.state(); names = env.env.channels; o = np.zeros((10,10,6), np.float32)
    for c, srcs in enumerate(MAP[game]):
        for n in srcs: o[:,:,c] = np.maximum(o[:,:,c], s[:,:,names[n]])
    return o.reshape(-1)
def policy_net(W):
    def f(o, env, rng):
        h = np.tanh(o @ W["actor/dense1/kernel"] + W["actor/dense1/bias"])
        h = np.tanh(h @ W["actor/dense2/kernel"] + W["actor/dense2/bias"])
        z = h @ W["actor/logits/kernel"] + W["actor/logits/bias"]; p = np.exp(z - z.max()); p /= p.sum()
        return rng.choice(6, p=p)
    return f
def follow_ball(o, env, rng):  # trivial heuristic: move paddle under the ball's column
    g = o.reshape(10,10,6); pc = np.argwhere(g[:,:,0])[0][1]; b = np.argwhere(g[:,:,3])
    if len(b) == 0: return 0
    bc = b[0][1]; return 1 if bc < pc else 3 if bc > pc else 0
def episode(args):
    game, pol, seed = args
    np.random.seed(seed); env = Environment(game, sticky_action_prob=0.1); getattr(env, "random", np.random).seed(seed); env.reset(); rng = np.random.default_rng(seed)
    f = policy_net(dict(np.load(pol))) if pol.endswith(".npz") else {"follow_ball": follow_ball, "random": lambda o,e,r: r.integers(6)}[pol]
    R, t, done = 0.0, 0, False
    while not done and t < 10000:
        r, done = env.act(f(obs(env, game), env, rng)); R += r; t += 1
    return R, t
if __name__ == "__main__":
    game, pol, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with Pool(int(sys.argv[4]) if len(sys.argv) > 4 else 8) as p: res = np.array(p.map(episode, [(game, pol, s) for s in range(n)]))
    print(f"{game} {pol.split('/')[-2] if '/' in pol else pol}: mean {res[:,0].mean():.1f} se {res[:,0].std()/np.sqrt(n):.1f} max {res[:,0].max():.0f} len {res[:,1].mean():.0f}  (n={n})")
