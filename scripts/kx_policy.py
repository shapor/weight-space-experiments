# Run base and edited Kinetix actors on l/h2_car_ramp: 1024 sampled episodes each, plus per-step policy certainty.
import sys, jax, jax.numpy as jnp, numpy as np
from kinetix.environment.env import make_kinetix_env
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.util.saving import load_evaluation_levels
from kinetix.environment.env_state import EnvParams
levels, sp = load_evaluation_levels(["l/h2_car_ramp.json"]); lvl = jax.tree.map(lambda x: x[0], levels)
ep = EnvParams(); env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, None, ep, sp, auto_reset=False)
def load(f):
    d = dict(np.load(f))
    if "embed.W" in d: return [(d["embed.W"], d["embed.b"])] + [(d[f"actor{i}.W"], d[f"actor{i}.b"]) for i in range(5)] + [(d["actor_head.W"], d["actor_head.b"])]
    g = "params/GeneralActorCriticRNN_0/"
    return [(d["params/Dense_0/kernel"], d["params/Dense_0/bias"])] + \
           [(d[f"{g}DenseAndActivation_{i}/Dense_0/kernel"], d[f"{g}DenseAndActivation_{i}/Dense_0/bias"]) for i in range(5)] + \
           [(d[f"{g}Dense_0/kernel"], d[f"{g}Dense_0/bias"])]
SPLITS = [3, 3, 3, 3, 2, 2]
def logits(P, o):
    h = jax.nn.relu(o @ P[0][0] + P[0][1])
    for W, b in P[1:6]: h = jnp.tanh(h @ W + b)
    return h @ P[6][0] + P[6][1]
def run(P, n, key, greedy=False):
    keys = jax.random.split(key, n)
    obs, st = jax.vmap(lambda k: env.reset(k, ep, override_reset_state=lvl))(keys)
    done = jnp.zeros(n, bool); ret = jnp.zeros(n); length = jnp.zeros(n, int); minp = jnp.ones(n)
    step = jax.jit(jax.vmap(lambda k, s, a: env.step(k, s, a, ep)))
    for t in range(256):
        z = jax.vmap(lambda o: logits(P, o))(obs); parts = jnp.split(z, np.cumsum(SPLITS)[:-1], axis=1)
        key, *ks = jax.random.split(key, 2 + len(SPLITS))
        acts, pmax = [], jnp.ones(n)
        for p, k in zip(parts, ks):
            pr = jax.nn.softmax(p); pmax = pmax * pr.max(1)
            acts.append(p.argmax(1) if greedy else jax.random.categorical(k, p))
        a = jnp.stack(acts, 1).astype(jnp.int32)
        key, k2 = jax.random.split(key)
        obs, st2, r, d, info = step(jax.random.split(k2, n), st, a)
        live = ~done; ret = ret + r * live; length = length + live; minp = jnp.where(live, jnp.minimum(minp, pmax), minp)
        st = jax.tree.map(lambda new, old: jnp.where(live.reshape((-1,) + (1,) * (new.ndim - 1)), new, old), st2, st)
        done = done | d
    return np.asarray(ret), np.asarray(length), np.asarray(minp)
for f in sys.argv[1:]:
    P = [tuple(map(jnp.asarray, x)) for x in load(f)]
    ret, L, mp = run(P, 1024, jax.random.PRNGKey(0))
    print(f"{f.split('/')[-1]}: success(return>0.5) {np.mean(ret > 0.5):.3f}  distinct (return,len) outcomes {len(set(zip(np.round(ret,5), L)))}  "
          f"mean len {L.mean():.1f}  min per-step joint action prob along episode: median {np.median(mp):.4f}, min {mp.min():.4f}")
    gr, gl, _ = run(P, 1, jax.random.PRNGKey(1), greedy=True); print(f"   greedy: return {gr[0]:.4f} len {gl[0]}")
