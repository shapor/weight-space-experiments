# Is l/h2_car_ramp deterministic across resets/keys? Roll out a fixed random action sequence from different PRNG keys.
import jax, jax.numpy as jnp, numpy as np
from kinetix.environment.env import make_kinetix_env
from kinetix.environment.spaces import ActionType, ObservationType
from kinetix.util.saving import load_evaluation_levels
from kinetix.environment.env_state import EnvParams
levels, sp = load_evaluation_levels(["l/h2_car_ramp.json"])
lvl = jax.tree.map(lambda x: x[0], levels)
env = make_kinetix_env(ActionType.MULTI_DISCRETE, ObservationType.SYMBOLIC_FLAT, None, EnvParams(), sp, auto_reset=False)
ep = EnvParams()
acts = np.random.default_rng(0).integers(0, 2, size=(120,) + env.action_space(ep).shape)
outs = []
for k in range(4):
    key = jax.random.PRNGKey(1000 + k)
    obs, st = env.reset(key, ep, override_reset_state=lvl)
    print("obs dim", obs.shape, "action shape", env.action_space(ep).shape) if k == 0 else None
    tr = [np.asarray(obs)]
    for t in range(120):
        key, sk = jax.random.split(key)
        obs, st, r, d, info = env.step(sk, st, jnp.asarray(acts[t]), ep); tr.append(np.asarray(obs))
    outs.append(np.stack(tr))
print("max |diff| between rollouts with different keys:", max(float(np.abs(outs[0] - o).max()) for o in outs[1:]))
