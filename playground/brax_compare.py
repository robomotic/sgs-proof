"""Stage A: does "SAC beats PPO" survive outside the maze? Standard Brax PPO vs Brax SAC on MuJoCo Playground PandaPickCube.

PPO uses Playground's tuned config; SAC uses Playground's tuned SAC config (from its DM Control tasks) with the same
discount as PPO. Both train on the task's own shaped reward. Success (not part of the task) = cube within 5 cm of the
target. Logged two ways: during training, Brax's evaluator reports `eval/episode_success` = steps within 5 cm per
episode; after training, the success rate at the final step over 1024 fresh episodes.
    python brax_compare.py --algo ppo --seed 0
    python brax_compare.py --algo sac --seed 0
"""
import argparse, functools, json, os, time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
import jax.numpy as jp
import numpy as np
from brax.io import model as brax_model
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from brax.training.agents.sac import networks as sac_networks, train as sac
from mujoco_playground import registry, wrapper
from mujoco_playground._src.manipulation.franka_emika_panda import pick
from mujoco_playground.config import manipulation_params

SUCCESS_RADIUS = 0.05  # m


class PickWithSuccess(pick.PandaPickCube):
    """PandaPickCube plus a per-step `success` metric (cube within SUCCESS_RADIUS of the target)."""

    def _success(self, state):
        err = jp.linalg.norm(state.info["target_pos"] - state.data.xpos[self._obj_body])
        return (err < SUCCESS_RADIUS).astype(float)

    def reset(self, rng):
        state = super().reset(rng)
        return state.replace(metrics={**state.metrics, "success": jp.zeros(())})

    def step(self, state, action):
        state = super().step(state, action)
        return state.replace(metrics={**state.metrics, "success": self._success(state)})


def final_success(env, make_inference_fn, params, n=1024, seed=12345):
    """Success rate at the last step of the episode (deterministic policy, fixed eval seeds)."""
    policy = make_inference_fn(params, deterministic=True)
    reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
    act = jax.jit(jax.vmap(lambda o, k: policy(o, k)[0]))
    state = reset(jax.random.split(jax.random.PRNGKey(seed), n))
    ever = jp.zeros(n)
    for t in range(env._config.episode_length):
        state = step(state, act(state.obs, jax.random.split(jax.random.PRNGKey(t), n)))
        ever = jp.maximum(ever, state.metrics["success"])
    return float(state.metrics["success"].mean()), float(ever.mean())


def save_rollouts(env, make_inference_fn, params, n, path, seed=7):
    """Play n episodes with the deterministic policy; save per-step qpos and target pose for rendering."""
    policy = jax.jit(make_inference_fn(params, deterministic=True))
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    qpos, mpos, mquat, succ = [], [], [], []
    for ep in range(n):
        state = reset(jax.random.PRNGKey(seed + ep))
        q, mp, mq, sc = [state.data.qpos], [state.data.mocap_pos], [state.data.mocap_quat], [state.metrics["success"]]
        for t in range(env._config.episode_length):
            state = step(state, policy(state.obs, jax.random.PRNGKey(t))[0])
            q.append(state.data.qpos), mp.append(state.data.mocap_pos), mq.append(state.data.mocap_quat)
            sc.append(state.metrics["success"])
        qpos.append(np.stack(q)), mpos.append(np.stack(mp)), mquat.append(np.stack(mq)), succ.append(np.stack(sc))
    np.savez(path, qpos=np.stack(qpos), mocap_pos=np.stack(mpos), mocap_quat=np.stack(mquat), success=np.stack(succ))


def main(args):
    registry.load("PandaPickCube")  # fetches the robot assets on first use
    cfg = registry.get_default_config("PandaPickCube")
    cfg.impl = "jax"  # classic MJX backend; the default MuJoCo Warp backend needs CUDA driver 13 (this machine has 12.8)
    env, eval_env = PickWithSuccess(config=cfg), PickWithSuccess(config=cfg)
    log, t0 = [], time.time()

    def progress(step, metrics):
        row = dict(env_steps=int(step), time=round(time.time() - t0, 1),
                   reward=float(metrics.get("eval/episode_reward", float("nan"))),
                   steps_at_target=float(metrics.get("eval/episode_success", float("nan"))))
        log.append(row)
        print(f"[{args.algo} s={args.seed}] {row['env_steps'] / 1e6:6.1f}M  reward {row['reward']:8.1f}  "
              f"steps-within-5cm {row['steps_at_target']:6.1f}/{cfg.episode_length}  {row['time']:.0f}s", flush=True)

    if args.algo == "ppo":
        p = manipulation_params.brax_ppo_config("PandaPickCube", "jax").to_dict()
        net = functools.partial(ppo_networks.make_ppo_networks, **p.pop("network_factory"))
        train = functools.partial(ppo.train, **p, network_factory=net)
    else:
        p = dict(num_timesteps=20_000_000, num_evals=10, reward_scaling=1.0, episode_length=cfg.episode_length,
                 normalize_observations=True, action_repeat=1, discounting=0.97,  # same discount as tuned PPO
                 learning_rate=1e-3, num_envs=128, batch_size=512, grad_updates_per_step=8,
                 max_replay_size=1048576 * 4, min_replay_size=8192)
        net = functools.partial(sac_networks.make_sac_networks, q_network_layer_norm=True)
        train = functools.partial(sac.train, **p, network_factory=net)
    if args.steps:
        train = functools.partial(train, num_timesteps=args.steps)
    if args.num_evals:
        train = functools.partial(train, num_evals=args.num_evals)

    make_inference_fn, params, _ = train(environment=env, eval_env=eval_env, seed=args.seed, progress_fn=progress,
                                         wrap_env_fn=wrapper.wrap_for_brax_training)
    train_time = time.time() - t0
    succ_final, succ_ever = final_success(eval_env, make_inference_fn, params)
    print(f"[{args.algo} s={args.seed}] FINAL success at last step {succ_final:.3f}  (ever within 5cm {succ_ever:.3f})  "
          f"train time {train_time / 60:.1f} min", flush=True)
    os.makedirs(args.out, exist_ok=True)
    brax_model.save_params(f"{args.out}/{args.algo}_s{args.seed}.params", params)
    if args.save_rollout:
        save_rollouts(eval_env, make_inference_fn, params, args.save_rollout, f"{args.out}/{args.algo}_s{args.seed}_rollout.npz")
    with open(f"{args.out}/{args.algo}_s{args.seed}.json", "w") as f:
        json.dump(dict(args=vars(args), config={k: str(v) for k, v in p.items()}, log=log, train_time=train_time,
                       success_final=succ_final, success_ever=succ_ever), f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)  # override num_timesteps (0 = config default)
    ap.add_argument("--num-evals", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--save-rollout", type=int, default=0)  # episodes to record for rendering (0 = none)
    main(ap.parse_args())
