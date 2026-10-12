"""SAC + Hindsight Experience Replay (HER) on the same GPU-batched maze as ../sgs_maze.py.

Same maze, config pool, eval set and success definition as the PPO runs, so results compare directly
(x-axis: env steps). Config sampling at reset can be uniform or SGS.
    python her/sac_her.py --sampler uniform --her-prob 0.8      # SAC + HER
    python her/sac_her.py --sampler uniform --her-prob 0         # plain SAC
    python her/sac_her.py --selftest
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sgs_maze import MazeEnv, Sampler, all_pairs_dist, evaluate, make_configs, make_maze, mlp  # noqa: E402


# ---------------------------------------------------------------- goal relabelling
# MazeEnv obs layout: [0:2] pos/n*2-1, [2:4] goal/n*2-1, [4:6] (goal-pos)/n, [6:31] wall patch (goal-independent)

def obs_pos(o, n):
    return (o[:, 0:2] + 1) / 2 * n


def obs_goal(o, n):
    return (o[:, 2:4] + 1) / 2 * n


def with_goal(o, g, n):
    o = o.clone()
    o[:, 2:4] = g / n * 2 - 1
    o[:, 4:6] = (g - obs_pos(o, n)) / n
    return o


class EpisodeBuffer:
    """Whole episodes on the GPU. Each env writes its running episode into a staging row; finished episodes are
    copied into a ring of `cap` episodes, so HER can pick goals from later in the same episode."""

    def __init__(self, cap, num_envs, horizon, obs_dim, act_dim, dev):
        self.obs = torch.zeros(cap, horizon + 1, obs_dim, device=dev)
        self.act = torch.zeros(cap, horizon, act_dim, device=dev)
        self.len = torch.zeros(cap, dtype=torch.long, device=dev)
        self.st_obs = torch.zeros(num_envs, horizon + 1, obs_dim, device=dev)
        self.st_act = torch.zeros(num_envs, horizon, act_dim, device=dev)
        self.cap, self.ptr, self.size = cap, 0, 0

    def finish(self, ids, lengths):
        slots = (self.ptr + torch.arange(len(ids), device=ids.device)) % self.cap
        self.obs[slots], self.act[slots], self.len[slots] = self.st_obs[ids], self.st_act[ids], lengths
        self.ptr = (self.ptr + len(ids)) % self.cap
        self.size = min(self.size + len(ids), self.cap)

    def sample(self, B, her_prob, n, dev):
        i = torch.randint(self.size, (B,), device=dev)
        L = self.len[i]
        t = (torch.rand(B, device=dev) * L).long()
        o, o2, a = self.obs[i, t], self.obs[i, t + 1], self.act[i, t]
        # HER "future": with prob her_prob, the goal becomes a position actually reached later in the same episode
        t_fut = t + 1 + (torch.rand(B, device=dev) * (L - t)).long().clamp(max=L - t - 1)
        her = torch.rand(B, device=dev) < her_prob
        g = torch.where(her[:, None], obs_pos(self.obs[i, t_fut], n), obs_goal(o, n))
        o, o2 = with_goal(o, g, n), with_goal(o2, g, n)
        success = (obs_pos(o2, n) - g).norm(dim=1) < 0.5  # same success rule as MazeEnv
        r = success.float() - 1e-3 * (a ** 2).sum(1)
        return o, a, r, o2, success.float()


# ---------------------------------------------------------------- SAC

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim)

    def forward(self, o):
        mu, log_std = self.net(o).chunk(2, -1)
        std = log_std.clamp(-5, 2).exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = torch.distributions.Normal(mu, std).log_prob(u).sum(1) - torch.log(1 - a ** 2 + 1e-6).sum(1)
        return a, logp

    def mean_action(self, o):  # deterministic, for evaluation
        return torch.tanh(self.net(o).chunk(2, -1)[0])


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q1, self.q2 = mlp(obs_dim + act_dim, 1), mlp(obs_dim + act_dim, 1)

    def forward(self, o, a):
        x = torch.cat([o, a], 1)
        return self.q1(x).squeeze(1), self.q2(x).squeeze(1)


class Policy:  # adapter so sgs_maze.evaluate() can call .actor(obs)
    def __init__(self, actor):
        self.actor = actor.mean_action


def train(args):
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.maze_seed)  # identical maze, pool and eval set to sgs_maze.py
    wall = make_maze(args.cells, args.braid, rng)
    D = all_pairs_dist(wall)
    starts, goals, _ = make_configs(wall, D, args.num_configs, args.max_dist, rng, args.min_dist)
    ev_starts, ev_goals, ev_dist = make_configs(wall, D, args.eval_episodes, args.max_dist, rng, args.min_dist)
    E, n = args.num_envs, wall.shape[0]
    env = MazeEnv(wall, starts, goals, E, args.horizon, args.speed, dev)
    eval_env = MazeEnv(wall, ev_starts, ev_goals, args.eval_episodes, args.horizon, args.speed, dev)
    sampler = Sampler(args.num_configs, 100, args.sampler, 0.66, 5.0, 1e-8, 2.0, dev)  # paper's locomotion SGS values

    od, ad = MazeEnv.obs_dim, MazeEnv.act_dim
    actor, critic, critic_t = Actor(od, ad).to(dev), Critic(od, ad).to(dev), Critic(od, ad).to(dev)
    critic_t.load_state_dict(critic.state_dict())
    log_alpha = torch.full((1,), float(np.log(args.alpha_init)), device=dev, requires_grad=True)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
    opt_al = torch.optim.Adam([log_alpha], lr=args.lr)
    buf = EpisodeBuffer(args.buffer_episodes, E, args.horizon, od, ad, dev)

    ar = torch.arange(E, device=dev)
    env.reset(ar, sampler.sample(E))
    buf.st_obs[:, 0] = env.obs()
    ev_dist_t = torch.as_tensor(ev_dist, device=dev)
    bins = [(1, 20), (21, 40), (41, 60)]
    log, t0, ep_succ = [], time.time(), []
    for step in range(1, args.vec_steps + 1):
        obs = env.obs()
        with torch.no_grad():
            a, _ = actor(obs)
        t_idx = env.t.clone()
        _, done, succ = env.step(a)
        buf.st_act[ar, t_idx] = a
        buf.st_obs[ar, t_idx + 1] = env.obs()  # next obs, before any reset
        ids = done.nonzero().squeeze(1)
        if len(ids):
            buf.finish(ids, env.t[ids])
            sampler.update(env.cfg[ids], succ[ids])
            ep_succ.append(succ[ids].float())
            env.reset(ids, sampler.sample(len(ids)))
            buf.st_obs[ids, 0] = env.obs()[ids]

        if buf.size >= args.learning_starts:
            for _ in range(args.updates_per_step):
                o, a_b, r, o2, d = buf.sample(args.batch, args.her_prob, n, dev)
                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    a2, logp2 = actor(o2)
                    q_t = torch.min(*critic_t(o2, a2)) - alpha * logp2
                    y = r + args.gamma * (1 - d) * q_t
                q1, q2 = critic(o, a_b)
                loss_c = (q1 - y).pow(2).mean() + (q2 - y).pow(2).mean()
                opt_c.zero_grad(); loss_c.backward(); opt_c.step()
                a_new, logp = actor(o)
                loss_a = (alpha * logp - torch.min(*critic(o, a_new))).mean()
                opt_a.zero_grad(); loss_a.backward(); opt_a.step()
                loss_al = -(log_alpha * (logp.detach() - ad)).mean()  # target entropy = -act_dim
                opt_al.zero_grad(); loss_al.backward(); opt_al.step()
                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), critic_t.parameters()):
                        pt.lerp_(p, args.tau)

        if step % args.eval_every == 0 or step == args.vec_steps:
            solved = evaluate(Policy(actor), eval_env, args.eval_episodes)
            row = dict(step=step, env_steps=step * E, time=round(time.time() - t0, 1),
                       eval_success=solved.float().mean().item(),
                       eval_by_dist={f"{lo}-{hi}": solved[(ev_dist_t >= lo) & (ev_dist_t <= hi)].float().mean().item()
                                     for lo, hi in bins},
                       train_success=torch.cat(ep_succ).mean().item() if ep_succ else 0.0,
                       alpha=log_alpha.exp().item())
            ep_succ = []
            log.append(row)
            print(f"[sac her={args.her_prob} {args.sampler} E={E} s={args.seed}] step {step:6d} "
                  f"({row['env_steps'] / 1e6:.0f}M)  eval {row['eval_success']:.3f}  "
                  f"by-dist {' '.join(f'{v:.2f}' for v in row['eval_by_dist'].values())}  "
                  f"train {row['train_success']:.2f}  alpha {row['alpha']:.3f}  {row['time']:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/sac_her{args.her_prob:g}_{args.sampler}_E{E}_s{args.seed}.json", "w") as f:
        json.dump(dict(args=vars(args), log=log), f)


def selftest():
    rng = np.random.default_rng(0)
    wall = make_maze(6, 0.1, rng)
    D = all_pairs_dist(wall)
    starts, goals, _ = make_configs(wall, D, 64, 30, rng)
    n, E, H = wall.shape[0], 64, 30
    env = MazeEnv(wall, starts, goals, E, H, 0.5, "cpu")
    ar = torch.arange(E)
    env.reset(ar, ar)
    o = env.obs()
    assert torch.allclose(with_goal(o, obs_goal(o, n), n), o, atol=1e-5)  # relabel with own goal is a no-op
    assert torch.allclose(obs_pos(o, n), env.pos, atol=1e-5)
    buf = EpisodeBuffer(16, E, H, MazeEnv.obs_dim, 2, "cpu")
    buf.st_obs[:, 0] = o
    for _ in range(H):  # random-walk until timeout, then store all episodes
        a = torch.rand(E, 2) * 2 - 1
        t_idx = env.t.clone()
        env.step(a)
        buf.st_act[ar, t_idx], buf.st_obs[ar, t_idx + 1] = a, env.obs()
    buf.finish(ar[:16], env.t[:16].clamp(max=H))
    _, _, r, _, d = buf.sample(4096, 1.0, n, "cpu")
    assert d.mean() > 0.2  # 'future' goals were actually reached later, so many relabelled steps succeed
    _, _, _, _, d0 = buf.sample(4096, 0.0, n, "cpu")
    assert d.mean() > d0.mean()
    print(f"her selftest ok (success with HER {d.mean():.2f} vs without {d0.mean():.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=["uniform", "sgs"], default="uniform")
    ap.add_argument("--her-prob", type=float, default=0.8)  # 0 = plain SAC
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vec-steps", type=int, default=12800)  # 12800 x 256 = 3.3M env steps
    ap.add_argument("--updates-per-step", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--alpha-init", type=float, default=0.01)  # entropy weight; 1.0 drowns the sparse reward
    ap.add_argument("--buffer-episodes", type=int, default=40000)
    ap.add_argument("--learning-starts", type=int, default=256)  # episodes in the buffer before updates begin
    ap.add_argument("--cells", type=int, default=8)
    ap.add_argument("--braid", type=float, default=0.1)
    ap.add_argument("--maze-seed", type=int, default=0)
    ap.add_argument("--max-dist", type=int, default=60)
    ap.add_argument("--min-dist", type=int, default=1)  # 41 = hard goals only (training pool and eval set)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--num-configs", type=int, default=32768)
    ap.add_argument("--eval-every", type=int, default=1600)
    ap.add_argument("--eval-episodes", type=int, default=4096)
    ap.add_argument("--out", default="her/runs")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    selftest() if args.selftest else train(args)
