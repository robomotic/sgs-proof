"""Success Guided Sampling (SGS) vs uniform sampling on a GPU-batched 2D point-mass maze.

One fixed maze; a task configuration is (start position, goal cell). PPO, sparse success reward.
    python sgs_maze.py --sampler sgs --num-envs 4096 --seed 0
    python sgs_maze.py --selftest
"""
import argparse, json, os, time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- maze

def make_maze(cells, braid, rng):
    """DFS perfect maze on a (2*cells+1)^2 grid (True = wall), then knock out a `braid` fraction of inner walls."""
    n = 2 * cells + 1
    wall = np.ones((n, n), bool)
    wall[1, 1] = False
    stack, seen = [(0, 0)], {(0, 0)}
    while stack:
        cy, cx = stack[-1]
        nbrs = [(cy + dy, cx + dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= cy + dy < cells and 0 <= cx + dx < cells and (cy + dy, cx + dx) not in seen]
        if not nbrs:
            stack.pop()
            continue
        ny, nx = nbrs[rng.integers(len(nbrs))]
        wall[2 * ny + 1, 2 * nx + 1] = wall[cy + ny + 1, cx + nx + 1] = False
        seen.add((ny, nx))
        stack.append((ny, nx))
    inner = [(y, x) for y in range(1, n - 1) for x in range(1, n - 1) if wall[y, x] and (y % 2) != (x % 2)]
    for i in rng.permutation(len(inner))[: int(braid * len(inner))]:
        wall[inner[i]] = False
    return wall


def all_pairs_dist(wall):
    """BFS grid distance between every pair of open cells, -1 if unreachable. Indexed by flat cell id."""
    n = wall.shape[0]
    D = np.full((n * n, n * n), -1, np.int32)
    for s in np.flatnonzero(~wall.ravel()):
        D[s, s] = 0
        q = deque([s])
        while q:
            c = q.popleft()
            y, x = divmod(c, n)
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                nc = ny * n + nx
                if not wall[ny, nx] and D[s, nc] < 0:
                    D[s, nc] = D[s, c] + 1
                    q.append(nc)
    return D


def make_configs(wall, D, num, max_dist, rng):
    """Sample `num` (start_xy, goal_xy, grid_dist) configs with 1 <= dist <= max_dist. Start jittered inside its cell."""
    n = wall.shape[0]
    s, g = np.nonzero((D >= 1) & (D <= max_dist))
    pick = rng.integers(len(s), size=num)
    s, g = s[pick], g[pick]
    start = np.stack([s % n, s // n], 1) + rng.uniform(0.2, 0.8, (num, 2))
    goal = np.stack([g % n, g // n], 1) + 0.5
    return start.astype(np.float32), goal.astype(np.float32), D[s, g]


# ---------------------------------------------------------------- env

class MazeEnv:
    """Point mass; action in [-1,1]^2 scaled by `speed`, axis-separated wall collision. Obs: pos, goal, delta, 5x5 wall patch."""
    obs_dim, act_dim = 6 + 25, 2

    def __init__(self, wall, starts, goals, num_envs, horizon, speed, device):
        self.wall = torch.as_tensor(wall, device=device)
        self.wpad = F.pad(self.wall, (2, 2, 2, 2), value=True)
        self.n = wall.shape[0]
        self.starts, self.goals = torch.as_tensor(starts, device=device), torch.as_tensor(goals, device=device)
        self.horizon, self.speed = horizon, speed
        self.pos = torch.zeros(num_envs, 2, device=device)
        self.goal = torch.zeros(num_envs, 2, device=device)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cfg = torch.zeros(num_envs, dtype=torch.long, device=device)
        off = torch.arange(5, device=device)
        self.oy, self.ox = off[None, :, None], off[None, None, :]

    def reset(self, env_ids, cfg_ids):
        self.cfg[env_ids] = cfg_ids
        self.pos[env_ids] = self.starts[cfg_ids]
        self.goal[env_ids] = self.goals[cfg_ids]
        self.t[env_ids] = 0

    def is_wall(self, xy):
        return self.wall[xy[:, 1].long(), xy[:, 0].long()]

    def obs(self):
        ij = self.pos.long()
        patch = self.wpad[ij[:, 1, None, None] + self.oy, ij[:, 0, None, None] + self.ox]
        return torch.cat([self.pos / self.n * 2 - 1, self.goal / self.n * 2 - 1,
                          (self.goal - self.pos) / self.n, patch.flatten(1).float()], 1)

    def step(self, action):
        a = action.clamp(-1, 1)
        d = a * self.speed
        for axis in (0, 1):  # move x then y, undo the axis that would enter a wall
            new = self.pos.clone()
            new[:, axis] += d[:, axis]
            self.pos = torch.where(self.is_wall(new)[:, None], self.pos, new)
        self.t += 1
        success = (self.pos - self.goal).norm(dim=1) < 0.5
        done = success | (self.t >= self.horizon)
        reward = success.float() - 1e-3 * (a ** 2).sum(1)
        return reward, done, success


# ---------------------------------------------------------------- sampler

class Sampler:
    """Keeps the last H outcomes per config. 'uniform' ignores them; 'sgs' samples softmax(log(Beta-kernel(p)+eps)/T);
    'plr' is Prioritized Level Replay (Jiang et al. 2021) with mean |GAE| as the score, as in the SGS paper."""
    plr_beta, plr_rho, plr_replay = 0.1, 0.1, 0.5  # rank temperature, staleness mix, replay prob (PLR defaults)

    def __init__(self, N, H, mode, target, kappa, eps, temp, device):
        self.N, self.H, self.mode = N, H, mode
        self.target, self.kappa, self.eps, self.temp = target, kappa, eps, temp
        self.hist = torch.zeros(N, H, dtype=torch.bool, device=device)
        self.ptr = torch.zeros(N, dtype=torch.long, device=device)
        self.score = torch.zeros(N, device=device)  # PLR only
        self.last = torch.zeros(N, device=device)
        self.seen = torch.zeros(N, dtype=torch.bool, device=device)
        self.clock = 0

    def update_scores(self, idx, values):
        # ponytail: score = mean |GAE| of the config's steps in the latest rollout, not over whole episodes
        s = torch.zeros(self.N, device=values.device).index_add_(0, idx, values)
        c = torch.zeros(self.N, device=values.device).index_add_(0, idx, torch.ones_like(values))
        m = c > 0
        self.score[m] = s[m] / c[m]
        self.seen |= m

    def plr_sample(self, n):
        dev = self.score.device
        seen, unseen = self.seen.nonzero().squeeze(1), (~self.seen).nonzero().squeeze(1)
        n_new = n if len(seen) == 0 else 0 if len(unseen) == 0 else int(torch.binomial(
            torch.tensor(float(n)), torch.tensor(1 - self.plr_replay)).item())
        out = [unseen[torch.randint(len(unseen), (n_new,), device=dev)]] if n_new else []
        if n - n_new:
            rank = torch.empty(len(seen), device=dev)
            rank[torch.argsort(self.score[seen], descending=True)] = torch.arange(1, len(seen) + 1, device=dev, dtype=torch.float)
            p_score = rank ** (-1 / self.plr_beta)
            stale = self.clock - self.last[seen]
            p_stale = stale / stale.sum() if stale.sum() > 0 else torch.full_like(stale, 1 / len(seen))
            p = (1 - self.plr_rho) * p_score / p_score.sum() + self.plr_rho * p_stale
            out.append(seen[torch.multinomial(p, n - n_new, replacement=True)])
        idx = torch.cat(out)
        self.clock += n
        self.last[idx] = self.clock
        return idx

    def update(self, idx, succ):
        # several envs can finish the same config in one step: give each its own slot in the ring buffer
        order = torch.argsort(idx, stable=True)
        idx, succ = idx[order], succ[order]
        _, counts = torch.unique_consecutive(idx, return_counts=True)
        group_start = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
        rank = torch.arange(len(idx), device=idx.device) - group_start
        self.hist[idx, (self.ptr[idx] + rank) % self.H] = succ
        self.ptr.index_add_(0, idx, torch.ones_like(idx))

    def success_rate(self):
        return self.hist.sum(1) / self.ptr.clamp(1, self.H)

    def probs(self):
        p, k, t, e = self.success_rate(), self.kappa, self.target, self.eps
        w = (p + e) ** (k * t) * (1 - p + e) ** (k * (1 - t))
        return torch.softmax(torch.log(w + e) / self.temp, 0)

    def sample(self, n):
        if self.mode == "uniform":
            return torch.randint(self.N, (n,), device=self.hist.device)
        if self.mode == "plr":
            return self.plr_sample(n)
        return torch.multinomial(self.probs(), n, replacement=True)


# ---------------------------------------------------------------- PPO

def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o))


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor, self.critic = mlp(obs_dim, act_dim), mlp(obs_dim, 1)
        self.logstd = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.logstd.exp())


@torch.no_grad()
def evaluate(agent, env, num):
    ids = torch.arange(num, device=env.cfg.device)
    env.reset(ids, ids)
    solved = torch.zeros(num, dtype=torch.bool, device=env.cfg.device)
    for _ in range(env.horizon):
        _, _, succ = env.step(agent.actor(env.obs()))
        solved |= succ
    return solved


def train(args):
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.maze_seed)  # same task + config pool across runs; only training seed varies
    E, S = args.num_envs, args.num_steps
    if args.task == "maze":
        wall = make_maze(args.cells, args.braid, rng)
        D = all_pairs_dist(wall)
        starts, goals, _ = make_configs(wall, D, args.num_configs, args.max_dist, rng)
        ev_starts, ev_goals, ev_diff = make_configs(wall, D, args.eval_episodes, args.max_dist, rng)  # fresh configs
        env = MazeEnv(wall, starts, goals, E, args.horizon, args.speed, dev)
        eval_env = MazeEnv(wall, ev_starts, ev_goals, args.eval_episodes, args.horizon, args.speed, dev)
        bin_name, bins = "dist", [(1, 20), (21, 40), (41, 60), (61, args.max_dist)]
    else:
        import acrobot
        tau = (args.tau_min, 3.0)
        cfgs, _ = acrobot.make_configs(args.num_configs, rng, tau)
        ev_cfgs, ev_diff = acrobot.make_configs(args.eval_episodes, rng, tau)
        env = acrobot.AcrobotEnv(cfgs, E, args.horizon, args.dt, args.height, args.hold, dev)
        eval_env = acrobot.AcrobotEnv(ev_cfgs, args.eval_episodes, args.horizon, args.dt, args.height, args.hold, dev)
        bin_name, bins = "torque", [(args.tau_min, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]  # low torque = hard
    sampler = Sampler(args.num_configs, args.hist, args.sampler, args.target, args.kappa, args.eps, args.temp, dev)
    sampler.plr_beta = args.plr_beta
    env.reset(torch.arange(E, device=dev), sampler.sample(E))

    agent = Agent(env.obs_dim, env.act_dim).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr)
    buf_obs = torch.zeros(S, E, env.obs_dim, device=dev)
    buf_act = torch.zeros(S, E, env.act_dim, device=dev)
    buf_logp, buf_rew, buf_done, buf_val = (torch.zeros(S, E, device=dev) for _ in range(4))
    buf_cfg = torch.zeros(S, E, dtype=torch.long, device=dev)

    ev_diff_t = torch.as_tensor(ev_diff, device=dev)
    log, t0 = [], time.time()
    obs = env.obs()
    for it in range(1, args.iters + 1):
        ep_succ = []
        for s in range(S):
            with torch.no_grad():
                d = agent.dist(obs)
                a = d.sample()
                buf_val[s] = agent.critic(obs).squeeze(1)
            buf_obs[s], buf_act[s], buf_logp[s], buf_cfg[s] = obs, a, d.log_prob(a).sum(1), env.cfg
            rew, done, succ = env.step(a)
            buf_rew[s], buf_done[s] = rew, done.float()
            ids = done.nonzero().squeeze(1)
            if len(ids):
                sampler.update(env.cfg[ids], succ[ids])
                ep_succ.append(succ[ids].float())
                env.reset(ids, sampler.sample(len(ids)))
            obs = env.obs()

        # GAE. ponytail: timeouts treated as terminal (no bootstrap); obs has no time feature anyway.
        with torch.no_grad():
            next_v = agent.critic(obs).squeeze(1)
            adv, last = torch.zeros_like(buf_rew), 0
            for s in reversed(range(S)):
                nv = next_v if s == S - 1 else buf_val[s + 1]
                nonterm = 1 - buf_done[s]
                delta = buf_rew[s] + args.gamma * nv * nonterm - buf_val[s]
                last = delta + args.gamma * args.lam * nonterm * last
                adv[s] = last
            ret = adv + buf_val
        if args.sampler == "plr":
            sampler.update_scores(buf_cfg.flatten(), adv.abs().flatten())

        b_obs, b_act, b_logp = buf_obs.flatten(0, 1), buf_act.flatten(0, 1), buf_logp.flatten(0, 1)
        b_adv, b_ret = adv.flatten(), ret.flatten()
        mb = len(b_obs) // args.num_minibatches  # fixed minibatch count: minibatch size grows with num_envs (as in the paper)
        for _ in range(args.epochs):
            perm = torch.randperm(len(b_obs), device=dev)
            for j in range(args.num_minibatches):
                i = perm[j * mb:(j + 1) * mb]
                d = agent.dist(b_obs[i])
                ratio = (d.log_prob(b_act[i]).sum(1) - b_logp[i]).exp()
                A = (b_adv[i] - b_adv[i].mean()) / (b_adv[i].std() + 1e-8)
                pg = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
                vf = 0.5 * (agent.critic(b_obs[i]).squeeze(1) - b_ret[i]).pow(2).mean()
                loss = pg + vf - args.ent * d.entropy().sum(1).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()

        if it % args.eval_every == 0 or it == args.iters:
            solved = evaluate(agent, eval_env, args.eval_episodes)
            p = sampler.success_rate()
            row = dict(iter=it, env_steps=it * S * E, time=round(time.time() - t0, 1),
                       eval_success=solved.float().mean().item(),
                       **{f"eval_by_{bin_name}": {f"{lo}-{hi}": solved[(ev_diff_t >= lo) & (ev_diff_t <= hi)].float().mean().item()
                                                  for lo, hi in bins}},
                       train_success=torch.cat(ep_succ).mean().item() if ep_succ else 0.0,
                       frac_frontier=((p > 0.05) & (p < 0.95)).float().mean().item(),
                       frac_mastered=(p >= 0.95).float().mean().item())
            log.append(row)
            print(f"[{args.sampler} E={E} s={args.seed}] it {it:5d}  eval {row['eval_success']:.3f}  "
                  f"by-{bin_name} {' '.join(f'{v:.2f}' for v in row[f'eval_by_{bin_name}'].values())}  "
                  f"frontier {row['frac_frontier']:.2f}  mastered {row['frac_mastered']:.2f}  {row['time']:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{args.sampler}_E{E}_s{args.seed}.json", "w") as f:  # one --out dir per task
        json.dump(dict(args=vars(args), log=log), f)


# ---------------------------------------------------------------- check

def selftest():
    rng = np.random.default_rng(0)
    wall = make_maze(6, 0.1, rng)
    D = all_pairs_dist(wall)
    open_ids = np.flatnonzero(~wall.ravel())
    assert (D[np.ix_(open_ids, open_ids)] >= 0).all(), "maze must be connected"
    starts, goals, _ = make_configs(wall, D, 512, 30, rng)
    env = MazeEnv(wall, starts, goals, 512, 50, 0.5, "cpu")
    env.reset(torch.arange(512), torch.arange(512))
    for _ in range(200):  # random actions never put the agent inside a wall
        env.step(torch.randn(512, 2) * 3)
        assert not env.is_wall(env.pos).any()
    s = Sampler(4, 3, "sgs", 0.5, 1.0, 1e-4, 2.0, "cpu")
    s.update(torch.tensor([2, 2, 2, 2, 0]), torch.tensor([True, True, False, True, True]))
    assert s.ptr.tolist() == [1, 0, 4, 0] and s.hist[2].sum() == 2  # 4th write wrapped over the 1st
    pr = s.probs()
    assert pr[0] < pr[2] and abs(pr.sum().item() - 1) < 1e-5  # p=2/3 beats p=1 at target 0.5
    s = Sampler(100, 3, "plr", 0, 0, 0, 1, "cpu")
    s.update_scores(torch.arange(50), torch.arange(50).float())  # configs 0-49 seen, 49 scores highest
    idx = s.sample(4000)
    assert 0.4 < (idx >= 50).float().mean() < 0.6  # ~half new levels
    assert (idx == 49).float().mean() > 0.3  # replay concentrates on the top-ranked config
    s.update_scores(torch.arange(100), torch.ones(100))
    assert len(s.sample(10)) == 10  # all configs seen: replay only
    import acrobot
    acrobot.selftest()
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=["sgs", "uniform", "plr"], default="sgs")
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--num-steps", type=int, default=32)
    ap.add_argument("--num-minibatches", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--task", choices=["maze", "acrobot"], default="maze")
    # maze
    ap.add_argument("--cells", type=int, default=10)
    ap.add_argument("--braid", type=float, default=0.1)
    ap.add_argument("--maze-seed", type=int, default=0)
    ap.add_argument("--max-dist", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--speed", type=float, default=0.5)
    # acrobot (--horizon also applies; acrobot runs should pass e.g. --horizon 400)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--height", type=float, default=1.8)
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--tau-min", type=float, default=0.5)
    # SGS (paper's locomotion values)
    ap.add_argument("--num-configs", type=int, default=32768)
    ap.add_argument("--hist", type=int, default=100)
    ap.add_argument("--target", type=float, default=0.66)
    ap.add_argument("--kappa", type=float, default=5.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--plr-beta", type=float, default=0.1)  # PLR default; ~90% of resets hit one config at this scale
    # eval / io
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-episodes", type=int, default=4096)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    selftest() if args.selftest else train(args)
