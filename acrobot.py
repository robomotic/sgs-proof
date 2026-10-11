"""GPU-batched torque-limited Acrobot swing-up for the SGS experiments (used via `sgs_maze.py --task acrobot`).

Dynamics follow Sutton & Barto / Gymnasium Acrobot (theta1 = 0 hangs down, only the elbow is actuated), but with a
continuous torque and RK4 integration. A task configuration is (initial state, torque limit, link-2 mass): high torque
or a near-upright start is easy; low torque from hanging down needs several energy-pumping swings, or is impossible.
Success = tip held above `height` (max 2.0) for `hold` consecutive steps.
"""
import math

import numpy as np
import torch

L1, L2, LC1, LC2, M1, I1, I2, G = 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 9.8
MAX_W1, MAX_W2 = 4 * math.pi, 9 * math.pi


def make_configs(num, rng, tau=(0.5, 3.0), m2=(0.5, 1.5)):
    """Columns: theta1, theta2, w1, w2, torque limit, m2. Returns (configs [num, 6], difficulty proxy = torque limit)."""
    c = np.stack([rng.uniform(-math.pi, math.pi, num), rng.uniform(-math.pi, math.pi, num),
                  rng.uniform(-1, 1, num), rng.uniform(-1, 1, num),
                  rng.uniform(*tau, num), rng.uniform(*m2, num)], 1).astype(np.float32)
    return c, c[:, 4]


def accel(q, a, m2):
    """Angular accelerations for state q = [th1, th2, w1, w2] and elbow torque a (book version of the dynamics)."""
    th1, th2, w1, w2 = q.unbind(1)
    d1 = M1 * LC1 ** 2 + m2 * (L1 ** 2 + LC2 ** 2 + 2 * L1 * LC2 * torch.cos(th2)) + I1 + I2
    d2 = m2 * (LC2 ** 2 + L1 * LC2 * torch.cos(th2)) + I2
    phi2 = m2 * LC2 * G * torch.cos(th1 + th2 - math.pi / 2)
    phi1 = (-m2 * L1 * LC2 * w2 ** 2 * torch.sin(th2) - 2 * m2 * L1 * LC2 * w2 * w1 * torch.sin(th2)
            + (M1 * LC1 + m2 * L1) * G * torch.cos(th1 - math.pi / 2) + phi2)
    dd2 = (a + d2 / d1 * phi1 - m2 * L1 * LC2 * w1 ** 2 * torch.sin(th2) - phi2) / (m2 * LC2 ** 2 + I2 - d2 ** 2 / d1)
    dd1 = -(d2 * dd2 + phi1) / d1
    return torch.stack([w1, w2, dd1, dd2], 1)


def energy(q, m2):
    """Total mechanical energy, for the conservation self-check."""
    th1, th2, w1, w2 = q.unbind(1)
    d1 = M1 * LC1 ** 2 + m2 * (L1 ** 2 + LC2 ** 2 + 2 * L1 * LC2 * torch.cos(th2)) + I1 + I2
    d2 = m2 * (LC2 ** 2 + L1 * LC2 * torch.cos(th2)) + I2
    ke = 0.5 * (d1 * w1 ** 2 + 2 * d2 * w1 * w2 + (m2 * LC2 ** 2 + I2) * w2 ** 2)
    pe = -M1 * G * LC1 * torch.cos(th1) - m2 * G * (L1 * torch.cos(th1) + LC2 * torch.cos(th1 + th2))
    return ke + pe


def tip_height(q):
    return -L1 * torch.cos(q[:, 0]) - L2 * torch.cos(q[:, 0] + q[:, 1])


class AcrobotEnv:
    obs_dim, act_dim = 8, 1

    def __init__(self, configs, num_envs, horizon, dt, height, hold, device):
        self.configs = torch.as_tensor(configs, device=device)
        self.horizon, self.dt, self.height, self.hold = horizon, dt, height, hold
        self.q = torch.zeros(num_envs, 4, device=device)
        self.tau = torch.zeros(num_envs, device=device)
        self.m2 = torch.ones(num_envs, device=device)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.up = torch.zeros(num_envs, dtype=torch.long, device=device)  # consecutive steps above `height`
        self.cfg = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids, cfg_ids):
        c = self.configs[cfg_ids]
        self.cfg[env_ids] = cfg_ids
        self.q[env_ids], self.tau[env_ids], self.m2[env_ids] = c[:, :4], c[:, 4], c[:, 5]
        self.t[env_ids] = self.up[env_ids] = 0

    def obs(self):
        th1, th2, w1, w2 = self.q.unbind(1)
        return torch.stack([torch.cos(th1), torch.sin(th1), torch.cos(th2), torch.sin(th2),
                            w1 / MAX_W1, w2 / MAX_W2, self.tau / 3, self.m2], 1)

    def step(self, action):
        a = action[:, 0].clamp(-1, 1)
        torque, q, h, m2 = a * self.tau, self.q, self.dt, self.m2
        k1 = accel(q, torque, m2)
        k2 = accel(q + h / 2 * k1, torque, m2)
        k3 = accel(q + h / 2 * k2, torque, m2)
        k4 = accel(q + h * k3, torque, m2)
        q = q + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q[:, :2] = torch.remainder(q[:, :2] + math.pi, 2 * math.pi) - math.pi
        q[:, 2] = q[:, 2].clamp(-MAX_W1, MAX_W1)
        q[:, 3] = q[:, 3].clamp(-MAX_W2, MAX_W2)
        self.q = q
        self.up = torch.where(tip_height(q) >= self.height, self.up + 1, 0)
        self.t += 1
        success = self.up >= self.hold
        done = success | (self.t >= self.horizon)
        reward = success.float() - 1e-3 * a ** 2
        return reward, done, success


def selftest():
    rng = np.random.default_rng(0)
    cfg, _ = make_configs(256, rng)
    cfg[:, 2:4] = 0.3 * cfg[:, 2:4]  # stay below the velocity clamps so energy is conserved
    env = AcrobotEnv(cfg, 256, 1000, 0.01, 1.8, 10, "cpu")
    env.reset(torch.arange(256), torch.arange(256))
    e0 = energy(env.q, env.m2)
    for _ in range(200):  # 2 s with zero torque: RK4 should conserve energy (angle wrapping doesn't change it)
        env.step(torch.zeros(256, 1))
    drift = ((energy(env.q, env.m2) - e0).abs() / (e0.abs() + 1)).max().item()
    assert drift < 1e-2, f"energy drift {drift}"
    up = AcrobotEnv(np.array([[math.pi, 0, 0, 0, 1, 1]], np.float32), 1, 100, 0.05, 1.8, 3, "cpu")
    up.reset(torch.tensor([0]), torch.tensor([0]))
    assert abs(tip_height(up.q).item() - 2.0) < 1e-6
    assert [up.step(torch.zeros(1, 1))[2].item() for _ in range(3)] == [False, False, True]  # upright counts as held
    print(f"acrobot selftest ok (energy drift {drift:.1e})")


if __name__ == "__main__":
    selftest()
