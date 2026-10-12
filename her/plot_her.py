"""SAC (+/- HER, +/- SGS) vs PPO on the maze: eval success vs env steps and vs wall-clock (her/sac_vs_ppo.png).
Run from the repo root: python her/plot_her.py"""
import glob, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, glob, color, linestyle). PPO 64K: seed 2 excluded, its logged time includes a pause.
SERIES = [
    ("SAC, uniform", "her/runs/main/sac_her0_uniform_*.json", "#2a78d6", "-"),
    ("SAC, SGS", "her/runs/main/sac_her0_sgs_*.json", "#2a78d6", "--"),
    ("SAC + HER, uniform", "her/runs/main/sac_her0.8_uniform_*.json", "#1baf7a", "-"),
    ("SAC + HER, SGS", "her/runs/main/sac_her0.8_sgs_*.json", "#1baf7a", "--"),
    ("PPO, uniform, 4K envs", "runs_sweep/uniform_E4096_s*.json", "#eb6834", "-"),
    ("PPO, SGS, 4K envs", "runs_sweep/sgs_E4096_s*.json", "#eb6834", "--"),
    ("PPO, SGS, 64K envs, scaled LR", "runs_lr/sgs_E65536_s[01].json", "#6250d6", "--"),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for label, pat, col, ls in SERIES:
    logs = [json.load(open(f))["log"] for f in sorted(glob.glob(pat))]
    x_steps = np.array([r["env_steps"] for r in logs[0]])
    x_min = np.mean([[r["time"] for r in l] for l in logs], 0) / 60
    y = np.array([[r["eval_success"] for r in l] for l in logs])
    for ax, x in zip(axes, (x_steps, x_min)):
        ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=col, alpha=0.12, lw=0)
        ax.plot(x, y.mean(0), color=col, ls=ls, lw=2, label=label)
axes[0].set_xscale("log")
axes[0].set_xlabel("Environment steps (log scale)")
axes[0].set_title("Sample efficiency", loc="left", fontsize=11)
axes[1].set_xlabel("Wall-clock minutes on one RTX 4060 Ti")
axes[1].set_xlim(0, 62)
axes[1].set_title("Compute efficiency", loc="left", fontsize=11)
axes[0].set_ylabel("Eval success (all goals)")
for ax in axes:
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e5e5e5")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=4, fontsize=9)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig("her/sac_vs_ppo.png", dpi=150)
print("wrote her/sac_vs_ppo.png")
