"""Stage A figure: PPO vs SAC on PandaPickCube, by env steps and by wall-clock (stage_a.png).
Run with the main repo venv (it has matplotlib): ../.venv/bin/python plot_stage_a.py"""
import glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
SERIES = [("PPO (tuned)", "runs/ppo_s*.json", "#eb6834", "-"),
          ("SAC, 5M steps", "runs/sac_s*.json", "#2a78d6", "-"),
          ("SAC, 20M steps", "runs_long/sac_s*.json", "#2a78d6", "--")]

fig, (ax_r, *axes) = plt.subplots(1, 3, figsize=(15, 4.3))
for label, pat, col, ls in SERIES:
    runs = [json.load(open(f)) for f in sorted(glob.glob(pat))]
    if not runs:
        continue
    n = min(len(r["log"]) for r in runs)
    x_steps = np.array([r["env_steps"] for r in runs[0]["log"][:n]])
    x_min = np.mean([[r["time"] for r in run["log"][:n]] for run in runs], 0) / 60
    y = np.array([[r["steps_at_target"] / 150 for r in run["log"][:n]] for run in runs])
    rew = np.array([[r["reward"] for r in run["log"][:n]] for run in runs])
    x_steps, x_min, y, rew = x_steps[1:], x_min[1:], y[:, 1:], rew[:, 1:]  # drop the step-0 eval (no place on a log axis)
    succ = np.mean([r["success_final"] for r in runs])
    name = f"{label} ({len(runs)} seed{'s' * (len(runs) > 1)}, final success {succ:.0%})"
    ax_r.plot(x_steps, rew.mean(0), color=col, ls=ls, lw=2, marker="o", ms=5)
    for ax, x in zip(axes, (x_steps, x_min)):
        if len(runs) > 1:
            ax.fill_between(x, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=col, alpha=0.15, lw=0)
        ax.plot(x, y.mean(0), color=col, ls=ls, lw=2, marker="o", ms=5, label=name)
ax_r.set_xscale("log")
ax_r.set_xlabel("Environment steps (log scale)")
ax_r.set_ylabel("Episode reward (task's shaped reward)")
ax_r.set_title("Reward: both stall near 670; SAC escapes at ~10M steps, PPO at ~14M", loc="left", fontsize=11)
ax_r.axhline(670, color="#999", lw=1, ls=":")
ax_r.grid(axis="y", color="#e5e5e5")
for sp in ("top", "right"):
    ax_r.spines[sp].set_visible(False)
axes[0].set_xscale("log")
axes[0].set_xlabel("Environment steps (log scale)")
axes[0].set_title("Success, by environment steps", loc="left", fontsize=11)
axes[1].set_xlabel("Wall-clock minutes on one RTX 4060 Ti (incl. ~2 min compile)")
axes[1].set_title("Success, by wall-clock", loc="left", fontsize=11)
axes[1].set_xlim(left=0)
axes[0].set_ylabel("Fraction of episode with cube\nwithin 5 cm of target")
axes[1].set_yticklabels([])
for ax in axes:
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e5e5e5")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, loc="lower center", ncol=len(l), fontsize=9)
fig.suptitle("MuJoCo Playground PandaPickCube: standard Brax PPO vs Brax SAC", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
fig.savefig("stage_a.png", dpi=150)
print("wrote stage_a.png")
