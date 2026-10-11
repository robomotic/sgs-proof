"""Acrobot figures: sample task configurations (acrobot_configs.png) and the negative-control result (acrobot_results.png)."""
import glob, json, math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import acrobot

INK, MUTED, GRID = "#333333", "#777777", "#e5e5e5"
COLORS = {"sgs": "#2a78d6", "uniform": "#eb6834"}
NAMES = {"sgs": "SGS", "uniform": "Uniform"}


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# ---- 1. what a task configuration looks like: same rng call order as training (--maze-seed 0, --tau-min 0.25)
cfgs, _ = acrobot.make_configs(32768, np.random.default_rng(0), (0.25, 3.0))
pick = [int(np.argmin(np.abs(cfgs[:, 4] - t) + 3 * np.abs(np.cos(cfgs[:, 0]) - c))) for t, c in
        ((0.35, 1), (0.8, 0.5), (1.4, 0), (2.0, -0.3), (2.6, -0.9), (2.9, 0.9))]  # spread torque and start angle
cmap = plt.get_cmap("Blues_r")  # dark = low torque = hard
fig, axes = plt.subplots(1, 6, figsize=(12, 2.9))
for ax, i in zip(axes, pick):
    th1, th2, _, _, tau, m2 = cfgs[i]
    elbow = (math.sin(th1), -math.cos(th1))
    tip = (elbow[0] + math.sin(th1 + th2), elbow[1] - math.cos(th1 + th2))
    col = cmap(0.1 + 0.6 * (tau - 0.25) / 2.75)
    ax.axhline(1.8, color=MUTED, ls="--", lw=1)
    ax.plot([0, 0], [0, 2], color=GRID, lw=6, solid_capstyle="round", zorder=0)  # upright target pose
    ax.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]], color=col, lw=2 + 2 * m2, solid_capstyle="round")
    ax.plot(0, 0, "o", color=INK, ms=6)
    ax.plot(*elbow, "o", color="white", mec=col, ms=8, mew=2)  # actuated joint
    ax.set_xlim(-2.2, 2.2), ax.set_ylim(-2.2, 2.3), ax.set_aspect("equal"), ax.axis("off")
    ax.set_title(f"torque ≤ {tau:.2f}\nm₂ = {m2:.2f}", fontsize=9, color=INK)
axes[0].text(-2.2, 1.9, "success\nheight", fontsize=7, color=MUTED, va="bottom")
fig.suptitle("Acrobot task configurations: start pose, torque limit (darker = weaker = harder), link-2 mass (thicker = heavier).\n"
             "Grey bar = upright target. Only the elbow (open circle) is actuated. Success = tip above the dashed line for 5 steps.",
             fontsize=9, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.82))
fig.savefig("acrobot_configs.png", dpi=150)

# ---- 2. result: learning curves + why (uniform's config breakdown, maze vs acrobot)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.15]})
for s in ("sgs", "uniform"):
    logs = [json.load(open(f))["log"] for f in sorted(glob.glob(f"runs_acro_D1k/{s}_E4096_s*.json"))]
    it = [r["iter"] for r in logs[0]]
    y = np.array([[r["eval_success"] for r in log] for log in logs])
    a1.fill_between(it, y.mean(0) - y.std(0), y.mean(0) + y.std(0), color=COLORS[s], alpha=0.15, lw=0)
    a1.plot(it, y.mean(0), color=COLORS[s], lw=2, marker="o", ms=7, mec="white", mew=2, label=NAMES[s])
a1.set_ylim(0, 1), a1.set_xlabel("PPO iteration"), a1.set_ylabel("Eval success")
a1.set_title(f"Acrobot swing-up, 4K envs, {len(logs)} seeds: no gap", loc="left", fontsize=11)
a1.grid(axis="y", color=GRID), a1.legend(frameon=False, loc="upper left"), clean(a1)


def breakdown(pat):  # mean fraction of configs never solved / in between / mastered, under uniform sampling
    rows = [json.load(open(f))["log"][-1] for f in glob.glob(pat)]
    mid, top = np.mean([r["frac_frontier"] for r in rows]), np.mean([r["frac_mastered"] for r in rows])
    return 1 - mid - top, mid, top


parts = {"Maze (SGS +11 pts)": breakdown("runs_sweep/uniform_E4096_s*.json"),
         "Acrobot (SGS +0 pts)": breakdown("runs_acro_D1k/uniform_E4096_s*.json")}
seg = [("never/rarely solved (<5%)", "#b8b8b8"), ("intermediate (5–95%)", "#1baf7a"), ("mastered (≥95%)", "#6b6b6b")]
for row, (task, fr) in enumerate(parts.items()):
    left = 0
    for (label, col), v in zip(seg, fr):
        a2.barh(row, v, left=left, color=col, edgecolor="white", linewidth=2, height=0.55, label=label if row == 0 else None)
        if v > 0.06:
            a2.text(left + v / 2, row, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if col != "#b8b8b8" else INK)
        left += v
a2.set_yticks(range(len(parts)), list(parts)), a2.invert_yaxis()
a2.set_xlim(0, 1), a2.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
a2.set_title("Why: config success rates under uniform sampling", loc="left", fontsize=11)
a2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
clean(a2)
fig.tight_layout()
fig.savefig("acrobot_results.png", dpi=150)
print("wrote acrobot_configs.png acrobot_results.png")
