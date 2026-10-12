"""Demo-value curve: SGS eval success vs parallel envs for each demo fraction (demos_curve.png)."""
import glob, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENVS = (256, 1024, 4096)
BASE = {256: "runs_base_256", 1024: "runs_base_1k", 4096: "runs_sweep"}
FRACS = [("0", "no demos"), ("0.001", "0.1% of configs (33)"), ("0.01", "1% (327)"), ("0.1", "10% (3,276)")]
COLORS = {"0": "#9a9a9a", "0.001": "#9ec5f0", "0.01": "#3987e5", "0.1": "#16457f"}  # one hue, light -> dark = more demos


def finals(frac, E, k):
    d = BASE[E] if frac == "0" else "runs_demo" if (frac == "0.01" and E != 256) else f"runs_demo_curve/f{frac}"
    rows = [json.load(open(f))["log"][-1] for f in glob.glob(f"{d}/sgs_E{E}_s*.json")]
    return np.array([r["eval_success"] if k == 0 else r["eval_by_dist"]["41-60"] for r in rows])


fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, k, title in zip(axes, (0, 1), ("All goals", "Hardest goals (41–60 cells)")):
    ref = (0.879, 0.844)[k]  # SGS at 64K envs, scaled LR, no demos (runs_lr)
    ax.axhline(ref, color="#555", lw=1, ls=":")
    ax.text(ENVS[0], ref + 0.015, "SGS, 64K envs, no demos", fontsize=8, color="#555")
    labels = []
    for frac, name in FRACS:
        v = [finals(frac, E, k) for E in ENVS]
        mu, sd = np.array([a.mean() for a in v]), np.array([a.std() for a in v])
        ax.fill_between(ENVS, mu - sd, mu + sd, color=COLORS[frac], alpha=0.15, lw=0)
        ax.plot(ENVS, mu, color=COLORS[frac], lw=2, ls="--" if frac == "0" else "-", marker="o", ms=8, mec="white", mew=2)
        labels.append([mu[-1], name])
    labels.sort()
    for a, b in zip(labels, labels[1:]):
        b[0] = max(b[0], a[0] + 0.045)
    for y, name in labels:
        ax.annotate(name.split(" (")[0], (ENVS[-1], y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color="#333")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ENVS, ["256", "1K", "4K"])
    ax.set_xlim(200, 6500)
    ax.set_ylim(0, 1)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("Parallel environments")
    ax.grid(axis="y", color="#e5e5e5")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Eval success, SGS (1000 PPO iters)")
fig.legend(handles=[plt.Line2D([], [], color=COLORS[f], lw=2, ls="--" if f == "0" else "-", marker="o", label=n)
                    for f, n in FRACS], frameon=False, loc="lower center", ncol=4, fontsize=9, title="Expert demos for")
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig("demos_curve.png", dpi=150)
print("wrote demos_curve.png")
