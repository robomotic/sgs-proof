"""Scaling plot: final eval success vs parallel envs per sampler (mean ± std over seeds). Writes <first dir>/scaling.png.
    python plot.py [dir ...]   (default: runs_sweep runs_plr_b1)"""
import glob, json, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

srcs = sys.argv[1:] or ["runs_sweep", "runs_plr_b1"]
src = srcs[0]
res = defaultdict(list)  # (sampler, E) -> [(overall, longest-bin)]
for f in [f for d in srcs for f in glob.glob(f"{d}/*.json")]:
    r = json.load(open(f))
    m = r["args"]["sampler"]
    if m == "plr":
        m = f"plr{r['args'].get('plr_beta', 0.1):g}"
    last = r["log"][-1]
    res[m, r["args"]["num_envs"]].append((last["eval_success"], last["eval_by_dist"]["41-60"]))

colors = {"sgs": "#2a78d6", "uniform": "#eb6834", "plr1": "#1baf7a", "plr0.1": "#eda100"}
names = {"sgs": "SGS", "uniform": "Uniform", "plr1": "PLR (β=1)", "plr0.1": "PLR (β=0.1)"}
present = [s for s in colors if any(m == s for m, _ in res)]
all_E = sorted({E for _, E in res})
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, k, title in zip(axes, (0, 1), ("All goals", "Hardest goals (41–60 cells)")):
    for s in present:
        Es = sorted(E for (m, E) in res if m == s)
        if not Es:
            continue
        v = [np.array([x[k] for x in res[s, E]]) for E in Es]
        mu, sd = np.array([a.mean() for a in v]), np.array([a.std() for a in v])
        ax.fill_between(Es, mu - sd, mu + sd, color=colors[s], alpha=0.15, lw=0)
        ax.plot(Es, mu, color=colors[s], lw=2, marker="o", ms=8, mec="white", mew=2)
        ax.annotate(names[s], (Es[-1], mu[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color="#333", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_E, [f"{E // 1024}K" for E in all_E])
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("Parallel environments")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#e5e5e5")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Eval success (1000 PPO iters)")
axes[0].legend(handles=[plt.Line2D([], [], color=colors[s], lw=2, marker="o", label=names[s]) for s in present],
               frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(f"{src}/scaling.png", dpi=150)
for key in sorted(res):
    a = np.array(res[key])
    print(f"{key[0]:8s} E={key[1]:6d}  n={len(a)}  all {a[:,0].mean():.3f}±{a[:,0].std():.3f}  hard {a[:,1].mean():.3f}±{a[:,1].std():.3f}")
