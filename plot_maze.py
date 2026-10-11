"""Pictures of the mazes: the three sizes tried, and the 8x8 sweep maze with sample (start, goal) configs by path length."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sgs_maze import all_pairs_dist, make_configs, make_maze

WALL, FLOOR = "#3a3a3a", "#f7f7f5"


def draw(ax, wall, title):
    ax.imshow(wall, cmap=matplotlib.colors.ListedColormap([FLOOR, WALL]), origin="lower",
              extent=(0, wall.shape[1], 0, wall.shape[0]))
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])


# same rng call order as sgs_maze.train() with --maze-seed 0, so these are the exact mazes used
mazes = {c: make_maze(c, 0.1, np.random.default_rng(0)) for c in (6, 8, 10)}

fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, (c, label) in zip(axes, [(6, "6×6 cells: too easy"), (8, "8×8 cells: used for the sweep"), (10, "10×10 cells: too hard")]):
    draw(ax, mazes[c], label)
fig.tight_layout()
fig.savefig("maze_sizes.png", dpi=150)

# sweep maze + a few sampled configs drawn along their shortest path, colored by path length (light = short, dark = long)
rng = np.random.default_rng(0)
wall = make_maze(8, 0.1, rng)
D = all_pairs_dist(wall)
starts, goals, dist = make_configs(wall, D, 32768, 60, rng)
n = wall.shape[0]


def shortest_path(s, g):
    cell, out = s, [s]
    while cell != g:  # walk downhill on BFS distance to the goal
        cell = next(c for c in (cell + n, cell - n, cell + 1, cell - 1) if D[c, g] == D[cell, g] - 1)
        out.append(cell)
    return np.array([[c % n + 0.5, c // n + 0.5] for c in out])


fig, ax = plt.subplots(figsize=(5.6, 5.6))
draw(ax, wall, "8×8 maze: five task configs, easy → hard")
cmap = plt.get_cmap("Blues")
for target in (6, 16, 28, 42, 58):
    i = int(np.argmin(np.abs(dist - target)))
    s = int(starts[i][1]) * n + int(starts[i][0])
    g = int(goals[i][1]) * n + int(goals[i][0])
    col = cmap(0.35 + 0.65 * dist[i] / 60)
    p = shortest_path(s, g)
    ax.plot(p[:, 0], p[:, 1], color=col, lw=3, alpha=0.85, solid_capstyle="round")
    ax.plot(*starts[i], "o", color=col, ms=8, mec="white", mew=1.5)
    ax.plot(*goals[i], "o", mfc="white", mec=col, ms=12, mew=2.5)
    ax.annotate(f"{dist[i]}", goals[i], xytext=(9, 6), textcoords="offset points", fontsize=8, color="#333")
ax.text(0, -1.0, "● start   ○ goal   number = path length in grid cells (max 60)", fontsize=8, color="#555")
fig.tight_layout()
fig.savefig("maze_configs.png", dpi=150)
print("wrote maze_sizes.png maze_configs.png")
