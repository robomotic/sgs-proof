"""Pictures of the PandaPickCube task: rendered example configurations (task_scenes.png) and where cube starts and
targets are sampled (task_distribution.png). Runs JAX on the CPU so it doesn't disturb GPU training.
    .venv/bin/python render_task.py"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco_playground import registry

env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})
m = env.mj_model
d = mujoco.MjData(m)
box = env._obj_body
box_qadr = m.jnt_qposadr[m.body_jntadr[box]]
target_geom = [g for g in range(m.ngeom) if m.body_mocapid[m.geom_bodyid[g]] >= 0][0]
m.geom_rgba[target_geom] = [0.9, 0.1, 0.1, 0.45]  # make the translucent target a bit easier to see in a still image

states = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2000))
qpos, mocap_pos, mocap_quat = (np.asarray(x) for x in (states.data.qpos, states.data.mocap_pos, states.data.mocap_quat))
targets = np.asarray(states.info["target_pos"])


def set_config(i, cube_at_target=False):
    d.qpos[:], d.qvel[:] = qpos[i], 0
    d.mocap_pos[:], d.mocap_quat[:] = mocap_pos[i], mocap_quat[i]
    if cube_at_target:  # illustrate the goal: cube placed exactly at the target
        d.qpos[box_qadr:box_qadr + 3] = targets[i]
    mujoco.mj_forward(m, d)
    return d.xpos[box].copy()


renderer = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera()
cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [0.45, 0.0, 0.2], 1.55, 140, -22


def render():
    renderer.update_scene(d, cam)
    return renderer.render()


# ---- 1. example configurations + the goal state
starts = np.array([set_config(i) for i in range(len(qpos))])  # cube start positions for all sampled configs
picks = [0, 1, 3]
frames, titles = [], []
for k, i in enumerate(picks):
    set_config(i)
    frames.append(render())
    dist = np.linalg.norm(targets[i] - starts[i])
    titles.append(f"Config {k + 1}: cube start → target {dist * 100:.0f} cm away")
set_config(picks[0], cube_at_target=True)
frames.append(render())
titles.append("Goal (config 1): cube held at the target")
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, img, t in zip(axes, frames, titles):
    ax.imshow(img)
    ax.set_title(t, fontsize=10, loc="left")
    ax.axis("off")
fig.suptitle("PandaPickCube: a Franka arm must pick up the green cube and hold it inside the red translucent target "
             "(success here = within 5 cm at the end of the 150-step episode).", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("task_scenes.png", dpi=120)

# ---- 2. where configurations come from (300 of the 2,000 samples drawn, for legibility)
sub = slice(0, 300)
fig, (top, side) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
for ax, (i, j) in ((top, (0, 1)), (side, (0, 2))):
    ax.scatter(targets[sub, i], targets[sub, j], s=22, facecolors="none", edgecolors="#d33", lw=1, label="target")
    ax.scatter(starts[sub, i], starts[sub, j], s=14, marker="s", c="#1baf7a", alpha=0.8, label="cube start")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
top.add_patch(plt.Rectangle((starts[:, 0].min(), starts[:, 1].min()), np.ptp(starts[:, 0]), np.ptp(starts[:, 1]),
                            fill=False, ec="#777", ls="--", lw=1))
top.scatter([0], [0], marker="s", s=140, c="#333", label="robot base")
top.annotate("robot base", (0, 0), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8, color="#333")
top.set(xlabel="x (m, forward from robot)", ylabel="y (m)", title="Top view: cube and target both land in the same\n40 × 40 cm square (dashed) in front of the robot")
top.set_aspect("equal")
side.set(xlabel="x (m, forward from robot)", ylabel="height z (m)", title="Side view: cube starts on the table,\ntarget floats 20–40 cm above it")
side.set_ylim(0, 0.48)
handles = [plt.Line2D([], [], marker="s", ls="", color="#1baf7a", label="cube start"),
           plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#d33", label="target")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
dists = np.linalg.norm(targets - starts, axis=1)
fig.suptitle(f"Task configurations (300 of 2,000 samples shown). The cube must travel {dists.min() * 100:.0f}–"
             f"{dists.max() * 100:.0f} cm, median {np.median(dists) * 100:.0f} cm.", x=0.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 0.93))
fig.savefig("task_distribution.png", dpi=130)
print("wrote task_scenes.png task_distribution.png")
