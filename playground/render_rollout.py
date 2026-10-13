"""Render recorded episodes of a trained policy (from brax_compare.py --save-rollout) as a GIF and a filmstrip.
    .venv/bin/python render_rollout.py runs_video/ppo_s0_rollout.npz     -> ppo_pick.gif, ppo_filmstrip.png"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco_playground import registry
from PIL import Image

roll = np.load(sys.argv[1] if len(sys.argv) > 1 else "runs_video/ppo_s0_rollout.npz")
env = registry.load("PandaPickCube", config_overrides={"impl": "jax"})
m = env.mj_model
d = mujoco.MjData(m)
target_geom = [g for g in range(m.ngeom) if m.body_mocapid[m.geom_bodyid[g]] >= 0][0]
m.geom_rgba[target_geom] = [0.9, 0.1, 0.1, 0.45]
renderer = mujoco.Renderer(m, 360, 480)
cam = mujoco.MjvCamera()
cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [0.5, 0.0, 0.28], 1.45, 140, -18


def frame(ep, t):
    d.qpos[:], d.mocap_pos[:], d.mocap_quat[:] = roll["qpos"][ep, t], roll["mocap_pos"][ep, t], roll["mocap_quat"][ep, t]
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, cam)
    return renderer.render()


n_ep, T = roll["qpos"].shape[:2]
ok = roll["success"][:, -1] > 0
print(f"{n_ep} episodes, success at final step: {ok.tolist()}")

# GIF: all episodes side by side, every 3rd control step at 60 ms/frame (= real time), downscaled to keep it small
def gif_frame(t):
    img = Image.fromarray(np.concatenate([frame(e, t) for e in range(n_ep)], axis=1))
    return img.resize((int(img.width * 0.7), int(img.height * 0.7)), Image.LANCZOS).quantize(colors=192)


frames = [gif_frame(t) for t in range(0, T, 3)]
frames[0].save("ppo_pick.gif", save_all=True, append_images=frames[1:], duration=60, loop=0, optimize=True)

# Filmstrip: episode 0 at evenly spaced moments
ts = np.linspace(0, T - 1, 6).astype(int)
fig, axes = plt.subplots(1, len(ts), figsize=(18, 3.2))
for ax, t in zip(axes, ts):
    ax.imshow(frame(0, t))
    ax.set_title(f"t = {t * 0.02:.1f} s", fontsize=10, loc="left")
    ax.axis("off")
fig.suptitle("Trained PPO policy on PandaPickCube: reach, grasp, lift and hold the green cube inside the red target "
             f"({'success' if ok[0] else 'failure'} on this episode)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("ppo_filmstrip.png", dpi=110)
print("wrote ppo_pick.gif ppo_filmstrip.png")
