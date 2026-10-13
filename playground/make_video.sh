#!/bin/bash
# After the 20M SAC run finishes: retrain PPO seed 0 saving weights + 3 recorded episodes, then render GIF + filmstrip.
cd "$(dirname "$0")"
while pgrep -f "^.venv/bin/python brax_compare.py --algo sac" >/dev/null; do sleep 60; done
.venv/bin/python brax_compare.py --algo ppo --seed 0 --num-evals 10 --save-rollout 3 --out runs_video >> video.log 2>&1 \
  && .venv/bin/python render_rollout.py runs_video/ppo_s0_rollout.npz >> video.log 2>&1 \
  && echo "VIDEO DONE" >> video.log || echo "VIDEO FAILED" >> video.log
