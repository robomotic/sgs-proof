#!/bin/bash
# Stage A: Brax PPO (tuned, 3 seeds) vs Brax SAC (5M steps) on PandaPickCube. Resumable. Logs to stage_a.log.
cd "$(dirname "$0")"
run() { [ -f runs/$1_s$2.json ] && return; .venv/bin/python brax_compare.py --algo $1 --seed $2 --out runs "${@:3}" >> stage_a.log 2>&1 || echo "FAILED $1 s=$2" >> stage_a.log; }
( run sac 0 --steps 5000000 --num-evals 10 ) &
( run ppo 0 --num-evals 10; run ppo 1 --num-evals 10; run ppo 2 --num-evals 10 ) &
wait
echo "STAGE A DONE" >> stage_a.log
