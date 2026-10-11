#!/bin/bash
# Maze follow-ups: PLR with flatter beta, other maze layouts, LR scaling, 128K envs. Resumable.
cd "$(dirname "$0")"
run() {  # out sampler envs seed [extra args...]
  local out=$1 s=$2 E=$3 seed=$4; shift 4
  [ -f $out/${s}_E${E}_s${seed}.json ] && return
  .venv/bin/python sgs_maze.py --sampler $s --num-envs $E --seed $seed --cells 8 --max-dist 60 \
    --iters 1000 --eval-every 100 --out $out "$@" >> next.log 2>&1 || echo "FAILED $out $s E=$E s=$seed $*" >> next.log
}
for seed in 0 1 2; do for E in 4096 16384 65536; do run runs_plr_b1 plr $E $seed --plr-beta 1.0; done; done
echo "PLR BETA1 DONE" >> next.log
for m in 1 2 3 4; do
  for s in sgs uniform; do run runs_mazes/m$m $s 4096 0 --maze-seed $m; done
  run runs_mazes/m$m plr 4096 0 --maze-seed $m --plr-beta 1.0
done
echo "MAZES DONE" >> next.log
for seed in 0 1 2; do for s in sgs uniform; do  # lr scaled by sqrt(batch growth)
  run runs_lr $s 16384 $seed --lr 6e-4
  run runs_lr $s 65536 $seed --lr 1.2e-3
done; done
echo "LR DONE" >> next.log
for seed in 0 1 2; do for s in sgs uniform; do run runs_sweep $s 131072 $seed; done; done
echo "ALL NEXT3 DONE" >> next.log
