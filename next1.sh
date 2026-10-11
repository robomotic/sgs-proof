#!/bin/bash
# Next step 1: seeds 3-5 at 16K, both samplers. Resumable.
cd "$(dirname "$0")"
for seed in 3 4 5; do for s in sgs uniform; do
  [ -f runs_sweep/${s}_E16384_s${seed}.json ] && continue
  .venv/bin/python sgs_maze.py --sampler $s --num-envs 16384 --seed $seed --cells 8 --max-dist 60 \
    --iters 1000 --eval-every 100 --out runs_sweep >> next.log 2>&1 || echo "FAILED $s E=16384 s=$seed" >> next.log
done; done
echo "NEXT1 DONE" >> next.log
