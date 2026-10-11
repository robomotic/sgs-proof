#!/bin/bash
# SGS vs uniform scaling sweep on the 8x8 maze. Seeds outermost so partial results are usable.
cd "$(dirname "$0")"
for seed in 0 1 2; do for E in 4096 16384 65536; do for s in sgs uniform; do
  [ -f runs_sweep/${s}_E${E}_s${seed}.json ] && continue  # resumable
  .venv/bin/python sgs_maze.py --sampler $s --num-envs $E --seed $seed --cells 8 --max-dist 60 \
    --iters 1000 --eval-every 100 --out runs_sweep >> sweep.log 2>&1 || echo "FAILED $s E=$E s=$seed" >> sweep.log
done; done; done
echo SWEEP DONE >> sweep.log
