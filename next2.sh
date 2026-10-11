#!/bin/bash
# Next steps 3, 5, 4, 2 (in that order). Resumable: finished runs are skipped.
cd "$(dirname "$0")"
run() {  # out sampler envs seed [extra args...]
  local out=$1 s=$2 E=$3 seed=$4; shift 4
  [ -f $out/${s}_E${E}_s${seed}.json ] && return
  .venv/bin/python sgs_maze.py --sampler $s --num-envs $E --seed $seed --cells 8 --max-dist 60 \
    --iters 1000 --eval-every 100 --out $out "$@" >> next.log 2>&1 || echo "FAILED $out $s E=$E s=$seed $*" >> next.log
}
# 3: PLR baseline across scales
for seed in 0 1 2; do for E in 4096 16384 65536; do run runs_sweep plr $E $seed; done; done
echo "STEP3 PLR DONE" >> next.log
# 5: other maze layouts, 4K envs
for m in 1 2 3 4; do for s in sgs uniform plr; do run runs_mazes/m$m $s 4096 0 --maze-seed $m; done; done
echo "STEP5 MAZES DONE" >> next.log
# 4: learning rate scaled by sqrt(batch growth): 16K -> 6e-4, 64K -> 1.2e-3
for seed in 0 1 2; do for s in sgs uniform; do
  run runs_lr $s 16384 $seed --lr 6e-4
  run runs_lr $s 65536 $seed --lr 1.2e-3
done; done
echo "STEP4 LR DONE" >> next.log
# 2: 128K envs
for seed in 0 1 2; do for s in sgs uniform; do run runs_sweep $s 131072 $seed; done; done
echo "ALL NEXT STEPS DONE" >> next.log
