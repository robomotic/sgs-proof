#!/bin/bash
# Prior-knowledge experiments: maze expert demos (DAPG-style BC) and Acrobot backward-from-upright resets. Resumable.
cd "$(dirname "$0")"
run() {  # out sampler envs seed [extra args...]
  local out=$1 s=$2 E=$3 seed=$4; shift 4
  [ -f $out/${s}_E${E}_s${seed}.json ] && return
  .venv/bin/python sgs_maze.py --sampler $s --num-envs $E --seed $seed --iters 1000 --out $out "$@" >> priors.log 2>&1 \
    || echo "FAILED $out $s E=$E s=$seed $*" >> priors.log
}
M="--cells 8 --max-dist 60 --eval-every 100"
A="--task acrobot --horizon 400 --height 1.8 --hold 5 --tau-min 0.25 --eval-every 250"
( for seed in 0 1 2; do for s in sgs uniform; do run runs_demo $s 4096 $seed $M --demo-frac 0.01; done; done ) &
( for seed in 0 1 2; do run runs_demo sgs 1024 $seed $M --demo-frac 0.01; run runs_base_1k sgs 1024 $seed $M; done ) &
( for seed in 0 1 2; do for s in sgs uniform; do run runs_acro_rev $s 4096 $seed $A --reverse-frac 0.5; done; done ) &
wait
echo "PRIORS DONE" >> priors.log
