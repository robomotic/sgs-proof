#!/bin/bash
# "Cheap and informative" follow-ups, 4 parallel lanes. Resumable. Logs to cheap.log.
cd "$(dirname "$0")"
run() {  # out sampler envs seed [extra args...]
  local out=$1 s=$2 E=$3 seed=$4; shift 4
  [ -f $out/${s}_E${E}_s${seed}.json ] && return
  .venv/bin/python sgs_maze.py --sampler $s --num-envs $E --seed $seed --iters 1000 --out $out "$@" >> cheap.log 2>&1 \
    || echo "FAILED $out $s E=$E s=$seed $*" >> cheap.log
}
M="--cells 8 --max-dist 60 --eval-every 100"
A="--task acrobot --horizon 400 --height 1.8 --hold 5 --tau-min 0.25 --eval-every 250 --reverse-frac 0.5"
# 1. Acrobot: where does SGS sample with reverse resets, and does a fixed 50/50 split fix it?
( for seed in 0 1 2; do run runs_acro_rev_log sgs 4096 $seed $A; run runs_acro_split sgs 4096 $seed $A --reverse-split; done ) &
# 2. SGS sensitivity on the maze (one setting changed at a time from t=0.66, kappa=5, T=2), 2 seeds
( for seed in 0 1; do
    for t in 0.3 0.5 0.8; do run runs_sens/t$t sgs 4096 $seed $M --target $t; done
    for k in 1 20; do run runs_sens/k$k sgs 4096 $seed $M --kappa $k; done
    for T in 0.5 8; do run runs_sens/T$T sgs 4096 $seed $M --temp $T; done
  done ) &
# 3. Demo value curve (demo fraction x envs) and noisy experts, SGS, 2 seeds
( for seed in 0 1; do
    run runs_base_256 sgs 256 $seed $M
    for f in 0.001 0.01 0.1; do for E in 256 1024 4096; do
      [ $f = 0.01 ] && [ $E != 256 ] && continue  # already in runs_demo
      run runs_demo_curve/f$f sgs $E $seed $M --demo-frac $f
    done; done
    for n in 0.2 0.5; do run runs_demo_noise/n$n sgs 4096 $seed $M --demo-frac 0.01 --demo-noise $n; done
  done ) &
# 4. More seeds on the other maze layouts
( for seed in 1 2; do for m in 1 2 3 4; do for s in sgs uniform; do run runs_mazes/m$m $s 4096 $seed $M --maze-seed $m; done; done; done ) &
wait
echo "CHEAP DONE" >> cheap.log
