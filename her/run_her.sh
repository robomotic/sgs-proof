#!/bin/bash
# SAC / HER / SGS experiment: 256 envs, 8 updates per step, 3.3M env steps per run. Two lanes (one per seed). Resumable.
cd "$(dirname "$0")/.."
run() {  # out her_prob sampler seed [extra args...]
  local out=$1 h=$2 s=$3 seed=$4; shift 4
  [ -f $out/sac_her${h}_${s}_E256_s${seed}.json ] && return
  .venv/bin/python her/sac_her.py --her-prob $h --sampler $s --seed $seed --out $out "$@" >> her/run_her.log 2>&1 \
    || echo "FAILED $out her=$h $s s=$seed $*" >> her/run_her.log
}
lane() {
  local seed=$1
  for h in 0 0.8; do for s in uniform sgs; do run her/runs/main $h $s $seed; done; done  # 1. main: SAC +/- HER x uniform/SGS
  for h in 0 0.8; do run her/runs/hard $h uniform $seed --min-dist 41; done              # 2. hard goals only
}
lane 0 & lane 1 & wait
echo "HER DONE" >> her/run_her.log
