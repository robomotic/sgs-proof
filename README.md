# SGS on a 2D maze: a single-GPU reproduction

This is a small reproduction of **Success Guided Sampling (SGS)** from *A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control* (Zhang, Guaman Castro, Yin et al., CoRL 2026, [sgs-rl.github.io](https://sgs-rl.github.io/)).

The paper trains with up to 2²⁰ parallel environments on clusters of L40S and H200 GPUs. This repo tests the same claim with a 2D goal-conditioned maze task on one RTX 4060 Ti (16 GB). The full sweep takes about 8 GPU-hours.

## Why a maze?

This was done on a budget: one consumer GPU, no cluster, no cloud credits. The paper's own setup is far out of reach. Its locomotion runs take about 36 hours and its nut-and-bolt runs about 128 hours, each on many datacenter GPUs, using up to a million Isaac Lab environments. The best a 4060 Ti can do in Isaac Lab is a few thousand environments of one simple task, and that's well below the regime where the paper's effect appears.

The goal was to test the *idea*, not to rebuild the *robots*. SGS is only a sampler over a fixed pool of task configurations, so any task works if it has:

1. **Many configurations with very different difficulty.** Some must be trivial, some impossible at first, and some in between. That is the situation where uniform sampling wastes the batch.
2. **A sparse success signal** to compute per-configuration success rates from.
3. **Cheap enough simulation to run tens of thousands of environments in parallel**, because the paper's claim is about scaling.

The options considered:

| Option | Many envs on one GPU? | Problem |
|---|---|---|
| The paper's Isaac Lab tasks | No (a few K at most) | Can't reach the scaling regime; days per run |
| [gym-pusht](https://github.com/huggingface/gym-pusht) | No (CPU pymunk, ~1 env per core) | Can't test scaling; RL from scratch with a sparse reward may never learn |
| [Gymnasium-Robotics Fetch](https://robotics.farama.org/envs/fetch/) | No (CPU MuJoCo) | Can't test scaling; hours to days per run; reviewers expect HER as a baseline |
| **2D maze in pure PyTorch** | **Yes (64K envs fit easily)** | Simplified physics, but none of the above |

The maze meets all three requirements. Path length gives a natural difficulty range, much like the paper's terrains run from flat ground to stepping stones. Success is a single boolean. The whole environment is a few tensor operations, so 64K agents step in milliseconds on the GPU. The closest analogue in the paper is multi-terrain locomotion, where navigation over increasingly hard terrain is also the task.

## The paper's claim

The paper trains from a large, fixed set of task configurations. Uniform sampling over that set wastes more and more of each batch on configurations the policy has already mastered or can't attempt yet. As a result, adding parallel environments stops helping.

SGS tracks a running success rate *p<sub>i</sub>* for each configuration over its last *H* episodes. It then samples configurations near a target success rate *t*:

```
w_i = (p_i + ε)^(κ·t) · (1 − p_i + ε)^(κ·(1−t))
P(i) = softmax(log(w_i + ε) / T)
```

The paper reports that SGS:

- **(Q1) scales**: it keeps improving as the number of parallel environments grows, while uniform sampling and PLR level off or collapse;
- **(Q2) unlocks new capabilities**: it solves configurations the baselines can't, even at small scale.

## Mapping to this repo

| Paper | This repo |
|---|---|
| Locomotion terrains and assembly tasks (Isaac Lab) | One fixed procedural maze. A 2D point mass runs in pure-PyTorch batched simulation on the GPU. |
| Task configuration τ = (s₀, g, e) | (start position, goal cell). The pool holds N = 32,768 configurations with path length 1–60 grid cells. |
| Shared sparse reward + small regularizers | +1 on reaching the goal (within 0.5 cells), −1e-3·‖a‖² per step |
| PPO, fixed number of minibatches, batch grows with environments | Same: 32 steps per environment per iteration, 4 minibatches, 5 epochs, 1000 iterations |
| SGS locomotion hyperparameters | Same: H=100, t=0.66, κ=5, ε=1e-8, T=2 |
| Evaluation on freshly sampled configurations | 4,096 held-out configurations from the same distribution, deterministic policy |
| 4K → 1M environments | 4K → 16K → 64K environments, 3 seeds each |

**Environment details:** an 8×8-cell DFS maze on a 17×17 grid with 10% of interior walls removed. Actions are in [−1, 1]² and scaled to 0.5 cells per step. Wall collision is resolved separately on each axis. Episodes last up to 200 steps. The observation is the agent position, the goal, the vector from agent to goal, and a 5×5 patch of nearby walls.

### The mazes

Three sizes were tried. The difficulty had to be one where uniform sampling plateaus without failing completely. The 6×6 maze was too easy, and the 10×10 maze was so hard that every method stalled.

![The three maze sizes](maze_sizes.png)

Each task configuration is a start position and a goal in the sweep maze. The pool of 32,768 configurations ranges from goals a few cells away to goals almost 60 cells away along winding corridors. The policy has to master all of them with one network. Five examples, drawn along their shortest paths:

![Example task configurations](maze_configs.png)

These are the exact mazes used in the runs: `--maze-seed 0`, regenerated by `plot_maze.py`.

## Results

Final evaluation success after 1000 PPO iterations, mean ± std over 3 seeds (6 at 16K). "Hardest goals" means goals 41–60 cells away along the maze path.

| Envs | SGS, all goals | Uniform, all goals | SGS, hardest goals | Uniform, hardest goals |
|---|---|---|---|---|
| 4K  | **0.63** ± 0.01 | 0.52 ± 0.02 | **0.47** ± 0.10 | 0.22 ± 0.10 |
| 16K | **0.62** ± 0.04 | 0.49 ± 0.02 | **0.41** ± 0.21 | 0.09 ± 0.06 |
| 64K | **0.68** ± 0.01 | 0.51 ± 0.02 | **0.62** ± 0.12 | 0.10 ± 0.01 |

![SGS vs uniform vs PLR scaling](runs_sweep/scaling.png)

### What reproduced

- **SGS beats uniform at every scale.** It leads by 11–17 points overall. On the hardest goals it does 2–6× better. All 6 SGS seeds at 16K beat all 6 uniform seeds overall.
- **Uniform sampling gets nothing from scale.** Overall success stays flat at about 0.5 from 4K to 64K. Success on the hardest goals *drops* from 0.22 to about 0.1. This matches the paper's diagnosis: the minibatch count is fixed, so larger batches only average more already-solved configurations into the same number of gradient steps.
- **SGS gets the most out of the largest scale.** At 64K, SGS reaches 0.62 on the hardest goals, against 0.10 for uniform.
- **SGS beats PLR at every scale (Q1), when PLR is given a sensible temperature.** Only SGS improves with more environments; PLR and uniform stay flat. See [PLR baseline](#plr-baseline).
- **Not specific to one maze.** SGS beats uniform by 12–13 points on each of 4 other layouts. See [Other maze layouts](#other-maze-layouts).
- **Capability at small scale (Q2).** Even at 4K, SGS solves roughly twice as many of the hardest goals as uniform. A separate single-seed run on a smaller 6×6 maze showed the same pattern: 0.46 vs 0.24 on long goals.

### What didn't (cleanly) reproduce

- **Steady improvement with scale.** With 3 seeds, SGS appeared to dip at 16K. Three more seeds showed the dip is mostly one bad seed. Per-seed results on the hardest goals at 16K are 0.35, **0.04**, 0.57, 0.27, 0.66 and 0.57 (median about 0.46, level with 4K). So the curve is flat from 4K to 16K and then rises at 64K. That's not the steady climb the paper reports.
- **SGS has a failure mode.** About 1 seed in 6 never solves the long goals. Uniform never solves them well on any seed either.

### PLR baseline

PLR (Prioritized Level Replay) is the paper's main baseline. It's implemented here as the paper describes: levels ranked by mean |GAE|, a staleness bonus, and 50% new levels until every level has been seen. PLR's temperature β controls how strongly it favours the top-ranked levels.

| Envs | SGS | **PLR β=1** | Uniform | PLR β=0.1 |
|---|---|---|---|---|
| *All goals* | | | | |
| 4K | **0.63** ± 0.01 | 0.58 ± 0.03 | 0.52 ± 0.02 | 0.22 |
| 16K | **0.62** ± 0.04 | 0.57 ± 0.05 | 0.49 ± 0.02 | 0.22 |
| 64K | **0.68** ± 0.01 | 0.57 ± 0.05 | 0.51 ± 0.02 | 0.21 |
| *Hardest goals* | | | | |
| 4K | **0.47** ± 0.10 | 0.33 ± 0.05 | 0.22 ± 0.10 | 0.00 |
| 16K | **0.41** ± 0.21 | 0.31 ± 0.12 | 0.09 ± 0.06 | 0.00 |
| 64K | **0.62** ± 0.12 | 0.36 ± 0.03 | 0.10 ± 0.01 | 0.00 |

Seeds: PLR β=1 has 3 per point; PLR β=0.1 has 2, 2 and 1 (cancelled early).

**With a sensible temperature (β=1), PLR is a strong baseline.** It beats uniform at every scale, and on the hardest goals by 3–4×. **SGS still beats PLR at every scale, and only SGS improves with more environments.** PLR stays flat at 0.57 overall from 4K to 64K, while SGS rises to 0.68, and to 0.62 vs 0.36 on the hardest goals. That is the paper's Q1 claim ("SGS scales better than uniform *and PLR*"), reproduced in miniature.

**The default PLR temperature (β=0.1) breaks at this scale.**

With β=0.1, PLR does **much worse than uniform**. This comes from the default setting, not a code bug. PLR weights configs by rank^(−1/β) = rank^(−10). With 32,768 configs that sends **about 90% of all resets to the single highest-scoring config**, so each batch of 4K+ environments is nearly one task. PLR's defaults were tuned for about 64 environments. The paper doesn't say which PLR settings it used, but its PLR roughly matches uniform on most tasks, so it can't have been this degenerate. With β=1, about 8% of resets go to the top config. That's the fair comparison above.

### Other maze layouts

Is the result specific to one maze? Here are four more 8×8 layouts (`--maze-seed 1..4`), 4K envs, 1 seed each. The original layout is maze 0.

| Maze | SGS, all goals | Uniform, all goals | PLR β=1, all goals | SGS, hardest | Uniform, hardest | PLR β=1, hardest |
|---|---|---|---|---|---|---|
| 0 (3 seeds) | **0.63** | 0.52 | 0.58 | **0.47** | 0.22 | 0.33 |
| 1 | **0.51** | 0.38 | 0.32 | **0.11** | 0.01 | 0.00 |
| 2 | **0.58** | 0.45 | 0.38 | **0.29** | 0.04 | 0.00 |
| 3 | **0.43** | 0.30 | 0.36 | 0.00 | 0.00 | 0.00 |
| 4 | **0.61** | 0.47 | 0.47 | **0.06** | 0.02 | 0.04 |

**SGS beats uniform on every layout, by 12–13 points.** That's a remarkably consistent gap. PLR (β=1) is less robust: it beat uniform on maze 0 but loses to it on mazes 1 and 2, ties on maze 4, and wins only on maze 3. The new layouts are harder than maze 0 for every method, and the hardest goals stay mostly out of reach at 4K.

### Acrobot: a negative control

To see whether the effect carries over to classical control, the same code also runs a torque-limited **Acrobot swing-up** (`--task acrobot`, in [acrobot.py](acrobot.py)). It's a two-link pendulum where only the elbow is actuated. A task configuration is (initial angles and velocities, torque limit 0.25–3, link-2 mass 0.5–1.5). Success means holding the tip above height 1.8 (out of a maximum of 2.0) for 5 steps, which means swinging up and nearly balancing. The dynamics follow Sutton & Barto with RK4 integration, and the selftest checks energy conservation.

![Acrobot task configurations](acrobot_configs.png)

Calibration (4K envs, 300 iterations, 1 seed) to find a difficulty where learning happens without saturating:

| Success criterion | SGS | Uniform |
|---|---|---|
| Tip above 1.8, held 10 steps | 0.09 | 0.08 |
| Above 1.7, held 10, torque from 0.25 | 0.15 | 0.14 |
| **Above 1.8, held 5, torque from 0.25** | 0.27 | 0.25 |
| Above 1.5, held 5 | 0.75 | 0.71 |
| Cross 1.0 once (Gymnasium-style) | 0.93 | 0.93 |

Final setting (bold row), 1000 iterations, 3 seeds:

| | All configs | Lowest torque (0.25–1) | Highest torque (2–3) | Configs at 5–95% success |
|---|---|---|---|---|
| SGS | 0.31 ± 0.02 | 0.14 ± 0.01 | 0.44 ± 0.03 | 3–4% |
| Uniform | 0.31 ± 0.01 | 0.14 ± 0.02 | 0.41 ± 0.03 | **57–61%** |
| SGS, κ=1, t=0.5 (1 seed) | 0.30 | 0.13 | 0.44 | 46% |

![Acrobot results: no gap, and why](acrobot_results.png)

**No meaningful SGS advantage on Acrobot.** The last column of the table, and the right panel above, suggest why. Under uniform sampling, about 60% of Acrobot configs already sit at intermediate success. So uniform sampling isn't wasting the batch on solved or impossible configs, and SGS has nothing to fix. In the maze, 65% of configs under uniform sampling are either never solved or already mastered. That's the waste SGS removes. In Acrobot it's 41%, and most of the rest are configs the policy already learns from.

Two likely reasons, neither proven:
1. One skill, pumping energy and then balancing, transfers across all Acrobot configs, while each maze goal needs its own route.
2. Acrobot success depends more on policy noise than on the config, which makes per-config success rates a weak signal.

The working conclusion is that SGS helps when competence is config-specific and difficulty is very uneven, which is the situation the paper describes.

### Not tested

- **SFL baseline, and hand-designed curricula.**
- **Real-robot transfer and DAgger distillation from the paper.** These are out of scope.

### Earlier exploratory runs

All single-seed. Logs are in `*.log`, results in `runs_*/`.

| Setting | SGS | Uniform | Note |
|---|---|---|---|
| 10×10 maze, max path 80, 4K envs | 0.26 | 0.20 | Too hard: every method stalls and goals beyond 40 cells are about 0% |
| 10×10 maze, 16K envs | 0.22 | 0.20 | More environments helped neither method |
| 10×10, SGS with κ=1, t=0.5, ε=1e-4 (paper's manipulation values) | 0.23 | – | Slightly worse than κ=5 here |
| 6×6 maze, max path 40, 4K envs | 0.78 | 0.70 | Easy regime |
| 8×8 maze, max path 60, 4K envs, 600 iters | 0.56 | 0.47 | Calibration run for the sweep |

The 8×8 maze was chosen so that uniform sampling plateaus around 50% instead of failing outright. A gap between the two methods only shows up in that kind of regime.

## Next steps

**Running now:** [next3.sh](next3.sh), in this order (resumable, logs to `next.log`):

| # | Experiment | Why | Est. GPU time | Status |
|---|---|---|---|---|
| 1 | **PLR with β=1.0**, 4K/16K/64K × 3 seeds | Fair version of the paper's main baseline (see above) | ~4 h | **Done** |
| 2 | **4 more maze layouts** (`--maze-seed 1..4`) × {SGS, uniform, PLR β=1}, 4K | Check that the result isn't specific to one maze | ~1 h | **Done** |
| 3 | **Learning rate scaled with batch size** (√ rule: 6e-4 at 16K, 1.2e-3 at 64K) × {SGS, uniform} × 3 seeds | Rule out that uniform's flat curve is a fixed-LR artifact | ~7.5 h | Running |
| 4 | **128K environments** × {SGS, uniform} × 3 seeds | One more step on the scaling curve | ~12 h | Queued |

Afterwards, possibly: **FetchPush** (Gymnasium-Robotics) with SGS vs uniform vs HER, as a real manipulation task. That one is CPU-bound and would take days.

**Done:** 3 more seeds at 16K (resolved the dip, see above); PLR at β=0.1 (degenerate, see above); Acrobot (negative control, see above).

`plot.py` reads `runs_sweep/` and `runs_plr_b1/` by default and picks up new JSON files automatically. Pass directories to plot another set, for example `plot.py runs_lr`.

## Running it

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128  # match your driver
uv pip install --python .venv/bin/python numpy matplotlib

.venv/bin/python sgs_maze.py --selftest                       # maze, collisions, sampler, PLR, acrobot energy conservation
.venv/bin/python sgs_maze.py --sampler sgs --num-envs 4096 --cells 8 --max-dist 60   # --sampler sgs|uniform|plr
.venv/bin/python sgs_maze.py --task acrobot --sampler sgs --horizon 400 --height 1.8 --hold 5 --tau-min 0.25 --out runs_acro
./sweep.sh                                                    # full sweep, resumable, logs to sweep.log
.venv/bin/python plot.py                                      # runs_sweep/scaling.png + summary table
```

## GPU budget

So far everything here has used **about 13 hours of one RTX 4060 Ti**:

| Phase | Runs | GPU wall-clock |
|---|---|---|
| Exploration (maze sizes, SGS settings, calibration) | 9 | ~1 h (2–3 runs shared the GPU at a time) |
| Scaling sweep (3 seeds × 3 scales × 2 samplers) | 18 | 7.9 h (one run at a time) |
| 3 more seeds at 16K | 6 | 1.6 h |
| PLR β=0.1 (cancelled after 5 runs) | 5 | ~1.9 h |
| Acrobot calibration and seeds | 17 | ~0.8 h (2–4 runs shared the GPU at a time) |
| **Total so far** | **55** | **~13 h** |

The queue under "Next steps" adds about 25 h.

Time per 1000-iteration run with the GPU to itself:

| Parallel envs | Time per run | Env steps per run |
|---|---|---|
| 4K | ~5 min | 131 M |
| 16K | ~15 min | 524 M |
| 64K | ~60 min | 2.1 B |

At 4K environments training uses about 0.7 GB of GPU memory. Memory at 64K wasn't logged. 128K should still fit in 16 GB, raising `--num-minibatches` if needed, so the practical limit is time. For comparison, a *single* locomotion run in the paper takes about 36 hours on multiple datacenter GPUs.

## Files

| File | Contents |
|---|---|
| `sgs_maze.py` | Maze generator, GPU environment, SGS / uniform / PLR samplers, PPO, evaluation, self-test; `--task maze\|acrobot` |
| `acrobot.py` | GPU-batched torque-limited Acrobot swing-up (RK4) with its energy-conservation self-test |
| `sweep.sh` | 3 seeds × {4K, 16K, 64K} × {SGS, uniform} |
| `next1.sh`, `next2.sh`, `next3.sh` | Follow-up queues: extra 16K seeds; PLR β=0.1 (cancelled); PLR β=1, mazes, LR scaling, 128K |
| `plot.py` | Scaling plot and summary from `runs_sweep/*.json` |
| `plot_maze.py` | Maze pictures: `maze_sizes.png`, `maze_configs.png` |
| `plot_acrobot.py` | Acrobot pictures: `acrobot_configs.png`, `acrobot_results.png` |
| `runs_sweep/` | Per-run JSON logs (evaluation curves, success by distance, fraction of mastered configs) |
| `runs_acro_*/`, `runs_plr_b1/`, `runs_mazes/`, `runs_lr/` | Acrobot, PLR β=1, other mazes, LR scaling |

## Implementation notes

- If several environments finish the same configuration in one step, each outcome gets its own slot in that configuration's ring buffer, so no outcome is lost.
- Timeouts are treated as terminal states, with no bootstrapping. The observation has no time feature, so the value function couldn't use one anyway.
- PLR scores each config by the mean |GAE| of its steps in the latest 32-step rollout, rather than over a whole episode. This is a shortcut, since episodes span several rollouts.
- Configurations that have never been visited have *p* = 0, so SGS samples them rarely once some configurations start succeeding. This is the paper's intended behaviour: the ε floor keeps every configuration's probability above zero.
