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

Final evaluation success after 1000 PPO iterations, mean ± std over 3 seeds. "Hardest goals" means goals 41–60 cells away along the maze path.

| Envs | SGS, all goals | Uniform, all goals | SGS, hardest goals | Uniform, hardest goals |
|---|---|---|---|---|
| 4K  | **0.63** ± 0.01 | 0.52 ± 0.02 | **0.47** ± 0.10 | 0.22 ± 0.10 |
| 16K | **0.60** ± 0.04 | 0.48 ± 0.02 | **0.32** ± 0.22 | 0.05 ± 0.05 |
| 64K | **0.68** ± 0.01 | 0.51 ± 0.02 | **0.62** ± 0.12 | 0.10 ± 0.01 |

![SGS vs uniform scaling](runs_sweep/scaling.png)

### What reproduced

- **SGS beats uniform at every scale.** It leads by 11–17 points overall. On the hardest goals it does 2–6× better.
- **Uniform sampling gets nothing from scale.** Overall success stays flat at about 0.5 from 4K to 64K. Success on the hardest goals *drops* from 0.22 to about 0.1. This matches the paper's diagnosis: the minibatch count is fixed, so larger batches only average more already-solved configurations into the same number of gradient steps.
- **SGS gets the most out of the largest scale.** At 64K, SGS reaches 0.62 on the hardest goals, against 0.10 for uniform.
- **Capability at small scale (Q2).** Even at 4K, SGS solves roughly twice as many of the hardest goals as uniform. A separate single-seed run on a smaller 6×6 maze showed the same pattern: 0.46 vs 0.24 on long goals.

### What didn't (cleanly) reproduce

- **Steady improvement with scale.** SGS dips at 16K before peaking at 64K. That point has a large spread on the hardest goals (std 0.22), so with only 3 seeds it may be one bad seed rather than a real effect.

### Not tested

- **PLR and SFL baselines.** The paper's main comparison is PLR; only uniform sampling is implemented here.
- **Scales above 64K.** The paper goes up to 1M environments.
- **More than one maze, more than 3 seeds, learning-rate scaling with batch size.**
- **The hand-designed curricula, real-robot transfer and DAgger distillation from the paper.** These are out of scope.

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

Nothing is running yet; these are the planned runs, in priority order. Each one targets a gap listed under "What didn't reproduce" or "Not tested".

| # | Experiment | Why | Est. GPU time |
|---|---|---|---|
| 1 | **3 more seeds at 16K** (seeds 3–5, both samplers) | Settle whether SGS's dip at 16K is real or one bad seed (hardest-goals std is 0.22) | ~1.5 h |
| 2 | **128K environments** (3 seeds × 2 samplers) | Extend the scaling curve one more step (2×); check that the SGS–uniform gap keeps widening | ~12 h (est. ~2 h per run) |
| 3 | **PLR baseline** (absolute-GAE score + staleness, as in the paper) | The paper's main comparison; uniform alone is the weakest baseline | ~30 lines of code, then ~4 h for 4K–64K × 3 seeds |
| 4 | **Learning-rate scaling** with batch size | Rule out that uniform's flat curve is just a fixed-LR artifact | ~4 h |
| 5 | **More mazes** (`--maze-seed 1..4`) | Make sure the result isn't specific to one maze layout | ~2 h at 4K |
| 6 | **FetchPush** (Gymnasium-Robotics) with SGS vs uniform vs HER | A real manipulation task with built-in goals and success flags; tests capability (Q2), not scaling | Days (CPU-bound) |

Commands for steps 1 and 2 (resumable, logs to `sweep.log`):

```bash
# 1: more seeds at 16K
for seed in 3 4 5; do for s in sgs uniform; do
  .venv/bin/python sgs_maze.py --sampler $s --num-envs 16384 --seed $seed --cells 8 --max-dist 60 \
    --iters 1000 --eval-every 100 --out runs_sweep >> sweep.log 2>&1
done; done

# 2: 128K point (plot.py's x-ticks need a "128K" entry added)
for seed in 0 1 2; do for s in sgs uniform; do
  .venv/bin/python sgs_maze.py --sampler $s --num-envs 131072 --seed $seed --cells 8 --max-dist 60 \
    --iters 1000 --eval-every 100 --out runs_sweep >> sweep.log 2>&1
done; done
```

`plot.py` picks up any new JSON files in `runs_sweep/` automatically. The seed count `n=` on the plot labels shows how many seeds each point averages.

## Running it

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128  # match your driver
uv pip install --python .venv/bin/python numpy matplotlib

.venv/bin/python sgs_maze.py --selftest                       # maze connectivity, collisions, sampler bookkeeping
.venv/bin/python sgs_maze.py --sampler sgs --num-envs 4096 --cells 8 --max-dist 60
./sweep.sh                                                    # full sweep, resumable, logs to sweep.log
.venv/bin/python plot.py                                      # runs_sweep/scaling.png + summary table
```

## GPU budget

Everything here used **about 9 hours of one RTX 4060 Ti**:

| Phase | Runs | GPU wall-clock |
|---|---|---|
| Exploration (maze sizes, SGS settings, calibration) | 9 | ~1 h (2–3 runs shared the GPU at a time) |
| Scaling sweep (3 seeds × 3 scales × 2 samplers) | 18 | 7.9 h (one run at a time) |
| **Total** | **27** | **~9 h** |

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
| `sgs_maze.py` | Maze generator, GPU environment, SGS and uniform samplers, PPO, evaluation, self-test |
| `sweep.sh` | 3 seeds × {4K, 16K, 64K} × {SGS, uniform} |
| `plot.py` | Scaling plot and summary from `runs_sweep/*.json` |
| `plot_maze.py` | Maze pictures: `maze_sizes.png`, `maze_configs.png` |
| `runs_sweep/` | Per-run JSON logs (evaluation curves, success by distance, fraction of mastered configs) |

## Implementation notes

- If several environments finish the same configuration in one step, each outcome gets its own slot in that configuration's ring buffer, so no outcome is lost.
- Timeouts are treated as terminal states, with no bootstrapping. The observation has no time feature, so the value function couldn't use one anyway.
- Configurations that have never been visited have *p* = 0, so SGS samples them rarely once some configurations start succeeding. This is the paper's intended behaviour: the ε floor keeps every configuration's probability above zero.
