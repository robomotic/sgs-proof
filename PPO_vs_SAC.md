# PPO vs SAC for Robot Learning

*Compiled 2026-10-13. Covers published papers and preprints up to Oct 2026. Items marked **(preprint)** are not peer-reviewed. Where an algorithm's own authors report the margins, the entry says **(self-reported)**.*

---

## 1. TL;DR

| | **PPO** (on-policy, clipped policy gradient) | **SAC** (off-policy, max-entropy actor-critic) |
|---|---|---|
| Sample efficiency | Low: each batch is used for a few epochs, then thrown away | High: replay buffer reuses every transition many times |
| Wall-clock time in GPU-parallel sim | Excellent, the default choice (legged_gym, Isaac Lab) | Was historically slower. Recent recipes (FastSAC, FlashSAC, tuned SAC) match or beat PPO |
| Stability and tuning | Robust. One config transfers across many robots | More fragile: action bounds, entropy target, UTD ratio, and timeout handling all matter |
| Exploration | Gaussian noise with a learned or scheduled std | Entropy bonus with an auto-tuned temperature, which helps on dense-reward, high-DoF tasks |
| Real-robot learning and fine-tuning | Impractical because it needs fresh on-policy data in large volumes | Natural fit: replay buffer, warm starts, mixing sim and real data |
| Scaling with action dimension | Degrades on high-DoF tasks (humanoids, dexterous hands) | Recent evidence shows the largest gains over PPO here |
| Multi-task at scale | Strong, because the critic is a smaller bottleneck | Struggles (MTBench), with the critic as the bottleneck |
| Compute per sample | Cheap: small networks, about 20 gradient steps per iteration | Expensive: bigger networks and many updates per collected transition |

**Rule of thumb (2026):**
- **Use PPO** for single-task, low-to-mid-DoF locomotion or manipulation in massively parallel sim with an existing reward that was tuned for PPO. It also wins when robustness and reproducibility matter more than speed.
- **Use SAC** (or a modern variant such as FastSAC, FlashSAC, or the CrossQ/BRO family) when:
  - the task is high-DoF, such as humanoids or dexterous hands;
  - simulation is expensive, such as CPU sims, contact-rich scenes, or rendering;
  - you plan to fine-tune or learn on real hardware.

---

## 2. Algorithmic differences that matter for robots

| Aspect | PPO | SAC |
|---|---|---|
| Objective | Clipped surrogate advantage, plus an optional entropy bonus | Q-value plus α·entropy, with α auto-tuned against a target entropy |
| Data | Rollouts from the current policy only | Replay buffer (often 1M–10M transitions) |
| Critic | State-value V(s) with GAE | Twin Q(s,a) networks with target networks |
| Action space | Unbounded Gaussian, usually clipped by the env | tanh-squashed Gaussian, so **action bounds must be chosen well** |
| Episode timeouts | GAE handles them through bootstrap masks | Treating timeouts as terminations **biases the critic**, a common silent bug |
| Key knobs | Rollout length, epochs, minibatches, clip ε, GAE λ, LR schedule | UTD ratio, batch size, buffer size, target entropy, τ, n-step |
| Typical network | Small MLP, e.g. [512,256,128] ELU | Increasingly large, normalised MLPs, e.g. [1024,512,256] or 2.5M-parameter residual blocks |

PPO's performance is very sensitive to implementation details. These include observation and advantage normalisation, value clipping, LR annealing, and others ([Andrychowicz et al., ICLR 2021](https://arxiv.org/abs/2006.05990), 250k agents trained; [Huang et al., "37 Implementation Details of PPO", ICLR Blog Track 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)). SAC has an equivalent set of hidden details, and most of them only surface in the massively parallel regime (§4).

---

## 3. Pros and cons

### PPO
**Pros**
- **Built for GPU-parallel sim.** With thousands of environments, the wasted-sample problem stops mattering. [Rudin et al., CoRL 2021](https://proceedings.mlr.press/v164/rudin22a.html) trained ANYmal flat-terrain walking in under 4 minutes and rough terrain in about 20 minutes on one GPU, then transferred to hardware. This became the template for legged_gym and Isaac Lab.
- **Robust defaults.** One hyperparameter set covers many robots, and seed variance is low.
- **Cheap updates.** Small networks and few gradient steps per sample.
- **Scales to multi-task.** MT-PPO reached much higher success than MT-SAC on Meta-World MT10/MT50 and took about 22 minutes instead of about 12 hours for 200M frames ([Joshi et al., 2025](https://arxiv.org/abs/2507.23172) **(preprint)**; success is measured on training environments).
- **Mature ecosystem.** rsl_rl, rl_games, PBT for dexterous hands ([DexPBT, RSS 2023](https://arxiv.org/abs/2305.12127)), and most published sim-to-real locomotion and manipulation pipelines.

**Cons**
- **Sample-hungry.** Close to useless when simulation is slow (CPU sims, rendering, soft-body or rod contact) and unusable for learning on real hardware.
- **Degrades with action dimension.** On high-DoF humanoids and dexterous hands, recent off-policy recipes reach higher returns in less wall-clock time (§4).
- **Struggles with hard perturbations and long sequences.** PPO had difficulty with strong-push humanoid locomotion and long dance tracking compared with FastSAC/FastTD3 ([Seo et al., 2025](https://arxiv.org/abs/2512.01996) **(preprint)**).
- **Cannot reuse old data.** No demonstrations, no prior-run data, and no mixing of real and sim data without extra machinery.

### SAC
**Pros**
- **Sample efficiency.** This makes learning on real robots possible:
  - The original work learned Minitaur walking in about 2 hours on hardware ([Haarnoja et al., 2018/19](https://research.google/blog/soft-actor-critic-deep-reinforcement-learning-for-robotics/)).
  - Modern RLPD/SAC-style systems learn precise vision-based manipulation in hours, including Jenga whipping and belt assembly ([HIL-SERL, Luo et al., 2024](https://arxiv.org/abs/2410.21845)).
- **The best path for sim-to-real followed by online fine-tuning.** A study of 100 real-world runs on 3 platforms used SAC with BRO critics throughout, and online SAC fine-tuning clearly beat zero-shot SAC and PPO. It found these choices **critical**:
  - retain sim replay data and anneal the sim/real mix;
  - warm-start the buffer;
  - delay actor updates (1 actor step per 20 critic steps, actor LR 1e-5);
  - pretrain with many environments (8192 transferred well, 128 transferred poorly).

  Source: [As et al., 2026](https://arxiv.org/abs/2602.20220) **(preprint)**.
- **Strong on high-DoF tasks:**
  - FlashSAC tested over 60 tasks in 10 simulators. It was only slightly better than PPO on low-DoF tasks (Franka, ANYmal, Go2) but clearly better on Allegro, Shadow Hand, G1, H1, and T1. It used 50M steps against PPO's 200M, with about 3× less compute ([Kim et al., 2026](https://arxiv.org/abs/2604.04539) **(preprint, self-reported)**).
  - Tuned SAC **surpasses** PPO on H1 and G1 humanoid rough-terrain locomotion ([Sabatini, Li & Hutter, 2026](https://arxiv.org/abs/2605.24975) **(preprint)**).
- **Faster sim-to-real humanoid training with modern recipes:**
  - FastSAC/FastTD3 train G1 and T1 locomotion with domain randomisation in **about 15 minutes on one RTX 4090** ([Seo et al., 2025](https://arxiv.org/abs/2512.01996) **(preprint)**).
  - FlashSAC reaches stable real-world flat-ground walking in about 20 minutes against about 3 hours for PPO, and 15 cm stairs in about 4 hours against about 20 hours for PPO ([Kim et al., 2026](https://arxiv.org/abs/2604.04539) **(preprint, self-reported)**).
  - For athletic sprinting, TD3 and SAC converged in about 12 hours against 36–48 hours for PPO ([arXiv 2609.11083](https://arxiv.org/abs/2609.11083) **(preprint)**).
- **Entropy-driven exploration** helps on dense, multi-term rewards. Sabatini et al. credit it for SAC beating PPO on humanoids.
- **Uses old data.** Demonstrations, interventions, and prior runs can all go in the buffer.

**Cons**
- **Off-the-shelf SAC fails in massively parallel sim.** Sabatini et al. report that vanilla SAC "has consistently failed to match PPO" there and identify three root causes:
  1. Badly scaled action bounds push the tanh-Gaussian onto the joint limits, which makes early exploration erratic.
  2. Timeouts are treated as terminations, or the critic bootstraps from the post-reset state.
  3. Noisy 1-step TD targets on rough terrain.

  Their fixes are bounds from soft joint limits, σ₀=0.15, timeout-aware targets, and 5-step returns. With these, SAC matches PPO on 5 quadrupeds using **one shared config**. [Raffin's ICLR 2026 blog post](https://iclr-blogposts.github.io/2026/blog/2026/sac-massive-sim/) independently found the action-space issue to be the main culprit ([Part II](https://araffin.github.io/post/tune-sac-isaac-sim/)). His fixes were bounds from PPO action percentiles, gSDE, n=3 returns, and tuning for wall-clock time.
- **A persistent wall-clock gap for plain SAC.** Even when reward matches, Sabatini et al. report that "a significant wall-clock gap persists." The causes are larger networks and about 8 sampled transitions per collected transition. Closing the gap takes systems-level recipes (FastSAC, FlashSAC, [PQL, ICML 2023](https://arxiv.org/abs/2307.12983)).
- **Higher seed variance.** Sabatini et al. found this, and on G1 a high reward did not always mean natural motion.
- **Plasticity loss and late instability at scale.** FastSAC rises fast but becomes less stable later in training ([FastDSAC, arXiv 2606.31691](https://arxiv.org/abs/2606.31691) **(preprint, self-reported)**).
- **Rewards tuned for PPO carry over poorly.** Raffin attributes SAC's residual gap partly to PPO-tuned reward weights.
- **Weak multi-task scaling.** The critic becomes the bottleneck because of gradient conflict ([Joshi et al., 2025](https://arxiv.org/abs/2507.23172)).

---

## 4. Making SAC work in GPU-parallel sim: the 2025–26 recipe

The papers above agree on these points:

| Ingredient | Evidence |
|---|---|
| **Sensible action bounds.** Derive them from soft joint limits or PPO action percentiles, not the unbounded or hard range | Sabatini '26; Raffin '25; Seo '25 |
| **Small initial σ** (≈0.15) and a low initial α (≈1e-3) | Sabatini '26; Seo '25 |
| **Timeout-aware bootstrapping**: store the pre-reset next state | Sabatini '26 |
| **n-step returns** (3–5) | Sabatini '26; Raffin '25 |
| **Large batches** (2k–8k) and large buffers (5M–10M) | Seo '25; Kim '26; Sabatini '26 |
| **Normalisation**: observation norm plus LayerNorm, or BatchNorm/WeightNorm | Seo '25; Kim '26; CrossQ+WN ([Palenicek et al., NeurIPS 2025](https://arxiv.org/abs/2502.07523)) |
| **Bigger critics**: [1024,512,256] up to about 2.5M parameters | Sabatini '26; Kim '26; BRO ([Nauman et al., 2024](https://arxiv.org/abs/2405.16158)) |
| **Distributional critic** (C51) and averaged rather than min twin-Q | Seo '25 (FastTD3/FastSAC) |
| **Lower γ** (0.97) for velocity tracking | Sabatini '26; Seo '25 |
| **Tune the UTD ratio for wall-clock time**, not sample count | Raffin '25; Seo '25; As '26 |
| **Simple rewards** with fewer than 10 terms plus a curriculum; avoid PPO-tuned reward stacks | Seo '25; Raffin '25 |

---

## 5. Evidence by robotics domain

| Domain | Current evidence | Edge |
|---|---|---|
| Quadruped locomotion (ANYmal, Go1/Go2, A1) | PPO is the standard (Rudin '21). Tuned SAC matches it on reward but is slower in wall-clock time (Sabatini '26). FlashSAC is slightly better (Kim '26) | **PPO** (simplest); SAC if you plan to fine-tune on hardware |
| Humanoid locomotion (G1, H1, T1) | SAC surpasses PPO (Sabatini '26). FastSAC/FlashSAC are about 10× faster to a deployable sim-to-real policy (Seo '25, Kim '26) | **SAC-family** |
| Whole-body motion tracking | FastSAC/FastTD3 finish long dance clips faster than PPO (Seo '25) | **SAC-family** |
| Dexterous in-hand manipulation (Allegro, Shadow) | PPO+PBT is the established pipeline (DexPBT '23). FlashSAC reports its largest gains here (Kim '26, preprint) | Contested; SAC trending up |
| Arm manipulation in sim (Franka) | Roughly equal on low-DoF tasks (Kim '26) | Either |
| Multi-task (Meta-World, MT50) | MT-PPO far ahead of MT-SAC (Joshi '25) | **PPO** |
| Real-world learning or fine-tuning | SAC/RLPD-based (HIL-SERL '24; As '26) | **SAC** |
| Slow or CPU simulators | PPO "performs particularly poorly" without parallelism (Kim '26) | **SAC** |

---

## 6. Caveats on the evidence

- Most 2025–26 head-to-head results are **arXiv preprints**, and several come from the authors of the proposed method (FlashSAC, FastTD3/FastSAC, FastDSAC). Treat the margins as indicative.
- Benchmarks such as Isaac Lab tasks and legged_gym rewards were **designed and tuned around PPO**. Raffin '25 and Sabatini '26 both note that this biases comparisons toward PPO.
- Comparisons usually hold one axis fixed (samples or wall-clock time) and rarely both. "SAC is more sample-efficient" and "PPO is faster" can both be true on the same task.
- Seed counts are often small; Raffin used 3 seeds.

---

## References

1. Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes Using Massively Parallel Deep RL.* CoRL 2021 (PMLR 164). https://proceedings.mlr.press/v164/rudin22a.html
2. Andrychowicz et al. *What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study.* ICLR 2021. https://arxiv.org/abs/2006.05990
3. Huang et al. *The 37 Implementation Details of Proximal Policy Optimization.* ICLR Blog Track 2022. https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
4. Haarnoja et al. *Soft Actor-Critic: Deep RL for Robotics.* Google Research blog / arXiv 2018–19. https://research.google/blog/soft-actor-critic-deep-reinforcement-learning-for-robotics/
5. Li, Chen, Hong, Ajay, Agrawal. *Parallel Q-Learning: Scaling Off-policy RL under Massively Parallel Simulation.* ICML 2023. https://arxiv.org/abs/2307.12983
6. Petrenko et al. *DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training.* RSS 2023. https://arxiv.org/abs/2305.12127
7. Shahid et al. *Benchmarking Population-Based RL across Robotic Tasks with GPU-Accelerated Simulation.* IEEE CASE 2025. https://arxiv.org/abs/2404.03336
8. Nauman et al. *Bigger, Regularized, Optimistic (BRO).* NeurIPS 2024. https://arxiv.org/abs/2405.16158
9. Palenicek et al. *Scaling Off-Policy RL with Batch and Weight Normalization.* NeurIPS 2025. https://arxiv.org/abs/2502.07523
10. Luo, Xu, Wu, Levine. *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop RL (HIL-SERL).* 2024. https://arxiv.org/abs/2410.21845
11. Seo et al. *FastTD3: Simple, Fast, and Capable RL for Humanoid Control.* 2025. https://arxiv.org/abs/2505.22642
12. Raffin. *Getting SAC to Work on a Massive Parallel Simulator.* ICLR Blogposts 2026. https://iclr-blogposts.github.io/2026/blog/2026/sac-massive-sim/ and Part II: https://araffin.github.io/post/tune-sac-isaac-sim/
13. Joshi, Xu, Liu, Stone, Zhang. *Benchmarking Massively Parallelized Multi-Task RL for Robotics Tasks (MTBench).* 2025 (preprint). https://arxiv.org/abs/2507.23172
14. Seo, Sferrazza, Chen, Shi, Duan, Abbeel. *Learning Sim-to-Real Humanoid Locomotion in 15 Minutes.* Dec 2025 (preprint). https://arxiv.org/abs/2512.01996
15. As, Tirumala, Zurbrügg, Li, Coros, Krause, Wulfmeier. *What Matters for Simulation to Online RL on Real Robots.* 2026 (preprint). https://arxiv.org/abs/2602.20220
16. Kim et al. *FlashSAC: Fast and Stable Off-Policy RL for High-Dimensional Robot Control.* Apr 2026 (preprint). https://arxiv.org/abs/2604.04539
17. Sabatini, Li, Hutter. *Bridging the Gap: Enabling SAC for High Performance Legged Locomotion.* May 2026 (preprint). https://arxiv.org/abs/2605.24975
18. *FastDSAC: Enhancing Policy Plasticity via Constrained Exploration for Scalable Humanoid Locomotion.* 2026 (preprint). https://arxiv.org/abs/2606.31691
19. *Learning Realistic Athletic Sprinting Without Demonstrations.* 2026 (preprint). https://arxiv.org/abs/2609.11083
