# Stage-2 Decoupled Command Executor V5

Date: 2026-08-02

## Objective

Train one policy to execute a generic strike command accurately without binding the
policy to the production HOPEPlanner implementation or to a motion-derived ball
route. The command contract remains 114D and includes target position, velocity,
normal, and time-to-strike.

The intended separation is:

1. **Physical route truth**: a randomized rigid PhysX ball route, independent of
   the motion clip.
2. **Generic command oracle**: after the real PhysX table bounce, repeatedly use
   the latest position and velocity of that same ball to estimate its strike-plane
   crossing. A randomized return landing/time is converted into a feasible racket
   velocity and normal. This is not the production HOPEPlanner.
3. **Policy execution**: the actor sees only the 114D observation/command contract
   and must realize position, timing, velocity, and normal.
4. **Outcome truth**: contact, net crossing, opponent bounce, and outgoing velocity
   are measured from the actual PhysX collision. Rewriting a command cannot create
   outcome credit.

Spin is intentionally excluded because the real Motive pipeline cannot observe it.

## Implementation

Two isolated configurations were added:

- `HOPEPingPongStage2CommandExecutorCore114V5`: moderate initial route and return
  distribution.
- `HOPEPingPongStage2CommandExecutorDiverse114V5`: wider route timing, bounce,
  landing, and robot-dynamics distribution.

Both configurations enforce the following contract:

- `route_geometry_mode=independent`: motion is an action prior only.
- Motion-box strike-position seeding is disabled.
- A post-bounce command is revised at 50 Hz from the latest PhysX state and frozen
  at a 0.20 s time-to-strike horizon.
- Production planner perturbation is disabled in this first executor experiment.
  With no ball state in the actor observation, independently perturbing the visible
  command away from hidden outcome truth would create an unobservable target and
  teach command averaging/ignoring rather than robustness.
- Command tracking is scored only at a real PhysX contact. Physical return credit
  still comes only from the measured collision outcome.
- The existing two-motion manifest remains the action prior, but does not define
  route position, velocity, timing, or station.

Relevant files:

- `cfg/task/HOPEPingPongStage2CommandExecutorCore114V5.yaml`
- `cfg/task/HOPEPingPongStage2CommandExecutorDiverse114V5.yaml`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/physical_ball_shadow_command.py`
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py`
- `scripts/launch_stage2_closed_loop_command_member.sh`
- `scripts/eval_stage2_closed_loop_command_member.sh`

## Timing Defect Found

`RacketTargetCommand._compute_strike_timing()` was invoked by both metric and
command updates. It also decremented the physical override time-to-strike, so one
20 ms policy step consumed 40 ms. Older physical evaluations therefore contacted
the ball with an artificial timing error around -0.2 to -0.4 s.

The decrement now occurs exactly once in `_update_command()`. Corrected evaluations
show median/mean contact time-to-strike close to zero (about -10.5 to +12.1 ms).
Static regression coverage prevents the double decrement from returning.

## Verification

- Unit and contract tests: `39 passed`.
- Python compile and `git diff --check`: passed.
- GPU smoke: 8 environments, 1 iteration, 114D actor input, all V5 overrides
  applied, repeated post-bounce revisions observed, no NaN.
- Route-generation failures in fixed evaluation: 0-5 out of roughly 1,300 serves;
  no unalignable routes.
- Roughly 17 command revisions occur after each bounce. The median subsequent
  revision is 0.49-0.67 mm in position and 0.15-0.17 ms in timing. The first
  revision can be much larger because it corrects the pre-bounce analytic estimate
  with the measured PhysX bounce.

## Fixed-Seed Results

All rows use 256 environments, 900 evaluation steps, seed 8830. `zero` is the
unchanged Stage-2 baseline `model_200.pt`; `100 it` is a paired 100-iteration
fine-tune initialized from the same actor with no optimizer state and actor-anchor
coefficient 2.0.

| Metric | Core zero | Core 100 it | Diverse zero | Diverse 100 it |
|---|---:|---:|---:|---:|
| Serves | 1339 | 1347 | 1315 | 1359 |
| Contact / serve | 25.91% | 29.10% | 22.51% | 22.15% |
| Net cross / serve | 0.075% | 0.000% | 0.000% | 0.074% |
| Opponent bounce / serve | 0.075% | 0.000% | 0.000% | 0.074% |
| Physical recovery / resolved contact | 98.41% | 99.12% | 99.26% | 98.91% |
| Contact time-to-strike | -10.5 ms | -3.3 ms | -0.8 ms | +12.1 ms |
| Position error at contact | 15.17 cm | 12.95 cm | 13.27 cm | 12.95 cm |
| Velocity error at contact | 1.178 m/s | 0.992 m/s | 1.175 m/s | 1.008 m/s |
| Velocity direction error | 42.38 deg | 33.95 deg | 38.74 deg | 29.81 deg |
| Normal error | 24.19 deg | 23.36 deg | 26.98 deg | 26.19 deg |
| Outgoing velocity error | 3.372 m/s | 3.101 m/s | 3.321 m/s | 3.046 m/s |
| Outgoing direction error | 70.26 deg | 61.47 deg | 66.05 deg | 56.01 deg |
| Usable-center contacts / all contacts | 16.7% | 16.3% | 20.9% | 26.6% |

The fixed evaluation is the primary comparison. Per-iteration training event rates
are batched/reset differently and are not interchangeable with these per-serve
physical rates.

## Interpretation

The experiment supports the command-executor design:

- Core improves contact rate and position/velocity/direction tracking while
  preserving recovery.
- Diverse improves velocity direction and central-contact quality despite not
  increasing total contact rate. Its single net-cross event is evidence of
  feasibility, not yet statistically stable return ability.
- Timing is no longer the current bottleneck. The dominant remaining errors are
  racket velocity magnitude/direction and normal, which propagate into a roughly
  3.0 m/s outgoing-velocity error.
- The old model's poor zero-shot transfer is expected: it was optimized on the old
  motion-coupled command distribution. The 100-iteration improvements show that
  the new generic command distribution is learnable rather than malformed.

## Remaining Structural Limit

The deployment-compatible `v4_wire_compatible` mode derives the actor-visible
normal from target-velocity direction. Therefore 114D exposes a normal vector, but
does not provide a fully independent normal degree of freedom in this wire contract.
This preserves current deployment compatibility but can cap independent control of
racket speed and face angle. It should not be changed until the deployed command
message and fallback/validation rules are changed at the same time.

## Decision

Do not start an unmonitored long run from the fully Diverse distribution yet. One
net crossing in about 1,350 serves is too sparse, and 100 iterations are not enough
to establish that physical return success is monotonic.

The next training member should use an **ability-driven Core-to-Diverse expansion**:

- start from the Core ranges;
- expand route width/speed and return landing/time only after fixed-window contact,
  center-contact, command-error, recovery, and safety gates pass;
- regress one level if recovery/safety fails;
- keep motion independent and keep command tracking plus physical outcome as
  separate diagnostics;
- add bounded real-planner latency/noise only after command execution is reliable,
  and keep each perturbation inside the measured deployment contract.

This progression broadens the distribution without overfitting one formula and
without asking the actor to solve a wide command manifold before it can execute the
core command accurately.

## Artifacts

- Core run: `logs/rsl_rl/hope_pingpong_stage2_command_executor114_v5/2026-08-02_03-40-06_cmdexec_v5_core_from200_256x100_s8842/model_99.pt`
- Diverse run: `logs/rsl_rl/hope_pingpong_stage2_command_executor114_v5/2026-08-02_03-40-14_cmdexec_v5_diverse_from200_256x100_s8842/model_99.pt`
- Corrected zero-shot and post-training physical JSON files are in this directory.

## V6 Ability Curriculum

V6 replaced the fixed Core/Diverse split with one ability-driven distribution. The
same physical completion records drive contact, centered/aligned contact, net,
bounce, recovery, and safety EMAs. Route geometry and table workspace expand only
after all active gates pass and regress after capability loss.

The 300-update V6 run established two facts:

- center contact rose, but the actor still preferred easy collision credit after
  about 200 updates;
- ability oscillated near 0.30--0.35 because contact/recovery passed while real
  net crossing did not.

V6A and V6B ended byte-identical because the optional aligned-contact curriculum
gate never became the binding gate before net crossing. This ruled out a long V6
A/B run and motivated changing physical credit rather than adding another gate.

## V7 Precision Credit Decision

Both V7 members start actor-only from V6A `model_200.pt`, use the same seed and
keep route, command, motion, dynamics, lifecycle, and curriculum identical.

- **V7A** raises real-contact joint planner-alignment weight from 1.5 to 6.0.
- **V7B** includes V7A and changes physical contact credit from
  `contact=1.0, quality=1.5` to `contact=0.5, quality=3.0`. A low-quality
  collision is worth less; a command-consistent outgoing collision can be worth
  more. The resolved inherited quality tolerances are 2.0 m/s and 45 degrees.

At updates 270--299, V7B reached:

- center-contact EMA 8.54%;
- aligned-contact EMA 5.52%;
- net and opponent-bounce EMA 0.226%;
- recovery EMA 58.20%;
- safety EMA 77.54%.

Net exceeded its active 0.173% threshold. Ability remained at 0.35 only because
safety was slightly below the 0.78 advance threshold. This is desired: broader
routes cannot be unlocked by trading away upright behavior.

### Fixed physical evaluation at update 200

All rows use 256 environments, 900 steps, seed 8830. Core fixes workspace/ability
at 0; Full fixes both at 1. Full is deliberately out of distribution because the
300-update training curriculum reached only ability 0.35.

| Metric | V6A Core | V7A Core | V7B Core | V6A Full | V7A Full | V7B Full |
|---|---:|---:|---:|---:|---:|---:|
| Contact / serve | 33.04% | 35.09% | **37.89%** | 17.54% | 15.29% | 15.07% |
| Center / all contact | 20.09% | 21.14% | **26.83%** | 22.45% | 34.30% | **35.47%** |
| Net cross / serve | 0.369% | 0.223% | **0.512%** | **0.716%** | 0.517% | 0.371% |
| Opponent bounce / serve | 0.295% | 0.223% | **0.512%** | **0.716%** | 0.517% | 0.371% |
| Physical recovery | **96.04%** | 92.47% | 94.99% | **89.70%** | 88.77% | 86.24% |
| Position error | **12.95 cm** | 13.27 cm | 13.48 cm | 15.37 cm | 14.01 cm | **13.10 cm** |
| Velocity error | 0.960 m/s | 0.853 m/s | **0.827 m/s** | 1.043 m/s | 0.974 m/s | **0.945 m/s** |
| Velocity direction error | 29.13 deg | **24.33 deg** | 25.58 deg | 30.65 deg | 28.80 deg | **27.13 deg** |
| Normal error | 24.35 deg | **21.55 deg** | 23.66 deg | 26.31 deg | **24.13 deg** | 26.11 deg |
| Contact timing | **+2.6 ms** | +9.7 ms | +23.1 ms | +10.8 ms | **+10.3 ms** | +24.9 ms |
| Outgoing velocity error | 3.050 m/s | 2.861 m/s | **2.814 m/s** | 3.020 m/s | **2.928 m/s** | 2.986 m/s |
| Outgoing direction error | 58.64 deg | 53.37 deg | **51.13 deg** | 54.72 deg | 53.17 deg | **52.41 deg** |

V7B is the long-training choice. It is the only member that improves Core contact,
center quality, net/bounce, command execution, and outgoing error together while
retaining about 95% physical recovery. Its Full regression is expected until the
ability curriculum actually expands to that distribution; it must not be hidden
by evaluating only Core.

V7A update 299 regressed center quality, command error, timing, recovery, and
reset count while increasing raw contact, reproducing the V6 credit drift. V7B
update 299 Core evaluation was stopped after 25 minutes of pathological PhysX CPU
cost; it is not a deployment or long-training candidate. Checkpoint selection is
therefore based on the fully completed fixed evaluation of V7B update 200.

## Stage-2 Long Training

Two reproducibility runs resume the complete V7B update-200 checkpoint (actor,
critic, optimizer, exploration state, and iteration). The actor anchor is
re-centered on that verified checkpoint instead of remaining centered on V6A.
The two runs differ only in environment seed:

- seed 8862: `logs/rsl_rl/hope_pingpong_stage2_command_precision114_v7/2026-08-02_07-32-28_stage2_v7b_long_from200_reanchor_256x3000_s8862`
- seed 8873: `logs/rsl_rl/hope_pingpong_stage2_command_precision114_v7/2026-08-02_07-32-40_stage2_v7b_long_from200_reanchor_256x3000_s8873`

The source optimizer step was 1608; the first saved long-run checkpoints report
step 1616, proving full optimizer continuation rather than actor-only restart.
Ability starts at zero because environment curriculum state is not checkpointed;
the policy must re-pass physical contact, recovery, and safety gates before route
expansion. This intentionally prevents a resumed actor from being dropped directly
into an unverified Full distribution.

### Full-state resume rejection at update 500

The two full-state runs were stopped at `model_500.pt`. Both seeds reproduced the
same failure mode while route difficulty was still fixed at Core: centered contact
increased, but safety and episode duration declined and ability remained at 0.10.

| Metric, updates 440--499 | seed 8862 | seed 8873 |
|---|---:|---:|
| Center-contact EMA | 9.71% | 10.28% |
| Aligned-contact EMA | 6.53% | 6.55% |
| Safety EMA | 75.09% | 76.36% |
| Mean episode length | 376.4 | 377.5 |
| Ability | 0.10 | 0.10 |

The checkpoint contains actor, critic, optimizer, and PPO iteration, but not the
environment curriculum controller. Restoring optimizer/critic state learned near
ability 0.35 while resetting the command distribution to ability 0 creates an
incomplete training-state resume. The safety gate correctly blocks expansion, but
cannot itself stop PPO from trading Core safety for more contact.

The replacement long runs therefore keep the verified V7B update-200 actor and
anchor, while resetting critic and optimizer so all learned state sees the same
fresh ability-0 curriculum:

- seed 8862: `logs/rsl_rl/hope_pingpong_stage2_command_precision114_v7/2026-08-02_08-25-09_stage2_v7b_freshopt_from200_256x3000_s8862`
- seed 8873: `logs/rsl_rl/hope_pingpong_stage2_command_precision114_v7/2026-08-02_08-25-17_stage2_v7b_freshopt_from200_256x3000_s8873`

The rejected runs and their `model_500.pt` checkpoints remain intact as controls.

### Fresh optimizer result and V8 safety-credit screen

Resetting critic and optimizer fixed the initial resume mismatch. During updates
85--164, both fresh runs crossed the 0.78 safety gate and expanded route ability.
However, the same contact-versus-safety drift returned later. At updates 440--499:

| Metric | fresh seed 8862 | fresh seed 8873 |
|---|---:|---:|
| Center-contact EMA | 8.40% | 11.62% |
| Aligned-contact EMA | 4.88% | 5.16% |
| Net/bounce EMA | 0.260% | 0.203% |
| Safety EMA | 76.66% | 74.71% |
| Mean episode length | 364.9 | 343.7 |
| Ability | 0.30 | 0.20 |

This separates two causes: full-state continuation was invalid when curriculum
state was lost, but a clean optimizer alone does not prevent PPO from exchanging
Core safety for more contact. The curriculum gate controls distribution expansion;
it is not an optimizer constraint. Dense command rewards still retained 30--35%
credit at poor impact health, physical contact/alignment used nonzero health floors,
and failed post-contact settlement cost only 20% of the corresponding success.

V8 tests one minimal correction without changing route, motion, actor input,
command refresh, physics, lifecycle, or physical outcome definitions:

- V8A progressively tightens dense command health floors from 0.25--0.30 to
  0.05, removes artificial positive health floors from physical contact and
  planner-alignment credit, and raises the curriculum safety regression floor
  from 0.65 to 0.74.
- V8B is identical to V8A except that a physical hit which fails to settle back
  to READY pays 50% instead of 20% of its outcome value.

Both use the same actor-only V7B update-200 initialization and seed 8862:

- V8A: `logs/rsl_rl/hope_pingpong_stage2_safety_credit114_v8/2026-08-02_09-56-35_stage2_v8a_soft_safety_credit_from_v7b200_256x3000_s8862`
- V8B: `logs/rsl_rl/hope_pingpong_stage2_safety_credit114_v8/2026-08-02_09-56-39_stage2_v8b_deferred_safety_credit_from_v7b200_256x3000_s8862`

The A/B selection criterion is not total reward. A candidate must preserve
contact/aligned-contact and net/bounce while keeping safety above 0.78 and avoiding
the episode-length decline observed in both V7B controls.

### V8 result: hard termination was missing from delayed credit

V8B was stopped after `model_350.pt`. In updates 255--354 it reduced centered
contact from 9.77% to 5.60% relative to V8A, but safety also fell slightly from
75.94% to 75.64%. The larger delayed failure cost therefore suppressed ordinary
unsettled hits without fixing falls.

Code inspection found a concrete accounting defect. The recovery settlement term
only failed a pending hit on deadline expiry or target resampling. Isaac Lab
computes hard terminations before rewards, but the term did not read that signal;
`reset()` then cleared the pending hit after base tilt, low base, table contact, or
persistent action overflow. Consequently a contact followed by a hard reset paid
no delayed recovery failure at all.

V9 fixes only this defect:

- a pending physical hit plus `termination_manager.terminated` is a settlement
  failure on the terminal step;
- time-limit truncation is not treated as a hard failure;
- no-contact falls remain owned by the existing termination penalty;
- `physical_recovery_terminal_failure_event` records the newly visible path.

The V9 config inherits V8A and uses `failure_cost=0.50`. Static contracts and the
host Isaac runtime smoke pass. Its long run starts actor-only from the same V7B
update-200 checkpoint, seed 8862, actor anchor 2.0, and fresh optimizer as V8A:

- V9: `logs/rsl_rl/hope_pingpong_stage2_safety_credit114_v8/2026-08-02_11-02-59_stage2_v9_terminal_safety_from_v7b200_retry_256x3000_s8862`

V9 also exposed a magnitude-coupling problem. Its single `failure_cost=0.50`
applied both to an ordinary READY timeout and a hard safety reset. Increasing that
shared value suppresses all unsettled contacts, while leaving it small cannot
repay the immediate physical outcome and planner-alignment value of a high-quality
hit. V10 separates these paths:

- ordinary timeout/next-target interruption: `failure_cost=0.20`;
- contact followed by hard termination: `terminal_failure_cost=4.0`.

For a contact-tier shot the latter contributes an unweighted `-4`, or `-8` after
the settlement reward weight, in addition to the existing hard-termination
penalty. This targets the unsafe credit loop without changing contact exploration
for trajectories that remain upright. V9 continues as the weak-debit control;
V10 uses the same source actor, seed, optimizer initialization, and 3000-update
budget:

- V10: `logs/rsl_rl/hope_pingpong_stage2_safety_credit114_v8/2026-08-02_11-23-05_stage2_v10_hard_terminal_debit_from_v7b200_256x3000_s8862`

At the first comparable 100--129 window, V9 versus V8A had safety
78.08% versus 77.77%, recovery 60.01% versus 59.56%, and centered contact
7.04% versus 8.08%. This confirms that terminal accounting is active, but a
shared 0.50 debit produces only a small safety gain while reducing contact.
V10 must outperform that trade before it is accepted for long training.

### Command-generator boundary and sim-to-real plan

The Stage-2 actor is a generic command executor, not a copy of production
HOPEPlanner. The physical route samples bounce geometry and pre/post-bounce time
independently of motion. After launch, actual PhysX state owns the trajectory. At
50 Hz, until the 0.20 s freeze horizon, the same ball's latest post-bounce position
and velocity are propagated to the strike plane and used to refresh strike
position, time, realizable racket velocity, normal, and dynamic station.

Desired outgoing landing position and flight time are sampled over their own
ranges. Thus incoming routes map to multiple feasible command tuples rather than
one memorized planner formula. Motion remains an action prior only. Physical
contact, net crossing, opponent bounce, and outgoing velocity remain collision
outcomes and are never replaced by analytic success labels.

The clean command-execution stage deliberately does not inject a command that is
inconsistent with its physical ball. Such corruption encourages the actor to
ignore command channels. Sim-to-real robustness is added after clean execution
passes, using bounded and separately measurable axes: ball/table/racket physics,
state-estimation noise, timestamp/latency/update cadence, command filtering, and
actuator variation. Each robust model must retain clean-command performance and
pass held-out route and command counterfactual evaluations. Ball spin is excluded
from the actor contract because Motive does not observe it.
