# Stage 1 PlannerExecutor

Date: 2026-08-02

## Why this restart exists

The previous experiments mixed command tracking, analytic return outcomes,
rigid-ball outcomes, persistent READY rewards, recovery shaping, and several
overlapping stability terms. That produced four recurring failure modes:

1. Dense motion/stability rewards dominated one-frame impact outcomes.
2. A policy could earn return by standing still or by making an unsafe strike
   and recovering only at the final frame.
3. Position, velocity, normal, and timing were rewarded independently, so the
   actor did not have to execute one coherent Planner command at impact.
4. Stage-2 physical difficulty was introduced before Stage-1 command execution
   was reliable, making Planner error, policy error, and collision error
   impossible to separate.

The historical 114D change also did not guarantee improvement by itself. In
the current deployment-compatible wire contract, actor-visible normal is
reconstructed from target-velocity direction. It is observable, but not an
independent Planner degree of freedom. Stage 1 therefore measures the exact
command the actor actually receives; Stage 2 will later evaluate physical truth.

## Frozen two-stage boundary

Stage 1 has no rigid ball and does not optimize net crossing, landing, or
outgoing-ball velocity. It learns repeated execution of clean, coherent
commands at `x_hit=+0.20 m`:

- target position and timing;
- racket speed ratio and velocity direction;
- actor-visible racket normal;
- healthy impact posture;
- operationally feasible q-des/action output;
- safe recovery to a reusable state after every targeted attempt;
- balanced forehand/backhand exposure.

Stage 2 will preserve the Stage-1 command score as an auxiliary contract and
add a rigid one-bounce ball, physical contact/net/landing truth, and measured
Planner perturbations. The actor cannot correct arbitrary unobservable Planner
errors, so Stage-2 noise must stay inside the real measured error envelope.

## Reward lifecycle

Dense rewards provide only a reachable path to impact. The high-value events
are one-shot impulses in RewardManager units:

1. `exact_impact_planner_task_space_alignment` jointly scores position, speed,
   direction, normal, virtual contact, and impact health.
2. A targeted attempt opens the recovery path even after a miss.
3. `safe_recovered_planner_command` is paid only after the full recovery window
   remained safe and reached an operationally reusable terminal state.
4. Unsafe and incomplete recovery receive one-shot debits.
5. Skipping a feasible shot and making a targeted virtual miss receive
   one-shot debits.

The main cycle reward cannot be earned from net/landing diagnostics, prolonged
READY time, or an unsafe path that happens to look upright at its last frame.

## Command curriculum

The task uses the two audited static forehand/backhand clips equally. Motion is
an upper-body prior and phase/side label; target position is sampled from a
table workspace independent of the motion box. A no-spin, one-bounce analytic
route generates a coherent inverse-impact command without spawning a rigid
ball. Workspace, route speed, and inverse-command blend advance only from
measured contact, both-side exposure, recovery, safety, targeted attempts, and
station saturation. Analytic net and landing rates are excluded from the gate.

## Verification

- Unit/contract suite: `297 passed, 1 skipped`.
- Runtime smoke:
  `logs/rsl_rl/hope_pingpong_stage1_planner_executor114/2026-08-02_22-43-23_stage1exec_peakimpulse_smoke_64x2_s9760`
- Runtime audit: all 16 checks passed, including impulse accounting for the
  incremental recovery-peak term.
- Resolved actor/action dimensions: `114 / 31`.
- Resolved active commands: `motion`, `racket_target`; no rigid-ball command.
- Resolved active rewards: 31; exact whitelist match.

## Short-run decision gates

The first dual-seed screen is diagnostic, not a long-training commitment. A
candidate can continue only if both seeds show the same direction of change:

- falling and hard termination decrease;
- forehand and backhand virtual-contact rates both rise;
- exact-impact position, speed, direction, and normal errors improve together;
- safe terminal settlement rises while unsafe settlement stays low;
- action clamp, operational-margin excess, and q-des violation do not grow;
- reward improvement is not explained by reduced targeted-attempt rate.

The formal endpoint thresholds are stored in
`cfg/contracts/hope_stage1_planner_executor_v1.yaml`.

## Screen 1 result and correction

The first two scratch screens were stopped at iterations 229 and 210. Both
learned survival but entered the same wrong-swing optimum:

- seed 9711: mean episode length about 383/500 at iteration 224, tilt
  termination about 3.84, but impact position error 0.82 m and no targeted
  attempt/contact;
- seed 9712: mean episode length about 304/500 at iteration 205, impact
  position error 0.81 m and no targeted attempt/contact.

The old convergent scratch control also had no contact at iteration 200, but
its impact position error had already fallen to 0.48 m and it began producing
targeted attempts near iteration 225. Target position and target velocity
distributions were comparable. The causal difference was reward reachability:
the first PlannerExecutor draft removed independent impact-position shaping,
while the joint position/velocity/normal kernel was nearly zero when all three
components were poor.

The corrected screen adds one shared `racket_position` bootstrap with a 0.35 m
kernel. It has no side-specific bonus and, under Stage-1 zero Planner noise,
its hidden strike point equals the actor-visible position command. Physical
outcomes remain forbidden. Stage 2 must remove or replace this hidden-truth
bootstrap once Planner perturbation is enabled.

That correction was screened with two seeds through roughly 140--150
iterations and was still insufficient. The hard termination penalty contributed
about `-0.121` per episode, five times the convergent scratch control, while the
joint command crossfade contributed only about `0.0038` versus `0.017` in the
control. Both policies reduced falls without producing a targeted attempt.

The second correction keeps all hard safety terminations but restores the
termination scalar to `-12`. It provides low-value position, velocity, and
normal feedback in the same strike window, with nonzero health floors so the
signal remains reachable from scratch. These components are bootstrap terms:
the high-value `exact_impact_planner_task_space_alignment` and
`safe_recovered_planner_command` still require coherent joint execution and a
safe recovered terminal state.

## Balanced-bootstrap screen

Two scratch seeds (9731/9732) completed 300 iterations. Their trailing 20-step
means were:

| metric | seed 9731 | seed 9732 |
|---|---:|---:|
| impact position error (m) | 0.167 | 0.211 |
| racket speed ratio | 1.086 | 0.975 |
| velocity direction error (deg) | 29.2 | 27.4 |
| targeted attempt rate | 0.652 | 0.497 |
| virtual contact rate | 0.201 | 0.189 |
| tilt termination | 5.53 | 4.21 |
| table-touch termination | 41.70 | 37.12 |
| safe terminal settlement rate | 0.0 | 0.0 |

The command bootstrap worked, but this is not a Stage-1 endpoint. In both
seeds, about 99.85% of attempted recovery paths exceeded the base angular
velocity envelope. Table contact then replaced tilt as the dominant hard
termination. The next isolated correction adds an incremental post-attempt
base-angular-velocity peak penalty, post-strike base damping, healthy trunk
support, and a stronger table proximity barrier. None of those terms directly
constrains the right arm or reduces the planner-command rewards.

The static recovery-gradient screen (scratch seeds 9742/9743) reduced table
touch but did not close recovery. Its trailing attempt rates diverged to 32.1%
and 1.7%, safe settlement stayed at zero, and recovery base angular velocity
remained near 4 rad/s. A warm-start control also collapsed exactly when the
actor unfroze after 20 critic-only updates (episode length about 88 to about 8),
so checkpoint fine-tuning is not used as evidence for this Stage-1 design.

The follow-up makes the two early constraints ability driven. The table barrier
starts at one third strength and reaches full strength only as targeted-attempt
EMA rises from 0.01 to 0.08. Post-strike damping is paid only after a swing,
not during ordinary motion hold, and its Gaussian width tightens from 3.0 to
1.2 rad/s over the same measured ability interval. The incremental peak
potential uses a broader 2.0 rad/s scale so it has gradient at the observed
4 rad/s failure instead of immediately saturating.

## Ability-gated recovery screen

Scratch seeds 9751/9752 completed 300 iterations. Their trailing 20-iteration
means were:

| metric | seed 9751 | seed 9752 |
|---|---:|---:|
| impact position error (m) | 0.304 | 0.224 |
| racket speed ratio | 0.952 | 0.972 |
| velocity direction error (deg) | 30.6 | 39.0 |
| normal error (deg) | 32.3 | 26.6 |
| targeted attempt rate | 43.3% | 20.0% |
| virtual contact rate | 17.5% | 1.2% |
| recovery base angular velocity (rad/s) | 2.99 | 3.63 |
| base-angular recovery-envelope violation | 99.95% | 99.80% |
| q-des acceleration violation | 83.68% | 83.34% |
| safe terminal settlement rate | 0.0 | 0.0 |
| tilt termination | 13.62 | 24.70 |
| table-touch termination | 26.43 | 9.47 |

Ability gating restored command exploration: both seeds eventually produced
targeted attempts and speed tracking approached one. It did not close safe
recovery and is not a long-training candidate. Seed 9751 traded tilt for table
touch as command execution improved; seed 9752 learned the same command later
and remained highly seed-sensitive.

The audit then found a units bug in the only attempt-local recovery peak
feedback. `post_contact_ready_peak_ang_vel_excess_increment` is a one-step
potential increment, but RewardManager multiplied it by `dt=0.02` a second
time. Its observed episode contribution was only -0.0028/-0.0016 despite the
near-universal envelope violation.

A same-seed causal screen first enabled impulse accounting at a fixed -0.25
weight. This was rejected at iterations 216/203. For seed 9751, the 160--179
mean recovery angular velocity moved only from 3.289 to 3.233 rad/s, while
targeted attempts fell from 10.2% to 1.9% and virtual contact from 0.64% to
0.02%. It reproduced the undesired stable-but-inactive local optimum.

The current correction preserves impulse units but makes their strength
ability driven. The initial scale is 0.08, so `-0.25 * 0.08 = -0.02`, exactly
matching the old term's effective 50 Hz magnitude. It increases toward -0.25
only after targeted-attempt EMA rises from 0.20 to 0.45. This is capability
driven rather than iteration driven and does not add any hold-phase or
right-arm penalty. Same scratch seeds 9751/9752 are the isolated final screen;
they must retain command exploration while reducing recovery angular velocity
before any long run is accepted.

## Table-touch root cause

The ability-scaled retest was stopped at iterations 136/145 after auditing the
table failure itself. Before the targeted-attempt EMA reaches 0.20, the new
peak term has exactly the old effective weight. Same-seed checkpoints confirm
the trajectories are identical: at seed-9751 iteration 74, old and new both
reported episode length 70.83, tilt termination 49.875, and table termination
8.3438; seed-9752 iteration 89 was also identical. The new peak term therefore
did not create early table entry.

The tracking scene contains only the robot and a flat floor. It has no rigid
table. `table_touch` tests whether torso/arm/wrist link origins or the analytic
racket center enter an axis-aligned table zone; it is not a PhysX table-contact
event. B+deploy had this termination disabled, its table proximity reward was
zero, and its target remained motion-box conditioned. Stage1 PlannerExecutor
instead enables the analytic termination after 25 steps, samples on the fixed
table `x=+0.20 m` plane, and limits dynamic-station x to [-0.02, +0.05] m.

The audited reference clips themselves never place an upper-body link origin
inside the table zone. Entry is learned/fall behavior: initially random forward
falls, and later unsafe forward command execution. As command tracking improves,
the first hard-failure label often moves from tilt to table entry; a higher
table count alone does not mean total hard failures increased. A deployment
candidate needs a rigid-table-only Stage1 scene, real per-link contact metrics,
and impact credit that is invalidated or escrowed by table contact. Increasing
the global table penalty alone previously produced the inactive local optimum.
