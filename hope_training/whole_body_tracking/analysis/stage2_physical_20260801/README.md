# Stage 2 Physical Outcome Training

## Initialization

- Stage-1 source checkpoint: `logs/rsl_rl/hope_pingpong_stage1_operational114/2026-08-01_18-31-02_stage1polishP1_fromB1500_250/model_1749.pt`
- SHA-256: `e648c8827073756aee3867bc6f478f8c3b022e045e7563dfbc0720dd437c8811`
- Actor observation contract remains `hope_pingpong_normal114` (114D).
- The actor is loaded without the Stage-1 optimizer. A parameter anchor protects the Stage-1 policy while PPO adapts it.

## Task Contract

- A rigid PhysX ball follows a valid one-bounce incoming route.
- Contact, net crossing, and opponent-table bounce come from PhysX events, not analytical proxy rewards.
- Ball routes are independent of motion. At ability level zero only the endpoint command distribution is bootstrapped from the Stage-1 motion box.
- Endpoint, impact-inverse velocity, planner perturbation, ball speed, and table workspace are released by measured physical ability.
- Normal completion counts as survival; reset or fall counts as a safety failure.
- Recovery is the best functional READY score after physical route resolution, not a single score sampled at command resampling.
- Unhealthy incidental contact receives only a 5% reward floor.

## Critical Articulation Fix

The first Stage-2 trials used `merge_fixed_joints=false` to retain a dedicated racket body. This changed the Stage-1 articulation into many fixed PhysX bodies and included tiny-inertia racket marker bodies. The policy tilted after about 1.3 seconds, before useful physical outcomes could be learned.

The corrected scene keeps `merge_fixed_joints=true`, matching Stage 1. Existing code then uses `right_wrist_yaw_Link` plus the calibrated racket mount offset for FK and contact capture. The validated conversion is stored as `assets/agibot_a3/usd/model_physical_merged/model_physical.usd`; training loads this read-only USD so two Isaac processes never race in the URDF importer. The corrected smoke test produced:

- zero final tilt/low terminations;
- mean racket position error about 0.098 m;
- mean functional READY about 0.564;
- valid wrist PhysX contact data;
- real contact, net crossing, and opponent bounce events.

The corrected smoke log is `merged_physics_smoke.log`. Logs named `adaptA` or `adaptB` without `merged` are invalid articulation controls and must not be used for policy selection.

## Physical A/B History

Both runs use the corrected merged articulation, conservative PPO (`5e-5` learning rate, KL `0.003`), and the same Stage-2 task.

| Branch | Task | Actor anchor | Noise | Purpose |
|---|---|---:|---:|---|
| mergedA | BalancedA | 10.0 | 0.10 | Maximize Stage-1 skill and stability preservation |
| mergedB | OutcomeB | 3.0 | 0.15 | Allow faster physical impact adaptation |

Historical merged-articulation logs:

- `stage2_mergedA_300.log`
- `stage2_mergedB_300.log`

The physical ability level may advance only after repeated event windows meet contact, functional recovery, and survival gates. In the revised schedule, net remains diagnostic below level 0.6 and bounce remains diagnostic below level 0.85.

## Curriculum Revision After Iteration 100

The first preconverted-USD A/B reached ability 0.4 while remaining safe, but contact fell from about 70% to 31%/26% and net crossing fell to 0.15%/1.3%. One scalar had simultaneously removed the endpoint seed, expanded workspace, and introduced 40% impact-inverse commands.

The revised schedule keeps the same final task but reduces adjacent difficulty jumps:

- ability step: 0.10;
- motion endpoint seed fades through level 0.80;
- impact-inverse blend is `ability ** 2`;
- net is required from level 0.60;
- opponent bounce is required from level 0.85.

Logs `stage2_usdA_300.log` and `stage2_usdB_300.log` are the linear-release control. Revised runs use the `stage2_usd2` prefix.

## Physical Precision Revision

The slower `stage2_usd2` schedule remained safe, but at ability 0.5/0.6 the conditional impact diagnostics showed about 0.12-0.14 m position error, 1.45-1.52 m/s racket velocity error, roughly 40% outgoing-speed ratio, and large direction/normal error. The inherited dense task-space reward still used loose Stage-1 tolerances: 0.28 m, 1.8 m/s, and 0.70 rad.

The `stage2_physq` A/B keeps the stable curriculum and changes feedback only:

- both branches tighten task-space tolerances continuously with physical ability, ending at 0.12 m, 0.85 m/s, and 0.30 rad;
- B adds a real-contact outgoing-velocity/direction quality term;
- the B-only term is multiplied by the impact-inverse command blend, so it is exactly zero for the Stage-1 velocity bootstrap at level zero.

## Active Physical-Precision A/B

| Branch | Run | Actor anchor | Noise | Physical event weights | Purpose |
|---|---|---:|---:|---:|---|
| physqA | `stage2phys_physqA_anchor10_noise010_300` | 10.0 | 0.10 | 3 / 2 | Preserve Stage-1 impact speed while tightening command precision |
| physqB | `stage2phys_physqB_anchor3_noise015_300` | 3.0 | 0.15 | 5 / 3 plus contact quality | Test stronger real-collision adaptation |

Logs:

- `stage2_physqA_300.log`
- `stage2_physqB_300.log`

At approximately iteration 100 both branches were physically stable enough to reach
ability level 0.3. The aligned trailing window showed:

- physqA: contact EMA about 0.53, net EMA about 0.05, recovery about 0.69,
  safety about 0.78, impact speed ratio about 0.84, direction error about 13 deg;
- physqB: contact EMA about 0.58, net EMA about 0.02, recovery about 0.70,
  safety about 0.79, impact speed ratio about 0.44, direction error about 22 deg.

The B branch currently finds lower-speed contact despite its stronger outcome term.
The A branch is the leading long-training candidate, but both continue to the planned
midpoint before final branch selection.

## Final 300-Iteration Screening Result

Both runs completed at `model_299.pt`. The final aligned 30-iteration window was:

| Metric | physqA | physqB |
|---|---:|---:|
| physical ability level | 0.60 | 0.60 |
| contact EMA | 0.181 | 0.195 |
| net-cross EMA | 0.0005 | 0.0005 |
| recovery EMA | 0.617 | 0.552 |
| safety EMA | 0.754 | 0.691 |
| impact speed ratio | 0.638 | 0.524 |
| impact velocity angle error | 17.9 deg | 41.6 deg |
| base angular velocity | 1.220 rad/s | 1.494 rad/s |
| tilted termination | 0.190 | 0.414 |
| q-des velocity violation | 0.039 | 0.068 |

`physqA` wins the A/B comparison, but `model_299.pt` is not approved for long
continuation. The more useful preserved checkpoint is A `model_100.pt`: around
iterations 70-99 it retained contact EMA 0.512, recovery 0.691, safety 0.777,
impact speed ratio 0.850, and impact direction error 13.2 deg.

The common collapse is a curriculum-controller defect. At level 0.60 the net gate
becomes active, while the command distribution has already removed 75% of the
motion endpoint seed, applied 36% impact-inverse velocity, and expanded planner
noise and incoming speed to 60%. Failure to meet contact or net thresholds only
blocks advancement; regression currently checks recovery and safety alone. Both
policies therefore remain stuck at the too-hard level instead of returning to a
learnable distribution. The next run must add capability-specific hysteretic
regression and decouple workspace, velocity, and perturbation unlocks.

## Clean Stage-2 Restart

The defective A `model_100.pt` is retained only as a diagnostic checkpoint. The
corrected Stage 2 restarts from the immutable Stage-1 `model_1749.pt` so no policy
state learned under the defective curriculum is inherited.

Controller corrections:

- net and bounce thresholds ramp continuously instead of switching on at full
  strength at one level;
- contact, active net, and active bounce collapse can regress the curriculum with
  hysteresis, in addition to recovery and safety collapse;
- workspace starts at ability 0.10, impact-inverse velocity at 0.50, ball-speed
  expansion at 0.60, and planner perturbation at 0.75;
- physical ability advances in 0.05 increments after three successful event
  batches and regresses by 0.10 after two failed batches.

Clean A uses that schedule. Clean Slow B delays workspace to 0.15,
impact-inverse velocity to 0.60, ball speed to 0.70, and planner perturbation to
0.85. Both keep the same policy, reward weights, motion manifest, seed checkpoint,
and 114D observation contract. The 1500-iteration process has a mandatory
selection check at iteration 300 rather than treating 300 as final convergence.

## Clean Restart Stop Decision

The clean runs were stopped after A iteration 654 and Slow B iteration 651. The
new controller behaved correctly: once net capability was insufficient, both
levels regressed and retried between roughly 0.51 and 0.57 instead of remaining
stuck at 0.60. The policy itself did not improve across those retries. From the
iteration-280 window to the iteration-640 window, A contact fell from about 0.47
to 0.32, net crossing from 0.032 to 0.004, impact speed ratio from 0.75 to 0.62,
and tilted terminations rose from 0.15 to 0.41. Slow B showed the same plateau
with weaker impact speed.

The selected diagnostic checkpoint is Clean A `model_200.pt`. Around iterations
180-209 it achieved contact 0.49, net crossing 0.105, recovery 0.687, safety
0.783, impact speed ratio 0.853, impact direction error 10.5 deg, and tilted
termination 0.074. It is not yet an approved deployment policy; it is the best
Stage-2 checkpoint for evaluating the remaining physical-outcome bottleneck.

## Workspace Causal Audit And Center-Contact Revision

Clean A `model_200.pt` was evaluated with ability frozen at zero while only the
workspace level changed. This isolates reach/generalization from impact-inverse,
ball-speed, and planner-noise effects.

| workspace | resolved contact | center / serve | center / contact | net cross | ready after contact | outgoing angle error |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.737 | 0.359 | 0.605 | 0.163 | 0.997 | 31.7 deg |
| 0.25 | 0.717 | 0.301 | 0.515 | 0.105 | 0.984 | 37.7 deg |
| 0.50 | 0.422 | 0.126 | 0.377 | 0.025 | 0.978 | 48.8 deg |
| 0.75 | 0.325 | 0.100 | 0.414 | 0.015 | 0.952 | 49.9 deg |

The failure is primarily expanded-workspace center contact and physical outgoing
quality, not post-contact recovery. The previous physical curriculum counted any
racket collision, including rim contacts, as contact capability. At workspace
0.50, 62.3% of contacts were rim/edge contacts, so this definition incorrectly
unlocked harder commands.

Revision:

- physical capability contact now requires radial error <= 0.061 m;
- net and bounce capability count only after a center-contact latch;
- raw-contact and center-contact EMAs are logged separately;
- contact/net/bounce rewards are multiplied by a smooth face-center quality;
- task-space position/velocity/normal tolerances tighten to 0.07 m, 0.65 m/s,
  and 0.22 rad at full ability without adding stronger whole-body constraints;
- contact-based curriculum thresholds begin from center-contact capability.

A 25-iteration runtime check confirmed the new event contract. A warm start from
`model_200.pt` retained better impact execution than a Stage-1 restart (speed
ratio 0.75 versus 0.57, velocity angle error 7.4 versus 18.8 deg), so the next
controlled run compares these two initializations under the identical revised
task.

## Mid-run net-quality audit

The first center-contact comparison was stopped around iterations 380/410. It
rejected rim contacts correctly, but center contact alone could advance ability
until level 0.65. Meanwhile, the physical outgoing-quality reward was multiplied
by an inverse-impact blend that stayed zero below ability 0.50. Workspace
expansion therefore drove net crossing toward zero while contact EMA stayed high.

The follow-up contract starts the physically solved impact command and its
outgoing-quality reward at a 0.30 blend. Only the first 0.05 ability increment can
bootstrap from center contact alone; later increments require a net rate that
ramps to the full threshold by ability 0.50. This keeps the Stage-1 action prior
while preventing contact-only workspace expansion.

The 0.30 warm smoke retained recovery and safety but reduced center contact and
net crossing relative to the incoming `model_200.pt` behavior. The selection
experiment therefore fixes initialization to `model_200.pt` and compares 0.10
versus 0.20 initial inverse-impact/quality blends. These task variants differ
only in those two scalar overrides.

## A10/B20 300-iteration selection

Both runs peaked around iteration 100 and then degraded. By iteration 299,
outgoing direction error had reached roughly 77 degrees for A10 and 71 degrees
for B20 while net EMA had fallen to 0.013 and 0.003. The terminal checkpoints
are rejected.

The iteration-100 checkpoints and incoming `model_200.pt` were evaluated for
1800 steps with the same seed, fixed workspace/ability zero, and the common A10
command contract:

| checkpoint | contact / serve | center / contact | net / serve | recovery after contact | outgoing speed ratio | planner velocity angle | peak torso ang. vel |
|---|---:|---:|---:|---:|---:|---:|---:|
| incoming model_200 | 0.5952 | 0.5603 | 0.1172 | 0.9939 | 0.8270 | 13.66 deg | 0.6979 rad/s |
| A10 model_100 | 0.5512 | 0.5798 | 0.1079 | 0.9934 | 0.7367 | 21.26 deg | 0.7794 rad/s |
| B20 model_100 | 0.5596 | 0.6361 | 0.1309 | 0.9889 | 0.7716 | 16.71 deg | 0.7997 rad/s |

B20 `model_100.pt` is the selected Stage-2 diagnostic candidate. It improves
center contact and real net crossing without materially reducing recovery, but
it does not improve planner execution and increases torso angular velocity.
Therefore this is a targeted physical-contact improvement, not yet a generally
better deployment model. Further continuation under the same objective is not
approved because both policies optimize toward slow, misdirected contact after
iteration 100.
