# Stage-2 Impact Calibration Screening

## Incoming checkpoint

- Checkpoint: `stage2phys_cleanA_restart_fromP1_1500/model_200.pt`
- Fixed-seed PhysX baseline at workspace/ability level zero:
  contact `0.5952`, net cross `0.1172`, opponent bounce `0.1168`, and
  post-contact recovery `0.9939`.

## Rejected fixed-level calibration A/B

The first A/B increased continuous strike-window velocity feedback while holding
workspace, planner perturbation, and ability at level zero. Both iteration-100
checkpoints reduced contact and net crossing. The stronger physical-outcome B
branch was worse, so neither checkpoint is approved for continuation.

## Rejected impact-credit V2

V2 extended PPO rollouts from 32 to 64 policy steps, reduced imitation only in
the strike phase, and increased near-impact signed velocity feedback. A full
actor/critic load was compared with an actor-only load for 100 updates.

| fixed-seed PhysX metric | incoming | full critic | actor only |
|---|---:|---:|---:|
| contact / serve | 0.5952 | 0.5746 | 0.6061 |
| net cross / serve | 0.1172 | 0.0766 | 0.0922 |
| opponent bounce / serve | 0.1168 | 0.0759 | 0.0915 |
| planner position error | 0.0847 m | 0.0953 m | 0.0838 m |
| planner velocity error | 0.5246 m/s | 0.7082 m/s | 0.5565 m/s |
| planner velocity direction error | 13.66 deg | 19.20 deg | 14.11 deg |
| planner normal error | 12.04 deg | 12.64 deg | 13.07 deg |
| outgoing speed ratio | 0.8270 | 0.7645 | 0.8131 |
| outgoing direction error | 32.96 deg | 37.16 deg | 33.88 deg |
| post-contact recovery | 0.9939 | 0.9911 | 0.9910 |

The actor-only branch preserves contact better, but neither branch improves
impact execution or return quality. Longer credit horizon is therefore not
sufficient by itself. The dense velocity term can improve a local final-window
score even when earlier position/timing errors prevent a useful collision.

## Next controlled test

V3 restores all incoming reward weights. Both branches use the same fixed level,
64-step rollout, actor-only initialization, seed, anchor, and optimizer settings.
The B branch adds exactly one term: a joint planner-alignment score paid only on
a real PhysX contact. It combines realized position, velocity magnitude and
direction, racket normal, contact timing, face-center quality, and impact health.
This separates contact discovery from contact quality without weakening the
existing motion or recovery priors.

## V3 result

Both branches completed 100 updates and were evaluated with the same 256-env,
1800-step, seed-8830 PhysX protocol as the incoming checkpoint.

| fixed-seed PhysX metric | incoming | V3 control | V3 contact alignment |
|---|---:|---:|---:|
| contact / serve | 0.5952 | 0.6159 | 0.6042 |
| net cross / serve | 0.1172 | 0.1150 | 0.1082 |
| opponent bounce / serve | 0.1168 | 0.1135 | 0.1075 |
| planner position error | 0.0847 m | 0.0798 m | 0.0857 m |
| planner velocity error | 0.5246 m/s | 0.5668 m/s | 0.5510 m/s |
| planner velocity direction error | 13.66 deg | 13.80 deg | 13.94 deg |
| planner normal error | 12.04 deg | 12.41 deg | 12.18 deg |
| outgoing speed ratio | 0.8270 | 0.8122 | 0.8200 |
| outgoing direction error | 32.96 deg | 31.94 deg | 32.68 deg |
| post-contact recovery | 0.9939 | 0.9953 | 0.9958 |

The control branch proves that conservative H64 fine-tuning can retain the
incoming policy, but it does not improve return quality. The contact-alignment
term changes training-window diagnostics yet fails the deterministic evaluation.
It is too late and too sparse to repair trajectories whose physical incoming
ball has already diverged from the command used by the actor.

## Root-cause boundary

The incoming model already contacts 59.5% of serves and recovers after 99.4% of
resolved contacts, while only 11.7% cross the net. The dominant failure is the
collision-to-outgoing transition, not contact discovery or standing recovery.

At contact, the fixed-seed trace reports:

- intended versus realized incoming velocity error: `0.406 m/s` and `11.26 deg`;
- actor versus planner racket velocity error: `0.525 m/s` and `13.66 deg`;
- planner command evaluated by the physical impact model: `1.120 m/s` outgoing
  target error even before policy execution error;
- realized outgoing velocity error: `2.014 m/s` and `32.96 deg`.

The rigid one-bounce ball is hidden from the 114D actor. Its command is generated
from the intended route, but PhysX bounce/contact dynamics change the actual
post-bounce velocity. The actor therefore cannot distinguish two physical balls
that share the same command but require different corrections. This is an
unobservable task-contract mismatch. Increasing PPO iterations or the weight of
a contact-only reward cannot remove it.

No V2 or V3 checkpoint is approved for long continuation. The next policy-stage
experiment must first make the command causally consistent with the realized
post-bounce ball while preserving the 114D interface: refresh the existing
position, velocity, normal, and time-to-strike fields from the simulated ball
state, rather than adding another observation dimension or another dense reward.
