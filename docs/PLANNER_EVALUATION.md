# Planner evaluation and debugging

`hope_planner_eval` answers three separate questions instead of collapsing the
planner into one success number:

1. **Is it accurate?** Intercept `y/z`, arrival-time, incoming-velocity and
   forehand/backhand errors are measured at the configured strike plane.
2. **How early is it accurate?** Every causal forecast is grouped by true
   time-to-strike (`0-50`, `50-100`, `100-200`, `200-350`, `350+ ms`).
3. **Where did it fail?** Each ball receives one primary diagnosis:
   `NO_PREDICTION`, `STATE_ESTIMATION`, `TRAJECTORY_MODEL`, `SIDE_SELECTION`,
   `COMMAND_UNREACHABLE`, `NET_CLEARANCE`, or `OK`.

The replay is causal: at timestamp `t` the planner only receives mocap samples
with timestamps `<= t`. A centred local polynomial fit over the later recorded
trajectory supplies the offline reference crossing. It is a useful debugging
reference, not a claim that mocap is noise-free ground truth.

## Input

Use one canonical CSV per contiguous ball trajectory:

```text
t,x,y,z
0.000,1.82,-0.71,0.43
0.003,1.80,-0.71,0.44
...
```

Units are seconds and metres in the HOPE world frame (`+x` toward the opponent,
`+y` left, `+z` up, table surface at `z=0`). Export a live bag with:

```bash
ros2 bag record /poses
ros2 run hope_planner hope_bag_to_csv --bag BAG_DIR --topic /poses --output shot.csv
```

For a simple raw `t,x,y,z` export in millimetres or with tracking gaps:

```bash
python hope_training/ball_physics_fit/extract_canonical.py raw.csv canonical/
```

Motive multi-row exports (the supplied `拍1/拍2` files) contain hundreds of
rigid-body and marker columns and are **not** canonical ball trajectories. First
identify/export the ball centroid as one labeled asset, transform it into the
HOPE table frame, then write `t,x,y,z`. Do not guess an unlabeled-marker column:
Motive changes unlabeled IDs during occlusion, which would make the evaluation
measure marker association errors rather than planner errors.

## Run

From `HOPE/hope_ws/src/hope_planner` (pure Python, no ROS runtime required):

```bash
python -m hope_planner.evaluation canonical/ \
  --output planner_eval_report \
  --x-hit 0.2 --split-y -0.7625 \
  --solve-period 0.02 --fit-window 31
```

After a ROS build the equivalent command is:

```bash
ros2 run hope_planner hope_planner_eval canonical/ --output planner_eval_report
```

The output directory contains:

- `report.html`: quick human-readable diagnosis;
- `summary.json`: aggregate metrics for parameter sweeps and regression gates;
- `events.csv`: one row per incoming strike-plane crossing;
- `predictions.csv`: every planner revision, including estimator, trajectory,
  timing, side, racket-speed and net-margin errors.

Always pass the same `x_hit`, split, fit window, solve period and physics YAML as
the deployed node. A coordinate-frame or unit mismatch normally appears as zero
detected crossings, a very high outlier count, or implausible estimator errors.

## Reading the diagnosis

| Label | Meaning | First checks |
|---|---|---|
| `NO_PREDICTION` | measured ball crossed the plane but no usable command existed | incoming sign, warm-up samples, horizon, dropped frames |
| `STATE_ESTIMATION` | offline position/velocity reference already disagrees before prediction | frame, timestamps, outliers, `fit_window`, bounce reset |
| `TRAJECTORY_MODEL` | state is reasonable but plane crossing is wrong | drag/restitution fit, spin/Magnus omission, bounce detection |
| `SIDE_SELECTION` | predicted and measured crossing lie on different sides of the split | split position, hysteresis, near-boundary uncertainty |
| `COMMAND_UNREACHABLE` | target is accurate but requested racket speed exceeds the debug limit | flight time, landing target, policy envelope |
| `NET_CLEARANCE` | outgoing model does not meet configured clearance | flight-time ceiling, target, contact model |

For full closed-loop attribution, log synchronized racket pose/velocity and the
outgoing ball. Then planner command error, controller tracking error, contact
model error and landing error can be measured independently. Ball-only mocap
cannot distinguish those downstream causes.

## Design references

The metric split follows the planner/controller hierarchy used by HITTER and
the established robot table-tennis separation of interception state from ball
placement. In particular, the report keeps prediction-at-interception,
time-to-interception, executable racket state, net clearance and landing outcome
as distinct quantities. See:

- HITTER, *A HumanoId Table TEnnis Robot via Hierarchical Planning and Learning*
  (2025), https://arxiv.org/abs/2508.21043
- Achterhold et al., *Black-Box vs. Gray-Box: Learning Table Tennis Ball
  Trajectory Prediction with Spin and Impacts* (L4DC 2023),
  https://openreview.net/forum?id=OHv-vlgXQOv
- Tobuschat et al., *Data-Efficient Online Learning of Ball Placement in Robot
  Table Tennis* (2023), https://arxiv.org/abs/2308.14562
- AIMY open-source launcher/evaluation assets, https://webdav.tuebingen.mpg.de/aimy/
