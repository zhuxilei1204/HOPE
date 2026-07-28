# Frames

HOPE uses a single right-handed, ROS 2 REP-103 world frame shared by mocap, the
planner, training, evaluation, and deploy. Its provenance is the real
ball-capture fit recorded in
[`configs/ball_physics.yaml`](../../configs/ball_physics.yaml); the geometry
module
[`tasks/table_tennis/geometry.py`](../../hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/table_tennis/geometry.py)
derives every dimension and landmark from that file, so nothing below is
duplicated by hand.

## World / table frame

| Axis | Direction | Range over the table |
|------|-----------|----------------------|
| +x | forward, toward the opponent (P2) | `[0, 2.74]` m |
| +y | left, from the robot's (P1) perspective | table occupies `[-1.525, 0]` m |
| +z | up | `0` **is the table surface** |

- **Origin**: the near-side **left** corner of the table *surface*, from P1's
  perspective.
- The floor is at `z = -0.76` m; the net plane is at `x = 1.37` m.
- Landmarks (net center `(1.37, -0.7625, 0)`, opponent-half center
  `(2.055, -0.7625, 0)`, …) are also published as named static-transform frames —
  see [`hope_ws/src/hope_bringup/config/hope_world_frame.yaml`](../../hope_ws/src/hope_bringup/config/hope_world_frame.yaml).
- Units are SI (metres, seconds) throughout.

## Training / evaluation placement

- The canonical placement is expressed in table frame: the robot stands on the
  P1 side, centered on the table width at base XY
  `(-0.5, -0.7625)` m, facing +x toward P2. The root body is
  `pelvis_link`.
- Robot-centred Isaac whole-body environments and MuJoCo keep the robot's
  startup XY at their local world origin. They therefore use the explicit
  axis-aligned transform `p_sim = p_table + (0.5, 0.7625, 0.76)`. Its single
  source of truth is [`configs/table_frame.yaml`](../../configs/table_frame.yaml).
  The modular table-tennis task may use table frame directly; consumers must
  transform at the boundary instead of changing planner coordinates.
- The racket is mounted on the right wrist (`right_wrist_yaw_Link`); the
  dedicated racket body is `pingpang_red_Link` where the asset keeps it (URDF
  import usually merges fixed joints into the wrist body — the code falls back
  accordingly). See [`A3_ASSETS.md`](../../A3_ASSETS.md).
- All planner quantities (`RacketCommand` position/velocity) and all policy
  racket-target observation terms are expressed in this world frame — see
  [POLICY_INTERFACE.md](../POLICY_INTERFACE.md) and
  [PLANNER_INTERFACE.md](../PLANNER_INTERFACE.md).
- On real hardware Motive establishes this table frame and both the ball and
  `P1_base_link` must be reported in it. The simulator translation above is a
  nominal robot-placement transform, not an offset to add to planner commands
  during deployment.

## Mocap frame

The arena motion-capture stream streams the named rigid bodies — `Ball`,
`P1`, and `P2` — in the same world frame during competition (a `Table` asset exists for
calibration only and is not streamed). Each ROS 2 pose contains
position `(x, y, z)` and quaternion orientation `(qx, qy, qz, qw)`; the current
no-spin planner consumes only the Ball position. The robot's base yaw comes from
the robot IMU, not from mocap. The authoritative mocap frame
and topic contract is [`mocap/README.md`](../../mocap/README.md); the preserved
arena design document
[`mocap/HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md`](../../mocap/HOPE_Motion_Capture_System_and_Coordinates_Reference_Setup.md)
covers rig setup (camera layout, marker placement, vendor frame conversions) and
predates the current stack — where the two differ, the README contract wins.
