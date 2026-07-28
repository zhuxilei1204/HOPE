# HOPE A3 Ping-Pong Experiment Handoff

本文档面向接手本仓库的人，帮助其快速理解当前 HOPE / Agibot A3 乒乓球实验的代码结构、实验脉络、已知问题和最短上手路径。

适用范围：截至 2026-07-28 的 `zhuxilei1204/HOPE` `main`，重点覆盖 `hope_training/whole_body_tracking`、`hope_ws/src/hope_planner`、`a3_deploy/a3_deploy_example` 和已推送的 motion / checkpoint / deploy artifact。

## 1. 当前仓库状态

当前代码已经包含三条主线：

1. ROS 2 planner：`hope_ws/src/hope_planner`
   - 球状态估计、无旋轨迹预测、球拍目标规划。
   - 已合入 planner evaluation、mocap outlier gate、bounce reset bridge、net clearance、command stability gate 等优化。

2. Isaac Lab 训练：`hope_training/whole_body_tracking`
   - Agibot A3 31-DOF 全身控制策略训练。
   - 训练目标从基础 forehand/backhand imitation 逐步扩展到 one-bounce return、continuous rally、cycle objective、workspace curriculum、impact-health gating。

3. 部署参考：`a3_deploy/a3_deploy_example`
   - ONNX policy runner、action adapter、MuJoCo sim bridge、planner command 接入。
   - 当前 runner 支持 111-D 和 114-D policy input，通过 ONNX 输入维度自动判断。

已推送到 GitHub 的主要二进制/数据：

- motion 数据：`hope_training/motions/`
- 补充视频：`backhand.mp4`、`fronthand.mp4`、`补充motion视频.zip`
- 部署包：`deploy_artifacts/B17996_deploy_ready_no_command_20260726/`
- 部署包压缩件：`deploy_artifacts/B17996_deploy_ready_no_command_20260726.tar.gz`
- 最新少量 checkpoint 快照：
  - `2026-07-28_14-47-49_M22750_R1_reward114_torsocom_common3000_stableppo_20260728/model_25000.pt`
  - `2026-07-28_14-48-42_M22750_F1_feedback122_torsocom_common3000_stableppo_20260728/model_24500.pt`
  - `2026-07-28_15-57-36_SMOKE_M22750_H1_122free_20260728/model_22750.pt`
  - `2026-07-28_15-58-50_M22750_F2_feedback122_newcolsfree_common3000_stableppo_20260728/model_22750.pt`
  - `2026-07-28_15-59-29_M22750_H1_feedback122_newcolsfree_healthimpact3000_stableppo_20260728/model_22750.pt`

未推送或不建议直接推送：

- `analysis/`：训练日志、视频、评估中间结果，体积大且混合临时分析。
- `data/`：raw motion pipeline 数据和原始视频，存在接近或超过 GitHub 单文件限制的文件。
- `hope_training/whole_body_tracking/logs/` 全量历史 checkpoint：约数 GB，不适合普通 Git 管理。
- `MUJOCO_LOG.TXT`：本地仿真日志。

## 2. 快速上手路径

先看根目录：

- `README.md`：项目总览和最短 train / export / eval / run 命令。
- `QUICKSTART_A3_ISAAC.md`：从新机器准备 Isaac / A3 asset 的完整步骤。
- `docs/POLICY_INTERFACE.md`：policy 输入输出合同。
- `docs/PLANNER_INTERFACE.md`：planner ROS topic 和 RacketCommand 合同。
- `docs/interfaces/frames.md`：table / world / robot / mocap 坐标系。
- `deploy_artifacts/B17996_deploy_ready_no_command_20260726/README.md`：当前可复用部署 baseline。

建议接手顺序：

```bash
# 1. 克隆后先确认 planner 纯 Python 测试
cd /path/to/HOPE
PYTHONPATH=hope_ws/src/hope_planner python3 -m pytest hope_ws/src/hope_planner/test

# 2. 准备 A3 Isaac asset
python3 hope_training/whole_body_tracking/scripts/prepare_a3_isaac_asset.py --force

# 3. 进入 Isaac Lab / torch 环境
cd hope_training/whole_body_tracking
source setup_train_env.sh

# 4. 训练一个小规模 smoke run
python scripts/train.py task=HOPEPingPong algo=ppo headless=true num_envs=8 max_iterations=10

# 5. 导出 ONNX
python scripts/export_onnx.py --checkpoint logs/rsl_rl/hope_pingpong/<run>/model_<iter>.pt

# 6. MuJoCo 中评估 ONNX
python scripts/mujoco_eval_onnx.py --onnx logs/rsl_rl/hope_pingpong/<run>/exported/hope_pingpong.onnx
```

如果只是验证 planner：

```bash
cd hope_ws
colcon build
source install/setup.bash
ros2 launch hope_bringup hope_bringup.launch.py use_fake_ball:=true
```

如果只是验证 deploy runner：

```bash
cd a3_deploy/a3_deploy_example/reference
python -m a3_deploy_onnx_ref_pingpong --planner --view --realtime
```

## 3. 推荐先复现的 baseline

### B17996 deploy-ready / no-command baseline

位置：

```text
deploy_artifacts/B17996_deploy_ready_no_command_20260726/
```

推荐 checkpoint：

```text
deploy_artifacts/B17996_deploy_ready_no_command_20260726/checkpoints/model_17996.pt
```

导出的 ONNX：

```text
deploy_artifacts/B17996_deploy_ready_no_command_20260726/models/hope_pingpong.onnx
```

这个包的意义：

- 是一个已经整理过的部署 baseline。
- 包含 checkpoint、ONNX、policy manifest、runtime config、planner config、joint order、训练参数快照和 git diff 快照。
- 不是最新所有实验里最强的模型，而是更适合回滚、对照和部署链路 sanity check 的基线。

该包 README 中记录过一次 MuJoCo gate：

```text
success: 0.325
contact: 0.475
net: 0.325
opponent bounce: 0.350
fall: 0.000
```

## 4. 实验演进脉络

这轮实验不是单次训练，而是围绕“能稳定接 planner、能连续回球、身体不过度失稳”迭代出来的。

### 4.1 Motion 数据阶段

目标：

- 替换 placeholder motion。
- 引入用户录制的 forehand / backhand。
- 通过 A3 FK、yaw stabilization、lower-body stabilization 和 contact-aware retarget，让 motion 更接近可训练、可部署状态。

相关目录：

```text
hope_training/motions/user_recorded_two_motion_20260723_canonical/
hope_training/motions/user_recorded_two_motion_a3fk_yaw_leg_stabilized_20260723/
hope_training/motions/user_four_motion_20260724_canonical/
hope_training/motions/user_four_motion_a3fk_yaw_leg_stabilized_20260724/
hope_training/motions/user_four_motion_manual_hits_a3fk_yaw_leg_stabilized_20260724/
hope_training/motions/supplemental_lower_body_contact_aware_20260728/
hope_training/motions/supplemental_lower_body_contact_aware_isaacfk_20260728/
hope_training/motions/supplemental_lower_body_deploy_ready_residual_0p50_isaacfk_20260728/
```

相关脚本：

```text
hope_training/whole_body_tracking/scripts/contact_aware_retarget.py
hope_training/whole_body_tracking/scripts/regenerate_motion_isaac_fk.py
hope_training/whole_body_tracking/scripts/rebase_lower_body_deploy_ready.py
hope_training/whole_body_tracking/scripts/add_racket_boxes_to_motion_manifest.py
hope_training/whole_body_tracking/scripts/add_motion_racket_offsets_to_manifest.py
```

经验：

- motion 的坐标系和桌子坐标系必须尽早统一，不然后面 planner、训练和 MuJoCo 会出现“都能跑但击球点不对”的问题。
- 不要只看手臂动作，lower body、torso、COM 和脚底接触决定了模型能否部署。
- 只用 raw 视频或 raw GVHMR 输出不够，需要转成 A3 可用的 `.npz` / `.yaml` motion pair，并写入 manifest。

### 4.2 训练任务阶段

训练配置集中在：

```text
hope_training/whole_body_tracking/cfg/task/
```

本轮新增了多组任务配置，名字很长但有规律：

- `M22494...`：围绕 cycle objective、table-width curriculum、contact-aware motion 做实验。
- `M25493...`：围绕 lower-body conflict、action feedback、safety-first delayed hit 做实验。
- `M22750...`：围绕 stability、torso/COM reward、feedback 122 维训练和 impact-health gating 做实验。
- `OneBounceCollisionRecovery...`：围绕 one-bounce、collision recovery、post-strike ready、transition bridge 做实验。
- `ScratchRealPlannerContinuousAbility...`：从真实 planner 连续能力角度做 scratch / motion-start mix / trunk-waist stable 实验。

常用训练入口：

```bash
cd hope_training/whole_body_tracking
source setup_train_env.sh

python scripts/train.py \
  task=HOPEPingPongM22750StabilityFeedback122F1 \
  algo=ppo \
  headless=true \
  num_envs=4096 \
  max_iterations=3000
```

最近新增的 impact-health 配置：

```text
hope_training/whole_body_tracking/cfg/task/HOPEPingPongM22750StabilityFeedback122HealthImpactH1.yaml
```

它的核心目的：

- contact / net / opponent-bounce 不再只看球是否回过去。
- cycle 和 ability EMA 可以要求 impact 时 torso / COM / foot support 处于健康范围。
- reward 中加入 health-gated shaping，避免模型靠后仰、失稳、极端动作换取短期击球。

### 4.3 Planner 阶段

planner 代码位置：

```text
hope_ws/src/hope_planner/hope_planner/
```

本轮重点优化：

- `ball_state_estimator.py`
  - mocap outlier gate。
  - bounce reset bridge。
  - post-bounce 冷启动缓解。

- `ball_trajectory_predictor.py`
  - 无旋球轨迹预测。
  - table / net / first bounce 相关逻辑。

- `racket_target_planner.py`
  - racket target 选择。
  - net clearance 保护。
  - strike timing 搜索。

- `command_stability_gate.py`
  - live command 稳定性过滤。
  - 减少目标命令抖动。

planner 快速测试：

```bash
PYTHONPATH=hope_ws/src/hope_planner python3 -m pytest hope_ws/src/hope_planner/test
```

当前已验证：

```text
61 passed
```

### 4.4 部署与评估阶段

部署参考 runner：

```text
a3_deploy/a3_deploy_example/reference/a3_deploy_onnx_ref_pingpong/
```

核心模块：

- `onnx_policy.py`：加载 ONNX。
- `observation.py`：构造 policy observation。
- `action_adapter.py`：policy raw action 到 A3 joint command。
- `runner.py`：主循环。
- `sim_bridge.py`：MuJoCo bridge。
- `config.py`：运行配置。

当前需要注意 111-D / 114-D observation：

- 公开基础合同是 111-D。
- B17996 deploy artifact 使用 114-D，增加了 `racket_target_normal_w`。
- runner 会读取 ONNX 输入维度来兼容两者。
- 如果 planner normal 不可用，runner 会用 target velocity 归一化方向 fallback。

## 5. 常见问题和排查

### 5.1 GitHub 上为什么没有所有 checkpoint？

`hope_training/whole_body_tracking/.gitignore` 和根 `.gitignore` 默认忽略：

```text
logs/
outputs/
runs/
*.pt
*.pth
*.onnx
```

原因：

- 全量训练日志和 checkpoint 很大。
- 当前本地 `logs/rsl_rl/hope_pingpong` 中 `.pt` 历史约数 GB。
- GitHub 单文件超过 100MB 会拒收；大二进制即使单文件没超，也会显著拖慢 clone/push。

本仓库目前只把必要的 deploy artifact 和少量关键 checkpoint 用 `git add -f` 推上去。后续如果要长期保存训练历史，建议使用 Git LFS、GitHub Release、对象存储或网盘，不建议普通 Git 直接追踪全量 `logs/`。

### 5.2 为什么有些测试在系统 Python 下不能跑？

planner 测试基本是纯 Python / numpy，可直接跑：

```bash
PYTHONPATH=hope_ws/src/hope_planner python3 -m pytest hope_ws/src/hope_planner/test
```

训练侧部分测试需要 `torch` 或 Isaac Lab。当前机器上系统 `python3` 没有 `torch`，可以用带 torch 的 conda / Isaac 环境。例如：

```bash
PYTHONPATH=hope_training/whole_body_tracking/source/whole_body_tracking \
conda run -n zxl-pace python -m pytest hope_training/whole_body_tracking/tests/test_actor_anchor.py
```

当前已验证：

```text
3 passed
```

### 5.3 训练看起来能击球，但部署时身体不稳

优先看这些指标：

- `Metrics/racket_target/recovery_ready_score`
- `Metrics/racket_target/base_backward_velocity`
- `Metrics/racket_target/action_clamp_fraction`
- `Metrics/racket_target/ability_safety_ema`
- `Loss/actor_anchor_rms`

相关脚本：

```text
hope_training/whole_body_tracking/scripts/monitor_training_health.py
hope_training/whole_body_tracking/scripts/summarize_balance_trace.py
hope_training/whole_body_tracking/scripts/summarize_joint_action_diag.py
hope_training/whole_body_tracking/scripts/mujoco_stand_diagnostic.py
hope_training/whole_body_tracking/scripts/no_ball_standing_gate.py
```

常见原因：

- reward 过度鼓励 contact / success，身体健康没有进入 gating。
- lower body 被 motion prior 或 imitation 约束得太死。
- action adapter clamp 太频繁，策略输出超出可部署动作域。
- actor anchor 过强导致新 observation columns 学不动，过弱又会破坏老策略。

### 5.4 planner 能跑，但策略接 planner 后不稳定

检查顺序：

1. `docs/interfaces/frames.md` 中的 world/table frame 是否一致。
2. `/racket/command` 的 target position / velocity / normal 是否在 policy 期望坐标系。
3. `hope_world_frame.yaml` 和 `configs/table_frame.yaml` 是否一致。
4. ONNX manifest 的 observation contract 是 111-D 还是 114-D。
5. runner 是否启用了 planner command source，而不是固定 command 或 no-command baseline。
6. command stability gate 是否过滤掉短时间跳变。

### 5.5 为什么 success_rate 不能完全代表部署能力？

`success_rate` 只回答“球有没有打回去并落到对面半台”。它不直接惩罚：

- 身体是否后仰。
- COM 是否跑出支撑范围。
- 脚底是否失去有效接触。
- action 是否长期被 clamp。
- 击球后是否能恢复 ready posture。

因此最近引入了 impact-health gating 和 recovery / safety 类指标。评估模型时建议同时看 contact、net、opponent bounce、fall、recovery 和 action clamp。

## 6. 推荐工作流

### 修改 planner

```bash
PYTHONPATH=hope_ws/src/hope_planner python3 -m pytest hope_ws/src/hope_planner/test
```

需要重点保持稳定的文件：

```text
hope_ws/src/hope_planner/hope_planner/constants.py
hope_ws/src/hope_planner/hope_planner/ball_state_estimator.py
hope_ws/src/hope_planner/hope_planner/ball_trajectory_predictor.py
hope_ws/src/hope_planner/hope_planner/racket_target_planner.py
hope_ws/src/hope_planner/hope_planner/node.py
```

### 修改 policy observation / action contract

必须同步检查：

```text
docs/POLICY_INTERFACE.md
hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/actor_observation_contract.py
a3_deploy/a3_deploy_example/reference/a3_deploy_onnx_ref_pingpong/observation.py
a3_deploy/a3_deploy_example/reference/a3_deploy_onnx_ref_pingpong/config.py
hope_training/whole_body_tracking/tests/test_policy_contract.py
hope_training/whole_body_tracking/tests/test_action_adapter_parity.py
```

如果 observation 维度变化，必须明确：

- 旧 checkpoint 能否继续加载。
- 第一层 actor weight 是否需要扩维。
- ONNX runner 是否能自动识别维度。
- policy manifest 是否记录了正确 contract。

相关脚本：

```text
hope_training/whole_body_tracking/scripts/expand_actor_observation_checkpoint.py
```

### 修改训练 reward / command

建议至少跑：

```bash
PYTHONPATH=hope_training/whole_body_tracking/source/whole_body_tracking \
conda run -n zxl-pace python -m pytest hope_training/whole_body_tracking/tests/test_actor_anchor.py
```

如果改动不依赖 Isaac，可优先补纯 Python 测试。依赖 Isaac 的变化，要用小规模 smoke training 验证 import、Hydra config 和 rollout。

### 新增 motion

建议流程：

1. 原始视频或 mocap 数据放在本地 `data/`，不要默认提交。
2. 生成 A3 `.npz` / `.yaml` motion pair。
3. 写入 `manifest.tsv`。
4. 用 Isaac FK 或 MuJoCo reference playback 做 sanity check。
5. 只提交最终可训练 motion，不提交大体积 raw 视频。

## 7. 重要文件速查

项目入口：

```text
README.md
QUICKSTART_A3_ISAAC.md
docs/QUICKSTART.md
```

坐标和接口：

```text
docs/interfaces/frames.md
docs/interfaces/joint_order.md
docs/POLICY_INTERFACE.md
docs/PLANNER_INTERFACE.md
docs/RUN_ON_AGIBOT.md
```

planner：

```text
hope_ws/src/hope_planner/hope_planner/
hope_ws/src/hope_planner/test/
hope_ws/src/hope_bringup/config/hope_world_frame.yaml
```

训练：

```text
hope_training/whole_body_tracking/cfg/train.yaml
hope_training/whole_body_tracking/cfg/task/
hope_training/whole_body_tracking/scripts/train.py
hope_training/whole_body_tracking/scripts/evaluate.py
hope_training/whole_body_tracking/scripts/export_onnx.py
hope_training/whole_body_tracking/scripts/mujoco_eval_onnx.py
```

MDP / reward / command：

```text
hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_commands.py
hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_rewards.py
hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_observations.py
hope_training/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/hope_actions.py
```

部署：

```text
a3_deploy/a3_deploy_example/reference/a3_deploy_onnx_ref_pingpong/
a3_deploy/a3_deploy_example/config/hope_pingpong_runtime.yaml
deploy_artifacts/B17996_deploy_ready_no_command_20260726/
```

motion：

```text
hope_training/motions/
hope_training/whole_body_tracking/scripts/contact_aware_retarget.py
hope_training/whole_body_tracking/scripts/regenerate_motion_isaac_fk.py
hope_training/whole_body_tracking/scripts/rebase_lower_body_deploy_ready.py
```

## 8. 接手后的第一天建议

1. 跑 planner 测试，确认基础环境。
2. 阅读 `docs/interfaces/frames.md` 和 `docs/POLICY_INTERFACE.md`，不要先改训练。
3. 用 `deploy_artifacts/B17996.../models/hope_pingpong.onnx` 跑一次 MuJoCo deploy runner。
4. 检查最新 task cfg，挑一个小 `num_envs=8` / `max_iterations=10` 的 smoke train。
5. 确认自己理解 111-D / 114-D observation 差异，再动 observation contract。
6. 如果要继续训练，先用 `monitor_training_health.py` 监控 safety / recovery / actor anchor。
7. 不要把 `analysis/`、`data/`、全量 `logs/` 直接提交到 Git。

## 9. 当前仍需继续解决的问题

1. Policy 稳定性
   - 需要在 success_rate、身体健康、recovery ready、action clamp 之间找到更稳的平衡。
   - impact-health gating 已加入，但需要更长训练和 MuJoCo / Isaac 双侧评估。

2. Motion 质量
   - motion 已有多批可训练版本，但不同 retarget 策略仍需系统比较。
   - lower body 和 torso 的 deploy-ready 质量比手臂轨迹更关键。

3. Planner-to-policy 闭环
   - planner 本身测试通过，但真实 mocap 噪声、table frame 偏差、command jitter 仍需要现场验证。
   - command stability gate 能减小抖动，但参数需要按现场 mocap 质量调。

4. Checkpoint 管理
   - 目前只在 Git 中保留少量关键 ckpt。
   - 后续需要一个正式 artifact 管理方式，避免仓库不断膨胀。

5. 真实机器人部署
   - MuJoCo runner 和 deploy artifact 已整理。
   - 真机还需要结合 Agibot vendor deploy 环境、PD 参数、安全流程和现场 mocap 做最终闭环验证。

