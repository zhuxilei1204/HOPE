# Stage 1 Plane020 合并任务与初始化结论

日期：2026-08-02

## 结论

本阶段采用固定 table-frame `x_hit=+0.20 m` 的解析击球任务，将基础击球与小范围平移合并在同一个策略中，但不在 level 0 同时释放全桌、planner 噪声和真实刚体球碰撞。

长训不从零开始。统一从以下不可变 Stage-1 checkpoint 做受控微调：

```text
logs/rsl_rl/hope_pingpong_stage1_operational114/
2026-08-01_18-31-02_stage1polishP1_fromB1500_250/model_1749.pt
SHA-256: e648c8827073756aee3867bc6f478f8c3b022e045e7563dfbc0720dd437c8811
```

使用完整 actor/critic 权重，不加载旧 optimizer。先进行 25 次 critic-only 适配，再以行为信赖域保护 actor 更新。

## 为什么不从零长训

同一任务、同一 seed、2048 env 的 300-update 对照结果：

| 指标，最后 30 update | scratch | guarded transfer |
|---|---:|---:|
| contact EMA | 0.089 | 0.514 |
| net EMA | 0.003 | 0.478 |
| success EMA | 0.000 | 0.034 |
| recovery EMA | 0.406 | 0.801 |
| forehand contact EMA | 0.079 | 0.735 |
| backhand contact EMA | 0.098 | 0.292 |
| safety EMA | 0.827 | 0.901 |
| planner position error | 0.185 m | 0.099 m |
| planner velocity error | 1.031 m/s | 0.677 m/s |
| mean episode length | 115 | 450 |
| tilted termination | 4.280 | 0.640 |
| persistent action overflow | 0 | 0 |

Scratch 在约 250 update 后才开始发现 contact，说明任务并非不可学习；但它需要重新学习已有的站稳、挥拍和时序能力，而且 300 update 时仍不具备安全基础。继续从零会显著增加样本成本和站稳不击球局部最优风险。

## 任务合同

- Actor observation 保持 114D，包含 `racket_target_normal`。
- 50 Hz 控制，31D canonical action。
- 固定 table-frame 击球面 `x_hit=+0.20 m`。
- 球路目标来自 table workspace，不受 motion box 覆盖。
- motion 只提供正反手、时序和上肢动作先验。
- 下肢没有 motion imitation，允许髋、膝、踝主动平移和补偿。
- 每次挥拍显式生成 dynamic station；level 0 只要求约 8 cm 内的小范围移动。
- episode 为 10 s，`wrap_teleport=false`，击球后的状态带入下一拍。
- 本阶段不启用 PhysX 刚体球；contact/net/opponent-bounce 来自同一命令和碰撞模型的解析事件。
- 全桌 y/z 范围只根据 contact、net、success、recovery、正反手和 safety 能力门控逐级释放。

## v3 奖励层级

### 1. 动作先验

- `imitation=0.90`，仅 torso/肩/肘，不包含腿和持拍腕。
- strike 阶段 imitation scale 为 0.40，planner command 有权偏离 motion。

### 2. 稠密命令反馈

- planner position/velocity/normal 联合 crossfade。
- pre-strike racket progress、near-impact velocity progress。
- racket position、velocity、blade direction。
- station progress 只奖励朝目标移动，不持续奖励站在 station。

这些项只在对应相位生效，避免 READY 长时间收益淹没短暂击球事件。

### 3. 接触与 planner 精度

- 健康 soft contact 提供接触前梯度。
- `contact/net/opponent-bounce` 使用一次性事件结算。
- `exact_impact_planner_task_space_alignment=4.0` 只在 contact 时联合结算位置、速度大小、速度方向和法向，避免四项分别优化成不可组合的动作。
- 反手额外启用健康位置和 soft-contact 项，防止平均回报 PPO 优先保留较容易的正手。

### 4. 健康是击球收益的乘性条件

- `impact_health_reward_power=2.0`，主要任务收益乘以健康分数平方。
- 课程 contact 只统计 healthy impact。
- 低健康姿态不能依靠 contact/net 获得完整回报。
- `upright=-0.40`、`healthy_trunk_support=0.40`、`strike_balance=0.30` 只约束主干，不直接限制右臂和腿部补偿。
- `table_no_touch=-1.0`、硬安全 termination penalty 为 `-60.0`。

### 5. 可部署动作包络

- operational joint margin、joint target slew、分组 actuator feasibility。
- waist/upper/leg 分相位 action-rate；strike 时右臂和腿保持低约束。
- joint limit 和 persistent action overflow 保留硬保护。

## PPO 更新保护

普通 warm start 在 critic 解冻后的第一次 actor 更新出现过两类灾难：

- fresh optimizer、`1e-4`：raw action 立即增至约 19，persistent overflow 超过 200/iter。
- 保留 optimizer、`1e-5`：动作未越界，但 table-touch 增至约 70/iter。

因此新增 rollout 行为信赖域：

- 在每轮当前 rollout observation 上比较更新前后 deterministic action。
- v3 限制 RMS 漂移不超过 0.01、p99 不超过 0.05。
- 超限时沿本次 actor 参数增量二分回缩，并同步缩放 actor Adam momentum。
- critic 更新不回缩，策略也不被永久锚定到父模型。
- 微调学习率为 `1e-5`，2 PPO epoch，clip 0.1，max grad norm 0.5。

## 已排除的方案

- Actor-only warm start：随机 critic 导致能力快速崩溃。
- 无行为信赖域的 full warm start：一次 actor 更新即可越界或碰桌。
- `3e-5` 快速微调：不同 seed 均在 80-120 update 内逐步牺牲 contact/健康换取少量 success。
- 全局 actor parameter anchor：会同时冻结父模型的稳定能力和旧的错误挥拍方式。
- 全局 stiff 或强腿 action-rate：会阻断必要的下肢平移与快速平衡补偿。

## v3 长窗口结果

两个 v3 运行均从同一 `model_1749.pt` 开始。短窗口都能保持或提高
contact，但 70--110 个 actor update 后出现了不同形式的累计能力漂移：

| 分支 | 早期最好状态 | 停止原因 |
|---|---|---|
| seed 9261 | iter 1800: contact 0.647，FH 0.735，BH 0.568，safety 0.958 | 后续 contact/BH 连续下降，tilted termination 升至约 1.04/episode |
| seed 9262 | safety 约 0.954，planner 方向误差约 12.5 deg | 最近 20 update FH 0.619、BH 0.330，形成明显偏科 |

两个分支均按预先定义的门槛停止，未把 1500 update 总轮数当作必须完成的目标。
这排除了“只需把 v3 无条件训久”的判断。局部 action trust region 防住了单次
PPO 灾难更新，但无法约束许多小更新累积出的策略遗忘。

选定以下中间 checkpoint 作为 v4 的共同起点：

```text
logs/rsl_rl/hope_pingpong_stage1_plane020_merged114/
2026-08-02_16-16-32_plane020V3_healthsquare_slow_seed9261_1500x4096/model_1800.pt
SHA-256: 6aaac18cd818613dd3fb66db17947bc2b6a1578a753c2e0d10dadeef87d0d702
```

## v4 长训对照

v4 不改任务几何、奖励函数、114D observation 或动作合同，只隔离两个长期
稳定性变量：

- 两组学习率均从 `1e-5` 降到 `3e-6`；actor 单 update 行为漂移由
  RMS/p99 `0.01/0.05` 收紧为 `0.005/0.025`。
- control 保持正反手 `50/50`。
- BH-replay 仅把采样改为 `40/60`，验证平均回报 PPO 是否因弱侧样本信用不足
  而逐步遗忘反手。

BH-replay 的 1 update x 64 env Isaac smoke 已通过，最终解析日志确认
`commands.motion.clip_sampling_weights=[0.4, 0.6]`，并且无 tilt、table-touch
或 persistent overflow。

运行位置：

```text
tmux zxl:p020V4Ctl
analysis/stage1_plane020_merged_20260802/V4_control_seed9461_1500.log

tmux zxl:p020V4BH
analysis/stage1_plane020_merged_20260802/V4_bhreplay_seed9462_1500.log
```

### v4 实际结果

两条分支都在约 60--75 个 actor update 后按预设门槛停止：

| 最近 20 update | 50/50 control | 40/60 BH-replay |
|---|---:|---:|
| contact EMA | 0.628 | 0.633 |
| net EMA | 0.580 | 0.592 |
| success EMA | 0.058 | 0.085 |
| forehand contact EMA | 0.707 | 0.683 |
| backhand contact EMA | 0.548 | 0.602 |
| recovery EMA | 0.825 | 0.825 |
| safety EMA | 0.927 | 0.919 |
| planner position error | 9.59 cm | 9.56 cm |
| planner velocity error | 0.658 m/s | 0.656 m/s |
| tilted termination | 1.04 | 1.15 |
| persistent action overflow | 0 | 0 |

结论：

1. `40/60` 回放有效消除了 v3 的反手遗忘，且没有先牺牲正手，因此弱侧
   样本信用不足确实是一个独立问题。
2. 把 LR 和单步 actor 漂移减半不能阻止长期 safety 漂移。两组都在 success
   上升时出现 tilt 增长，说明不是 PPO 单步过大，而是当前目标允许以更冒险的
   击球换取即时 outcome。
3. `impact_health_score^2` 仍是软乘数，而且只检查触球瞬间。策略可以在触球帧
   健康、拿到 contact/net/success 后，于后续状态失稳；10 s carried state 和
   terminal penalty 没有形成足够直接的信用分配。
4. 下一项应隔离验证 outcome escrow：保留 contact 的稠密/低额反馈，把 net/
   success 的大额收益延迟到触球后固定安全窗口通过后结算。不能继续仅调大
   upright、action-rate 或 termination 权重，否则会再次压制挥拍或下肢补偿。

本轮保留的 Pareto checkpoint 仍是 v3 `model_1800.pt`；v4 后期 checkpoint
不应作为后续初始化。

每条计划 1500 update、4096 env。前 200 update 是强制健康检查窗口，不因总轮数目标忽略退化。

## 停止与推进门槛

以下任一现象持续 20 update 时停止对应 seed：

- safety EMA < 0.90；
- backhand contact EMA < 0.35 且 forehand contact EMA > 0.55；
- tilted termination > 1.0 且继续上升；
- planner position error > 0.11 m 且 contact 同时下降；
- persistent action overflow 非零并持续。

workspace 只有在正反手 contact、net、success、recovery、targeted attempt、safety 和 station saturation 同时过门槛后才扩展。长训选择依据是多指标 Pareto 改善，不以单独 `return` 或 `success` 最高作为模型选择标准。
