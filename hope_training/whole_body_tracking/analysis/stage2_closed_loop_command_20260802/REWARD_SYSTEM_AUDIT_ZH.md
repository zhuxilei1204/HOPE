# Stage 2 奖励系统总体审计

日期：2026-08-02

## 1. 审计对象

本报告检查实际运行的解析后配置，而不是 YAML 文件名或注释：

- V7B：`stage2_v7b_precision_outcome_from_v6a200_256x300_s8862`
- V8A/V8B：soft health 与 deferred safety credit
- V9：terminal failure accounting，停止于 `model_200.pt`
- V10：hard terminal debit，停止于 `model_50.pt`

V9/V10 已保留 checkpoint 和 TensorBoard 日志。由于本审计确认了系统性奖励预算问题，
不再让它们按原配置跑满 3000。

可重复审计命令：

```bash
python scripts/audit_stage2_reward_budget.py /absolute/path/to/run --window 30
```

## 2. 总结结论

当前 Stage 2 的主要问题不是某一个 weight 太大或太小，而是六个结构问题叠加：

1. **真实物理结果几乎不参与优化。** V9 最近 30 update 中，contact、planner-aligned
   contact 和 recovery settlement 合计只占正奖励 `0.381%`。
2. **持续奖励按相位时长重复支付。** motion、command shaping 和 stability 每步累积；
   contact/net/bounce/settlement 只在单帧支付，并且都被 Isaac Lab 再乘 `dt=0.02`。
3. **Stage 1 shaping 被整体继承到 Stage 2。** 35 个有效项同时参与，多个项重复评价同一
   command 或同一稳定状态，偏离了先前冻结的白名单方案。
4. **物理状态在 reward 之后更新。** `physical_shadow` 属于 command manager，而 Isaac Lab
   的顺序是 `termination -> reward -> reset -> command update`。contact、速度、法向和 timing
   信号对 PPO 至少延迟一个控制帧；触球同帧硬 reset 时，事件还可能在结算前丢失。
5. **奖励相位与实际 lifecycle 不一致。** 当前显式 recovery phase 从未进入，但旧
   `recovery_health` 仍在约 57% 帧上支付，主要覆盖 command-acquire/hold，而不是真正的
   post-impact durable recovery。
6. **延迟闭环超出有效信用窗口。** rollout 只有 64 step（1.28 s），而 contact 到飞行结算再到
   durable READY 常超过 1.28 s；`gamma * lambda = 0.9405` 使 1 s 后的 GAE 信用仅剩约 4.7%。

因此，继续只调整 terminal debit、health floor 或单个 contact weight，不能形成可靠的
Stage 2。它只会改变旧 dense objective 内部的折中点。

## 3. 实际奖励预算

V9 update `180..209`，解析配置共 35 个有效奖励项：

| 类别 | 实际 reward/s | 正奖励占比 |
|---|---:|---:|
| motion prior | 0.30108 | 27.475% |
| command dense | 0.35514 | 32.408% |
| stability dense | 0.43545 | 39.736% |
| physical sparse | 0.00418 | **0.381%** |
| constraint cost | -0.07943 | 不计入正奖励 |

原冻结设计的目标预算是：motion 15%、command 30%、physical outcome 25%、impact health
10%、recovery 15%、station/footwork 5%。实际分布与目标相反：motion 和持续稳定收益共占
约 67%，真实碰撞结果不足 0.5%。

如果只修正 event 单位，使三个 physical pulse 在函数内部除以 `dt=0.02`，同时保持当前策略
分布不变，则按 V9 实测预算估算，physical 占比会从 `0.381%` 上升到约 `16.1%`。这说明当前
差距很大一部分不是“需要再加几十倍经验权重”，而是 rate 与 impulse 的单位合同错误。该估算
不能代替训练验证，因为策略分布会随奖励变化，但可用于确定数量级。

贡献最大的单项：

| 奖励项 | reward/s | 正奖励占比 | 主要问题 |
|---|---:|---:|---|
| imitation | 0.30108 | 27.475% | 独立球路下仍强约束 torso/shoulder/elbow |
| recovery_health | 0.18097 | 16.514% | 无需 targeted attempt，恢复/hold 持续支付 |
| prestrike_racket_progress | 0.13655 | 12.460% | 长 pre-strike 持续支付 |
| healthy_trunk_support | 0.13038 | 11.897% | 与 strike balance/health gate 重叠 |
| planner task-space | 0.10277 | 9.378% | position/velocity/normal 联合核重复积分 |
| lower_body_support | 0.06325 | 5.772% | 与 trunk/strike/recovery 稳定项重叠 |
| physical recovery settlement | 0.00229 | 0.209% | 单次事件，被 `dt` 缩小 |
| physical outcome | 0.00113 | 0.103% | 单次事件，被 `dt` 缩小 |
| physical planner alignment | 0.00076 | 0.069% | 五分量乘性且只在 contact 一帧出现 |

## 4. 逐层问题

### 4.1 时间尺度错误

Isaac Lab 的 `RewardManager.compute()` 对所有项统一执行：

```text
reward = raw_value * weight * dt
```

这适合每秒持续的 state cost，但单次事件也被乘 `0.02`。例如：

- `termination_penalty=-12` 的单次直接值只有 `-0.24`；
- `physical_contact_planner_alignment=6` 的满分 contact 只有 `+0.12`；
- `physical_recovery_settlement=2` 的 contact-tier 满分只有 `+0.04`。

与此同时 imitation、recovery 和 task-space 在几十到几百帧上累计。配置 weight 不能直接
比较，必须区分 rate reward 与 impulse reward。

### 4.2 重复 command 信用

同一击球 command 同时被以下项评价：

- `racket_position`
- `racket_velocity`
- `planner_racket_task_space_crossfade`
- `blade_direction`
- `prestrike_racket_progress`
- `near_impact_planner_velocity_progress`
- `physical_contact_planner_alignment`
- `physical_outcome_events` 中的 outgoing velocity/direction quality

其中前六项不要求真实接触。策略可通过跟踪一个宽松的运动路径获得绝大多数 command
收益，即使最终 miss 或出球失败。

此外，`racket_position` 使用 hidden `ball_strike_pos_w`，`racket_velocity` 使用 hidden
`racket_impact_target_vel_w`；其他项使用 actor-visible planner command。clean command 时二者
重复，加入 planner perturbation 后二者会变成 actor 无法同时精确满足的两个目标。

### 4.3 持续稳定收益可被无击球策略获取

`recovery_health(require_targeted_attempt=false)` 在所有 recovery/hold 帧支付，即使没有有效
挥拍；`no_command_ready_stability` 和 `post_strike_base_ang_vel` 又对相近状态重复支付。这样会
出现稳定但少挥拍的局部最优。

当前 recovery settlement 只在真实 contact 后建立 tier。targeted miss 不进入同一个恢复
合同，和“miss 后也必须收住身体”的真实任务不一致。

### 4.4 Motion 与 task command 的权限没有清晰分离

当前只有两个静态核心 motion。`imitation` 不约束 wrist，但在 strike 仍以 `0.45` 相位比例
跟踪 torso、左右 shoulder/elbow。独立随机球路要求 shoulder/elbow 调整速度和拍面时，这个
先验会持续拉回原动作。

当前没有生效的 lower-body motion prior。下肢仅由 station/support reward 自行探索，因此
更容易得到滑动、重心挪动和上身代偿，而不是成本更高的侧步、制动和恢复。

### 4.5 稳定奖励相互重叠，但没有 durable path gate

`healthy_trunk_support`、`strike_balance`、`lower_body_support`、`recovery_health`、
`post_strike_base_ang_vel` 都评价部分相同的 upright、角速度、后退和支撑信息。

物理 settlement 只检查 route resolved 后 `functional_ready_score >= 0.62` 连续 5 帧，未锁存
impact 后整个路径上的 peak tilt、peak angular velocity、COM/overflow violation。模型可以先
产生危险后倾或高角速度，随后赶在 deadline 前恢复并获得成功结算；这对真机不是等价安全。

### 4.6 一个 ability 标量同时改变过多维度

当前 `_ability_curriculum_level` 同时控制：

- rigid-ball route 范围；
- table workspace；
- ball speed；
- task-space reward std；
- health floor；
- 后续 planner perturbation。

V7B 末期 level 约 `0.35`，V9 回退到 `0.10`，V10 早期接近 `0.0`。回退会同时让球路更容易、
command 评分更宽松并改变 health gate，因此无法判断指标变化来自策略提升还是任务变简单。

V7B/V9/V10 的 aligned-contact curriculum threshold 又是 0，课程并不要求 actor 真正按
position/velocity/normal/timing 联合执行后才升级。

### 4.7 Command/event 更新顺序不满足精确 impact credit

Stage 2 使用 `TableTennisEnv`，但该类只注册 physics callback，没有覆盖 `step()`。默认顺序为：

```text
physics -> termination -> reward -> reset -> command_manager.compute
```

`RacketTargetCommand` 的 racket state 和 `PhysicalBallShadowCommand` 的 contact/net/bounce event
都在最后一步更新。因此：

- action 导致的 impact 奖励被放到下一 transition；
- `|TTS|`、拍速、法向与接触帧存在 20 ms 对齐误差；
- contact 与 table-touch/fall 同帧时，reset 可能先清掉物理状态；
- V10 的 terminal debit 只能处理已经在上一帧建立 tier 的 contact。

必须先建立 pre-reward physical snapshot/latch，之后才值得校准 exact-impact reward。

### 4.8 Action 可实现性仍未解决

V7B/V9/V10 日志均显示：

- q-des acceleration violation fraction 约 `53.7%`；
- leg q-des acceleration violation fraction 约 `68%`；
- action clamp fraction 约 `1.1%--1.6%`；
- constraint 中 actuator 三项合计贡献接近 0。

这解释了仿真和实机上的高频补偿/抖动。当前 slew/actuator cost 有记录但优化影响不足；同时
不能简单全局加大，否则会压制 strike。必须继续按 joint group 和 phase 处理，并以 effective
q_des/实际执行链路为准。

### 4.9 114D normal 的独立信息不足

当前 `planner_command_mode=v4_wire_compatible` 会执行：

```text
racket_target_normal = normalize(racket_target_vel)
```

因此新增的 3D normal 对 actor 基本是 velocity direction 的重复编码，不是独立拍面命令。
这解释了 114D 相比 111D 不一定明显提升。normal reward 本身可以训练实际拍面，但输入没有
提供新的独立目标。要么部署协议提供 normal，要么训练和部署共同采用经过验证的稳定重建规则。

### 4.10 全局 actor parameter anchor 抑制新能力

V9/V10 使用 `actor_anchor_coefficient=2.0`。V9 约 200 update 后 actor 相对参数变化只有约
`0.25%`。这会同时保护父模型的稳定能力和旧的错误挥拍/后倾方式。原冻结设计明确要求不使用
全局 parameter anchor，而采用健康样本上的小权重 behavior KL 并随能力衰减。

### 4.11 当前的 recovery 指标并不代表持续恢复

V9 最近 30 update 的关键日志为：

| 指标 | 均值 |
|---|---:|
| `recovery_functional_ready_score` | 0.243 |
| `physical_ability_recovery_ema` | 0.590 |
| `post_contact_ready_pending` | 0.000 |
| `post_contact_ready_durable_pending` | 0.000 |

二者差异来自 curriculum 实现：它记录 route resolved 后的
`best_post_route_recovery = max(score)`，在下一次 target resample 时把这个单帧最大值记为
recovery；它不要求连续 READY，也不锁存中间危险状态。与此同时 Stage 2 明确设置
`post_contact_ready_enabled=false`，所以已有的 targeted-attempt/durable recovery 状态机没有
参与本次训练。

因此 `physical_ability_recovery_ema≈0.59` 不能解释为“59% 的击球完成了稳定恢复”，只能解释为
“完成 route 的样本中曾出现过相对较好的单帧状态”。

### 4.12 相位定义发生了语义漂移

V9 最近 30 update 的显式 lifecycle 占比：

| phase | 占比 |
|---|---:|
| READY/no-command | 14.45% |
| command acquire | 67.69% |
| pre-strike | 16.55% |
| strike | 1.10% |
| follow-through | 0.21% |
| recovery | 0.00% |
| next-ready | 0.00% |

但 legacy `recovery_phase_gate` 的占比是 56.87%。原因是 `recovery_health` 仍用
`(~pre_strike & ~strike_window) | in_hold` 定义“恢复”，把 command acquire、普通 hold 和真正
post-impact recovery 混在一起。策略可以在没有发生 targeted attempt 的长 hold 中反复获取
recovery 分数，而真实击球后的制动过程没有独立 phase 合同。

### 4.13 加权相加的 stability score 可掩盖单项失效

`recovery_health`、`strike_balance`、`lower_body_support` 均使用加权和。例如
`recovery_health` 中 height/upright/lin/ang/station/feet 相加，脚接触、角速度或站位中的一个
严重失效，仍可被其余正常项抵消。它适合作为诊断或平滑 shaping，不适合作为“可继续下一球”
的成功判据。

已有 `recovery_functional_ready_score` 使用几何组合，这是更合适的 gate 基础，但下一版应把
硬阈值、连续帧和 path latch 用于 one-shot settlement；不能继续用多个 additive positive
reward 近似安全合同。

### 4.14 稳定目标使用旧球 dynamic station，而不是下一球可用状态

解析配置中 `recovery_health(use_dynamic_station=true)`。当前 swing 结束后，dynamic station
仍对应这一颗球的击球位置；下一条 command resample 后 station 才跳到新目标。于是策略被奖励
先回到旧球 station，再立即追下一球，而不是回到固定中性 READY，或在已知下一球时直接进入
下一球 station。

这不是固定 station 与 dynamic station 谁绝对更好的问题，而是 lifecycle 所属关系错误：

- 下一球未知：恢复目标应是固定 deploy READY 区域；
- 下一球已知：恢复目标可以交给下一球 station；
- 不应在恢复结算时继续使用上一球 station，再在 resample 时瞬间改目标。

### 4.15 PPO 时域与延迟奖励不匹配

当前 PPO：50 Hz、`num_steps_per_env=64`、`gamma=0.99`、`lambda=0.95`。因此 rollout 只覆盖
1.28 s，GAE 的跨步衰减是 `(gamma * lambda)^N`：

| 延迟 | 近似保留信用 |
|---|---:|
| 0.30 s / 15 step | 39.9% |
| 0.60 s / 30 step | 15.9% |
| 1.00 s / 50 step | 4.7% |
| 1.60 s / 80 step | 0.7% |

contact 到 net/bounce 已有延迟，再等待 durable READY 会进一步超出 rollout。只把大额奖励全部
推迟到 recovery 末端，会使 impact action 很难获得信用；只保留即时 dense reward，又回到当前
可钻漏洞。下一版需要同时满足：小额即时 contact/alignment 信号、主要价值 escrow 到安全结算、
以及更长的 rollout/适当的 gamma-lambda。三者缺一不可。

### 4.16 Dense kernel 过宽且存在常数基线

解析配置中的 legacy `racket_position(std=0.30 m)`、`racket_velocity(std=2.0 m/s)` 较宽；即使
误差等于一个 std，仍得到 `exp(-1)≈0.37`。再叠加 task-space、blade 和 signed-progress 后，
远离真正 contact 的轨迹也有稳定正收益。下一版不应简单收紧所有 std，因为那会让从零探索
失去梯度；应使用 bounded potential progress 负责接近，用短 impact window component tracking
负责精度，用真实 contact 负责最终结算。

## 5. V8/V9/V10 实验能说明什么

- V8B 把失败 credit 加大后，contact 明显下降，但 safety 没有实质改善，说明它主要抑制探索。
- V9 修复了 hard termination 前 pending settlement 被 reset 清除的 accounting bug；修复有效，
  但 reward budget 没变，因此只有很小 safety 改善。
- V10 把 contact 后 hard reset 的 debit 提高，早期 episode length/health 有改善，但 ability
  同时退到最简单分布，且 physical sparse 仍不足 0.5%。不能据此选择它长训。

这些结果支持“信用结构有问题”，不支持继续增加 failure cost。

## 6. 下一版 Stage 2 的硬合同

在写新权重前先固定以下结构：

1. **奖励类型分离**：rate reward 与 impulse event 使用不同结算 API；event 的配置值表示
   每次事件的真实 return，不再隐式乘小 `dt`。
2. **phase 固定预算**：PRE_STRIKE、STRIKE、RECOVERY、READY 每个周期总预算固定，不随
   hold/episode 时长增加。
3. **command shaping 去重**：保留 signed progress 和短窗 component tracking；移除 hidden
   target 与 actor-visible command 的重复/冲突项。
4. **physical outcome 成为有效信号**：真实物理结果初始贡献目标 15%--25%，但 contact 的
   即时探索值较小，完整价值在 durable recovery 后结算。
5. **recovery 从 targeted attempt 启动**：hit 和 miss 都必须恢复；只奖励 bounded progress
   与 one-shot durable-ready，不逐帧支付高额 READY。
6. **durable safety latch**：保存 impact 后 peak tilt/ang-vel/COM/overflow/table violation；
   终点姿态恢复不能抹掉中间危险状态。
7. **课程轴解耦**：command precision、route width/speed、recovery strictness、planner noise
   分别维护能力 EMA，不用一个 level 同时改变全部难度。
8. **motion 权限限制**：motion 是姿态/支撑时序先验；strike 的 right arm/wrist 由 command
   和物理结果主导，motion 不得获得比真实结果更高的周期预算。
9. **action feasibility 进入 gate**：严重 q_des overflow、持续 slew/torque violation 不可用
   击球奖励抵消；strike 允许短时可实现峰值，recovery/hold 更严格。
10. **信用时域匹配**：闭环 rollout 至少覆盖 contact 到大多数 settlement 的时长，并根据
    实际延迟选择 gamma/lambda；不能依赖 critic 跨多个 rollout 猜测稀疏结果。
11. **训练前预算门**：任何单个 dense 项 >25%、任何 dense 类别 >45%，或 physical outcome
    <10%，禁止长训。command 类允许由“接近进展”和“短触球窗精度”两个互补项组成，但不允许
    再叠加第三组重复 shaping。

## 7. 实施前验证

1. 物理 snapshot/event 在 reward 前更新，覆盖 contact 同帧 hard reset 单元测试。
2. synthetic lifecycle replay：miss、contact-no-net、net、bounce、恢复成功、恢复超时、hard fall。
3. 每个 impulse event 恰好结算一次，且配置值等于实际 episode return 增量。
4. phase occupancy 从 10 万步回放计算，每个 phase 的总预算不随 hold 长度变化。
5. actor-visible command 与所有 dense command reward 使用同一合同；hidden truth 只用于物理
   outcome、robustness 评价和 critic。
6. synthetic recovery 中单帧 READY spike 不得算成功；必须连续满足阈值并且 path latch 安全。
7. 32 env / 10 update smoke 后运行 reward budget 审计；再做 256 env / 100 update。
8. 只有 contact、alignment、safety、durable recovery 同时健康，才进入 300/1500 update。

## 8. 当前决策

V9/V10 作为失败结算 A/B 证据保留，不继续训练。下一步先实现 pre-reward physical event
合同和 phase-normalized reward primitive，再建立一个隔离的 Stage 2 配置；不在 V10 上继续
叠加第 36 个奖励项。

## 9. V11 实现结果

V11 没有在 V10 上继续叠加权重，而是建立了隔离任务
`HOPE-PingPong-Stage2-RewardV11-AgibotA3-v0`，主要合同如下：

1. `TableTennisEnv.step()` 在 reward/termination 前同步球、球拍、command 和物理事件 snapshot，
   同帧 contact 后立即 hard reset 也不会丢失结算。
2. contact、alignment、outcome、settlement、miss debit 等一次性事件使用 impulse 修正，配置值即
   单次事件的真实 return，不再额外乘 `control_dt=0.02 s`。
3. 只保留 29 个有效 reward term；motion 只约束躯干、肩和肘，strike 权重降为 0.18，手腕与
   下肢不使用 imitation。
4. 移除了 `healthy_trunk_support` 在 no-command READY 中的常量正奖励；miss recovery 不再获得
   独立成功奖金，只能通过有界 progress 降低失败代价。
5. physical settlement 同时要求 durable endpoint 和 post-impact safe-path latch。中途后倾、
   高角速度、COM 越界、桌面碰撞或 overflow 不能被最终一帧站稳掩盖。
6. recovery、route、planner perturbation 和 command precision 课程相互独立。本轮只启用四级
   recovery curriculum，球路和 planner 扰动保持固定，避免归因混淆。
7. PPO rollout 从 64 增至 128 step（2.56 s），`gamma=0.995`、`lambda=0.98`，使 contact 到
   durable settlement 尽量处于同一个优势估计时域。
8. V11 启动器强制加载完整 actor+critic、丢弃旧 optimizer，并先运行 3 次 critic-only warmup。
   actor-only checkpoint 会被拒绝，避免随机 critic 在第一次 PPO 更新时破坏已有动作。

实现入口：

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/agibot_a3/hope_stage2_reward_v11_env_cfg.py`
- `cfg/task/HOPEPingPongStage2RewardV11.yaml`
- `cfg/algo/ppo_stage2_reward_v11.yaml`
- `scripts/launch_stage2_reward_v11_member.sh`
- `scripts/audit_stage2_reward_budget.py`

## 10. 训练前验证

32-env / 10-update smoke：

`logs/rsl_rl/hope_pingpong_stage2_reward_v11/2026-08-02_12-55-10_v11_budgetfix_smoke_32x10_s1401`

smoke 的 reward-budget gate 通过。最终全量测试为 `275 passed, 1 skipped`，同时通过
`py_compile`、launcher `bash -n` 和相关文件的 `git diff --check`。

初始化探针还排除了 B19495：在 V11 lifecycle/action 合同下它约 0.7 s 即重置，并持续出现
action overflow。V7B model 200 保留触球探索且无 overflow，因此受控训练只从以下完整 checkpoint
开始：

`logs/rsl_rl/hope_pingpong_stage2_command_precision114_v7/2026-08-02_06-07-16_stage2_v7b_precision_outcome_from_v6a200_256x300_s8862/model_200.pt`

## 11. 双种子受控训练

两组均使用 256 env、128-step rollout，计划最多 50 update。种子 1502 在第 41 次更新停止：
第 30--40 次更新中 tilt termination 回升 13.0%，recovery 线/角速度恶化 10.4%/8.3%，
impact health 下降 12.7%，而 aligned contact 也没有改善。该分支作为失败对照保留：

`logs/rsl_rl/hope_pingpong_stage2_reward_v11/2026-08-02_12-58-07_v11_safeclosed_gate_256x50_s1502`

种子 1501 完成 50 update，共 1,638,400 环境步。下表比较 actor 开始更新后的早期窗口
（update 3--12）和最终 10 update；termination 值是“每环境每 update 的重置次数”，不是百分比。

| 指标 | 早期 | 最终 | 变化 |
|---|---:|---:|---:|
| base tilted termination | 0.830 | 0.560 | -32.5% |
| raw/contact/aligned contact EMA | 43.82% / 11.23% / 6.74% | 47.92% / 12.25% / 7.80% | +9.4% / +9.1% / +15.7% |
| net/bounce EMA | 0.139% | 0.169% | +21.3%，但绝对值仍很低 |
| recovery base angular velocity | 2.186 rad/s | 1.673 rad/s | -23.5% |
| recovery base linear velocity | 0.940 m/s | 0.620 m/s | -34.1% |
| impact health score | 0.211 | 0.460 | +118.2% |
| safe/unsafe terminal settlement | 77.57% / 20.58% | 78.96% / 19.30% | 安全路径小幅改善 |
| q_des acceleration violation | 54.62% | 51.61% | -5.5%，仍然过高 |
| leg q_des acceleration violation | 70.62% | 66.89% | -5.3%，仍然过高 |
| contact position error | 0.021 m | 0.031 m | +44.4%（退化） |
| contact velocity error | 0.105 m/s | 0.209 m/s | +99.4%（退化） |
| contact velocity direction error | 2.94 deg | 5.98 deg | +103.0%（退化） |
| contact normal error | 2.96 deg | 6.19 deg | +109.0%（退化） |
| landing target error | 0.173 m | 0.328 m | +89.2%（退化） |

最终 reward budget：physical positive 15.99%、motion 23.93%、command dense 44.45%、
stability dense 15.63%；最大单个 dense term 为 imitation 23.93%，预算门通过。

最终 checkpoint：

`logs/rsl_rl/hope_pingpong_stage2_reward_v11/2026-08-02_12-58-07_v11_safeclosed_gate_256x50_s1501/model_249.pt`

## 12. 当前结论

V11 已修复 reward 同帧丢失、event 缩放、READY 奖励泄漏、miss recovery 套利和不安全终态
结算五类结构问题。受控训练证明它能在不牺牲触球探索的同时降低倾倒和恢复速度，因此结构方向
有效；但它还不是长训或部署候选：过网绝对率仍低，planner command 在真实 contact 时的速度、
方向、法向和落点误差均退化，腿部目标加速度违规仍接近 67%。

因此下一步不应继续增加全局 stability 权重，也不应直接长训 V11。应先针对两个剩余合同做独立
诊断：一是只在真实 contact frame 结算的 command component 精度，二是按 phase 和 joint group
拆分的 q_des 可实现性。只有这两项改善且 contact/safety 不退化，才放大球路或进入长训。
