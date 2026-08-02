# Plane020 Escrow 长训诊断

## 共同起点

- checkpoint: `logs/rsl_rl/hope_pingpong_stage1_plane020_merged114/2026-08-02_16-16-32_plane020V3_healthsquare_slow_seed9261_1500x4096/model_1800.pt`
- SHA-256: `6aaac18cd818613dd3fb66db17947bc2b6a1578a753c2e0d10dadeef87d0d702`
- actor observation: 114D
- motion: 正手/反手核心两条，采样比例 40/60
- workspace: 固定 `x_hit=+0.20 m`、能力课程 level 0
- PPO: `lr=3e-6`，actor 单更新信赖域 RMS/p99 `0.005/0.025`

冻结 actor 审计的基础能力约为：contact 0.66、正手 contact 0.75、反手
contact 0.60、recovery 0.81、safety 0.94、planner 位置误差 8.2 cm、速度
误差 0.59 m/s。post-contact terminal safe rate 约 0.70。

## 已验证方案

| 方案 | 结果 | 结论 |
|---|---|---|
| soft escrow | 约 83 update 后 contact 0.43、反手 0.23、tilt 1.38 | 即时 net/bounce 信用仍诱导偏科 |
| strict escrow | 约 150 update 后 contact 0.41、反手 0.24、planner 位置误差 11.5 cm | 延迟全部高阶结果仍不能保持覆盖 |
| balanced escrow | 约 129 update 后 contact 0.40、反手 0.23、safety 0.90 | 增大安全 contact 结算仍会缩小命中分布 |
| capability guard | iter 1884 停止；末段 contact 0.47、反手 0.37、safety 0.90、tilt 1.48 | 全局门控能检测退化，但不能提供恢复方向 |
| guard + targeted miss | iter 1905 停止；末段 contact 0.45、正手 0.53、反手 0.40、safety 0.93、planner 位置误差 10.8 cm | 逐次漏球成本延缓退化，但不足以抵消当前奖励预算 |

guard 的冻结审计先发现门控在 0.15--0.81 间抖动。保护阈值调整后，健康
基线后十轮门控主要位于 0.78--1.00；因此训练失败不是门控初始化锁死。能力
跌破保护线后门控可降到接近 0，说明检测逻辑生效。

## 根因

当前 PPO 优化的是每步奖励总和，不是显式的 contact/safety 约束。退化模型的
典型每步日志量级为：

- imitation: `+0.324`
- planner task-space crossfade: `+0.265`
- pre-strike progress: `+0.181`
- safe terminal outcome: 约 `+0.008`
- targeted contact miss: 约 `-0.0045`

稠密 shaping 比单周期物理事件大一个数量级以上。策略可以保持轨迹相似、只在
较小目标子集上得到高质量 success，同时放弃大范围 contact，仍提高平均回报。
全局 capability gate 是无梯度的批次标量；关闭高阶奖励只能停止额外收益，不能
告诉某条漏球轨迹应该如何修正。单步 actor 信赖域只能限制一次更新，不能阻止
100 次小更新的累计遗忘。

## Checkpoint 处理

- 继续保留 `model_1800.pt` 作为后续共同初始化和能力下界。
- guard-miss 的 `model_1900.pt` 仅用于分析退化，不作为训练或部署候选。
- 本轮所有分支均不满足长训条件，不能依据较高 success 单指标选型。

## 下一验证

下一版应先校准周期奖励预算，并把任务结算改为逐轨迹的完整 resolution：健康
命中、真实过网/落台、随后安全恢复共同决定主要收益；健康机会下的 miss 和危险
结算形成同量级 debit。稠密 position/velocity/imitation 仅作为接近目标的辅助，
并在已有 contact 能力时降低占比。验证仍从冻结审计、50 update、100 update
顺序推进，必须同时保持正手、反手、safety、planner 误差和 terminal safe rate。

## Impulse 修复验证

进一步检查 Isaac Lab `RewardManager` 后确认所有 reward term 都会乘
`step_dt=0.02`。原 Stage1 的单帧 outcome/miss 事件因此被额外缩小 50 倍；
Stage2 PhysX 代码已经通过 `impulse=True` 修正，但 Stage1 escrow 未使用该
合同。

隔离任务 `HOPE-PingPong-Stage1-Plane020-EscrowImpulse-AgibotA3-v0` 已完成：

- 旧任务默认行为保持不变；只有新任务对单帧事件除以 `step_dt`。
- 安全 terminal outcome 采用 contact/net/bounce `0.5/2.0/5.0` 的周期值。
- unsafe settlement、健康机会下 inactivity 和可执行挥拍的 miss 使用同单位 debit。
- 冻结审计中 outcome 从约 0.008 提升到约 0.15，targeted miss 约 -0.042；
  contact、safety 和动作输出与父模型一致，奖励预算未爆炸。

训练结果表明 impulse 修复显著延缓了能力崩溃，并首次在约 50--90 update
同时提高 contact 和 success。但共享策略随后把收益集中到单侧：

| 分支 | 中间健康窗口 | 后续现象 |
|---|---|---|
| 40/60，无 miss | contact 约 0.69，success 约 0.07 | 约 117 update 后 contact 0.61、反手 0.55 |
| 40/60，miss -0.50 | contact 约 0.69，safety 约 0.95 | 约 152 update 后正手 0.58、反手 success 0.27 |
| 50/50，miss -0.50 | 早期 contact 0.67、safety 0.95 | 约 100 update 后正手 contact 0.51、反手 success 0.20 |
| 60/40，miss -0.50 | 正手 contact 约 0.72 | safety 0.90、tilt 1.57，提前停止 |

冻结复测 40/60 的 `model_1900.pt` 后，带 miss 分支实际为 contact 0.669、
正手 0.635、反手 0.687、success 0.110、recovery 0.827。它优于无 miss
分支，但仍不是两侧能力同时保持的长训解。

固定采样比例已被排除为充分解。下一项结构修正应使用逐侧信用：当正手能力
下降时，只关闭反手的高阶 outcome，仍保留正手的恢复梯度；反之亦然。当前
global-min gate 会同时关闭两侧，能检测偏科却无法告诉策略应恢复哪一侧。
