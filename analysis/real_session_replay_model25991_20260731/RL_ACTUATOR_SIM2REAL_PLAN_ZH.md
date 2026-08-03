# A3 策略执行器 Sim-to-Real 合同

## 目标与边界

本阶段只处理强化学习策略可实现性，不修改实机控制代码、planner、motion、114D
观测语义或既有击球/恢复奖励。MuJoCo 也不被视为唯一真值；实机日志用于识别策略在
Isaac 中利用、但实机执行器无法复现的行为。

## 已排除与已确认

- 31 维关节顺序、ONNX 打包和推理已验证正确，实机日志重推理最大 action 误差
  `3.80e-6`。
- Isaac、MuJoCo 和 `model_25991` 训练 provenance 已包含用户提供的名义输出端
  armature，遗漏名义电机惯量不是根因。
- 实机 swing/follow 日志出现持续 q_des 机械限位：左踝 roll `89.5%`、右踝 roll
  `58.4%`、waist pitch `56.7%`、waist roll `27.5%`。
- 实机右肩 roll 峰值约 `57.56 Nm`，接近 `60 Nm` 峰值；右腕 pitch/yaw 约
  `6.02/6.63 Nm`，已经到达或超过 `6 Nm` 峰值。
- 当前 Isaac 隐式 PD 将力矩上限和速度上限独立处理，不能表达“峰值力矩和峰值
  转速不可同时达到”；并联腰踝仅能使用静态、对角 armature 近似。

## 已实现的隔离合同

1. `agibot_a3_actuator_contract.py`：31 关节输出端额定/峰值力矩、额定/峰值速度、
   armature、串联/并联分类的唯一合同。
2. 串联关节 armature startup 随机化 `[0.97, 1.03]`，对应供应商约 3% 差异。
3. 并联腰踝 armature startup 随机化 `[0.80, 1.20]`，覆盖姿态相关、非对角耦合
   未建模造成的不确定性；这不是在声称真实误差一定为 20%。
4. phase-aware 可实现性软惩罚：
   - READY/recovery/hold 对持续超过额定力矩或额定速度最敏感；
   - pre-strike 降权；
   - strike 降到 `0.05`，保留短时峰值挥拍；
   - 所有阶段继续惩罚高力矩与高转速同时越过额定区，以及突破峰值包络。
5. 不更改硬 effort limit，不把峰值上限直接压成额定值，避免策略退化成“站稳但
   不挥拍”。

## 可复现 A/B

- Control：`task=HOPEPingPongM25991ReplayControlV1`
- Robust-light：`task=HOPEPingPongM25991ActuatorRobustLightV1`
- 两组均从归档 `model_25991.pt` 加载相同 optimizer、相同 seed、相同单正手 motion。
- Robust-light 相对 Control 的唯一有效差异是三个执行器软惩罚和两组 armature
  随机化事件。

该 A/B 只验证“执行器约束是否在不破坏强正手的情况下减少不可实现行为”，不是最终
任务分布。`model_25991` 是单正手专化模型，不能用它的结果代表反手、平移或全桌。

更准确地说，`model_25991` 的用途是建立一个正手能力 no-regression gate：新增物理
约束后，正手 contact、拍速、normal 和回球不能明显低于它。它不是最终策略初始化的
唯一选择，也不能通过把单正手 motion 等概率扩展成多 motion 就直接得到最终策略。

## 冻结策略诊断结果

已从 `model_25991` 加载 actor，以近似冻结的学习率运行 `64 env x 12 iteration`
（`18432` 个仿真 step），覆盖 pre-strike、impact、brake、settle 和 hold：

- impact planner position error 约 `0.117 m`；
- impact planner velocity error 约 `0.853 m/s`；
- racket/target speed ratio 约 `0.760`；
- velocity direction error 约 `14.6 deg`；
- normal error 约 `4.0 deg`；
- torque clip fraction 为 `0`；
- action clamp fraction 约 `9.7%--13.4%`；
- impact 后腰部 action overflow 出现在约 `75%` 的 phase，`600 ms` 后仍约 `66.7%`；
- impact 后 base angular velocity 超过 `0.8 rad/s` 的比例为 `100%`。

这说明用户提供的电机参数足以建立第一版 actuator contract，但当前主矛盾不是 Isaac
遗漏了名义 armature，也不是静态峰值力矩被大量截断。更直接的问题是策略持续请求腰腿
不可实现的 position target，随后通过 raw last-action 和闭环状态放大，并在恢复段保留
过大的躯干角速度。执行器包络随机化和软惩罚是最终多技能策略的底层鲁棒性约束，不能
替代 operational joint margin、effective-action 语义、完整 cycle outcome 和多技能课程。

## 最终多技能接入

执行器合同本身与 swing side 和 motion 数量无关，单正手 gate 通过后按能力逐步复用：

1. 正手核心区；
2. 正手横向球路和通过质检的正手平移先验；
3. 反手核心区，单独统计反手 contact/normal/velocity，不能被正手均值掩盖；
4. 通过质检的反手平移先验；
5. 正反手切换、全桌宽度和真实 acquire/drop/reacquire lifecycle。

最终验收对象始终是同一个策略，而不是四个独立专化策略。每个中间阶段可用于定位能力，
但最终评估必须至少拆成 `forehand_static`、`forehand_translate`、`backhand_static`、
`backhand_translate` 和 `side_transition` 五个 bucket，并报告最差 bucket，不能只报告
被正手样本主导的全局平均值。

目前通过完整足锁/COM/ground/Isaac-FK 质检的 motion 有 5 段：两段反手不动、一段
正手不动、两段正手短平移。反手平移 motion 尚未通过脚滑 gate，因此它不能以强全身
tracking 的方式直接加入；反手核心能力先用合格静态先验和独立球路学习，反手平移则在
补齐人工足接触标注后接入。这个缺口必须显式保留，不能用单正手结果掩盖。

球路始终独立于 motion。motion 只提供上肢风格、支撑转移、制动和恢复先验；actor
执行带误差的 planner command，奖励仍以物理接触、出球、过网、落台和 durable READY
结算。每一阶段只有在分 side 的击球指标、安全指标和 actuator feasibility 同时通过时
才解锁，不能按 iteration 或全局平均成功率升级。

## 判定指标

先比较 300-500 iteration 的方向，不直接长训：

- 不可退化：contact、net cross、return success、cycle ready、fall。
- 必须改善：waist/leg/right-arm feasibility reward，q_des overflow，踝 roll 与
  waist pitch/roll 限位率。
- 实机相关：planner 对齐后的 racket speed ratio、velocity direction error、normal
  error，以及 READY/恢复段 base angular velocity。
- 只有在击球指标保持且不可实现负载下降时，才进入 1500-3000 iteration 和
  MuJoCo/实机日志回放。

## 尚未纳入首轮的项目

- 实机逐关节精确 PD、电流限幅和 500 Hz 原始链路不是首轮训练阻塞项。
- 不在首轮替换成通用 DC motor 模型，因为它无法准确表达该执行器的短时峰值持续
  时间，直接替换有较大概率压制拍速。
- 不在同一实验里更改 operational joint bounds、last_action 反馈语义、planner 或
  motion；这些必须在执行器 A/B 后分别验证。
