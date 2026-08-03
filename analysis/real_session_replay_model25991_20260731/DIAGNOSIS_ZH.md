# A3 实机记录与 MuJoCo Planner-Free 回放诊断

## 1. 诊断对象

- 数据包：`/mnt/ssd/zxl/A3_sessions_20260731_135740_592579588_20260731_125723_761563263.zip`
- 完整会话：`20260731_125723_761563263`
- 策略：`model25991_dynamic_station`
- ONNX SHA-256：`751f664ddce9bdb017cea84e456e6bdbfa5eee56ce6a29a9a62cc9cde3422a2a`
- 有效策略帧：11,714，约 234.44 s
- 有效击球任务：25 个，全部为正手任务

压缩包中的另一个会话 `20260731_135740_592579588` 是中断会话，只包含
1,050 个策略帧和 2 个任务，且缺少完整的 `racket_tcp_samples.csv`，因此没有把它
纳入触球和 MuJoCo 结果统计。

## 2. 回放数据合同

这次回放没有重新调用 Planner：

1. 从 MDU 日志恢复真正送入 actor 的 114D observation、动态 station、目标位置、
   目标速度、目标法向和 `time_to_strike`。
2. 按 `accepted_initial` 与 phase 状态恢复完整任务生命周期。不能直接用每帧 UDP
   `task_id`，因为没有新包时该字段会回到 0，但 actor 仍持有上一条有效 command。
3. Motive 球路已经是 HOPE table frame：近端左下角为原点，`+x` 指向对手，
   `+y` 指向左侧，`+z` 向上。MuJoCo 只施加固定平移
   `[0.5, 0.7625, 0.76] m`，没有再次做轴转换。
4. 球在记录触球时刻前按 Motive 轨迹运动；随后释放给 MuJoCo 物理碰撞。若触球后
   仍强制球跟随 Motive，就无法评估策略和碰撞结果。
5. 球旋转没有观测，本次合同明确不使用 spin。

三种 MuJoCo 模式：

- `closed-loop`：MuJoCo 本体状态生成 observation，运行相同 ONNX。
- `open-loop-action`：原样发送实机记录的 actor action。
- `open-loop-qdes`：原样发送实机记录并经过 500 Hz 平滑后的 `q_des`。

前两种模式复现部署侧 500 Hz、`alpha=0.5` 的命令滤波。

## 3. 已排除的问题

### 3.1 模型或 114D 打包错误

对全部 11,714 帧重新运行 ONNX：

- raw action 最大绝对误差：`3.80e-6`
- raw action RMSE：`2.43e-7`
- `target_pos - base_pos` observation RMSE：`1.10e-8`
- `last_action` observation RMSE：`2.40e-9`
- 当前动态 station observation RMSE：`3.23e-9`

因此 ONNX、114D 维度顺序、command 打包和 last-action 契约均正确。日志中的
station 是当前动态 station，不是固定启动 station。

### 3.2 actor 完全没有使用 Planner command

在 1,838 个有效 command 帧上做反事实替换，action L2 中位变化为：

| 替换项 | 全身 | 右臂 | 下肢 |
|---|---:|---:|---:|
| 动态 station 改固定 station | 0.325 | 0.180 | 0.173 |
| 目标位置改 READY | 2.491 | 1.911 | 0.986 |
| 目标速度清零 | 1.283 | 1.012 | 0.456 |
| timing 改 READY | 5.134 | 4.198 | 1.424 |
| 法向改默认值 | 0.367 | 0.286 | 0.129 |
| 整段 command 清零 | 4.179 | 3.305 | 1.895 |

actor 明确响应 position、velocity、timing、normal 和 station。normal 的作用弱于
position/timing/velocity，但不是未被使用。

## 4. 实机触球结果

高置信触球采用以下定义：

- 几何距离满足触球候选；
- 球的拟合 `vx` 从来球方向 `< -0.2 m/s` 反转为对手方向 `> 0.2 m/s`；
- 拟合速度变化至少 `1.0 m/s`。

结果：

- 高置信实机触球：9/25，`36%`
- 可观测到过网：3/25，`12%`
- 同时满足触球位置误差不超过 6 cm、触球时 `|time_to_strike| <= 50 ms`：
  4/25，`16%`
- 这 4 个 Planner 对齐触球中有 3 个过网

9 次实机触球的中位结果：

- 球拍实际速度：`1.894 m/s`
- Planner 目标速度：`3.001 m/s`
- 速度向量误差：`1.465 m/s`
- 速度方向误差：`21.82 deg`
- 触球时 `time_to_strike`：`+29.4 ms`
- 球拍位置到目标位置：`0.200 m`

其中 5 次是明显提前或偏离目标面的偶然触球，拉高了位置误差。只看 4 次
Planner 对齐触球时，位置误差中位数为 `0.037 m`。

球碰撞速度变化方向与 Planner target normal 的夹角：

- 全部高置信触球中位数：`12.23 deg`
- Planner 对齐触球中位数：`9.95 deg`

这说明策略在真正到达计划触球窗口时，法向并非当前第一瓶颈；更大的问题是能否
按计划位置、时刻和速度到达。不能仅用 contact 奖励继续强化，因为 9 次触球中有
5 次属于错误位置或错误时刻的偶然触球。

严格的对方台落点成功目前不能可靠统计。多数 Motive 轨迹在触球后很快丢失，
没有覆盖首次落台；“可验证成功为 0”不等价于所有球都落台失败。

## 5. 实机执行异常

### 5.1 关节目标大量贴限位

有效任务期间总目标截断比例为 `8.19%`。挥拍与随挥阶段最严重的关节为：

| 关节 | 截断比例 |
|---|---:|
| 左踝 roll | 89.53% |
| 右踝 roll | 58.43% |
| 腰 pitch | 56.73% |
| 腰 roll | 27.51% |
| 右踝 pitch | 12.24% |
| 左踝 pitch | 4.85% |

实机 active 阶段 `q_des - q` 的全身 L2 中位数为 `1.005 rad`，p95 为
`1.843 rad`。这表明策略主要依赖已经接近饱和的踝和腰来保持平衡及追球，会直接
放大抖动、后倾和 sim-to-real 敏感性。当前问题不是单纯“运动幅度不够”。

### 5.2 command 新鲜度

- Planner age：中位 `7.72 ms`，p95 `23.02 ms`
- UDP packet age：中位 `8.14 ms`，p95 `23.30 ms`
- command header age：中位 `104.00 ms`，p95 `344.36 ms`
- bridge queue age：中位 `80.84 ms`，p95 `323.16 ms`
- 固定执行提前量：`40 ms`

数据包本身到达很快，但 command header 反映的轨迹生成时间明显更旧。MDU 会随
时间修正 `time_to_strike`，所以这不是简单的“网络延迟 344 ms”，但旧 revision
会减少真正可用于动作准备的提前量，应继续审计 HDU bridge 的排队和 revision
替换逻辑。

### 5.3 球拍 TCP 法向还不能直接作为真值

日志中的 FDU racket TCP 标记为
`INFERRED_LOCAL_Y_UNVERIFIED_TCP_ORIGIN`。它与 MuJoCo 球拍法向夹角中位数约
`66.5 deg`，说明当前外参或轴定义未经物理验证。因此：

- TCP 位置可以作为近似位置观测；
- FDU 推出的法向不能用于判定策略拍面是否正确；
- 本报告用触球前后球速度变化方向作为碰撞法向代理。

## 6. MuJoCo Planner-Free 对照

25 个任务的结果：

| 模式 | 触球 | 过网/成功 | fall |
|---|---:|---:|---:|
| closed-loop ONNX | 10/25 (40%) | 2/25 (8%) | 1/25 (4%) |
| 实机 recorded action | 4/25 (16%) | 0/25 | 22/25 (88%) |
| 实机 recorded q_des | 4/25 (16%) | 0/25 | 23/25 (92%) |

实机高置信触球任务与 MuJoCo closed-loop 触球任务只有 5 个重合：
`23, 29, 42, 43, 45`。当前 MuJoCo 对真实触球的 precision 为 5/10，recall 为
5/9，只能作为中等相关的诊断工具，不能作为实机成功率真值。

更重要的是，实机动作原样发送到 MuJoCo 后很快发生状态发散：

- 40 ms 时关节位置 L2 中位误差约 `0.114 rad`
- 40 ms 时关节速度 L2 中位误差约 `4.00 rad/s`
- 40 ms 时 closed-loop action L2 中位差约 `1.49`
- 100 ms 时 closed-loop action L2 中位差约 `2.30`
- recorded action 在 0.8 s 时关节位置 L2 约 `0.575 rad`
- recorded action 在 1.2 s 时 base 位置误差约 `0.148 m`

初始 observation 可以匹配到约 `4e-8`，因此发散并非初始 114D 打包造成，而是
MuJoCo 与实机的 actuator、PD、命令插值、刚体参数和接触参数合同尚未对齐。
open-loop 的大量 fall 是动力学 gap 证据，不是该策略在实机必然 fall 的证据。

## 7. 当前根因排序

1. **计划触球执行不足**：只有 4/25 次触球同时满足计划位置和时刻；实际拍速明显
   低于 Planner 目标，且方向仍有约 22 deg 中位误差。
2. **踝和腰的饱和式平衡策略**：挥拍阶段大量目标贴限位，解释了抖动、后倾、
   坏姿态累积和迁移敏感性。
3. **Planner target 可实现性不足**：7/25 个 target 中位高度低于球拍半径
   `0.081 m`，当拍面接近竖直时存在桌面碰撞或不可实现风险。
4. **MuJoCo 执行合同未识别**：相同实机 action/q_des 无法复现实机轨迹，现阶段
   不能用 MuJoCo 单一成功率判断策略优劣。
5. **command revision 有效提前量不足**：bridge queue/header age 偏大，可能使
   策略获得目标时已经进入激进追球状态。
6. **球拍 TCP/normal 外参未验证**：限制了实机拍面误差的直接闭环诊断。

## 8. 下一步验收顺序

### P0：先校准执行与测量合同

1. 提供并核对实机每个关节的 PD、力矩/电流限制、控制模式、插值/滤波链路和机械
   软硬限位。
2. 做安全的小幅无球 step/chirp，记录 500 Hz 的 `q_des/q/dq/tau`，逐关节识别
   响应，优先处理踝 roll、腰 pitch/roll。
3. 对 FDU 到球拍中心与拍面法向做物理外参标定，明确球拍哪一个局部轴是正法向。
4. 检查 HDU bridge 的 command header 与 revision 队列，统计最终有效 command
   在 strike 前实际剩余的可用时间。

MuJoCo 重新具备评分资格的最低 gate 是：recorded-action 回放不再普遍 fall，
且 0.8 s 关节轨迹误差显著下降。达标前，MuJoCo 只用于相对诊断。

### P1：Planner/action 可实现性 gate

在 Planner 输出进入 actor 前或训练采样时联合检查：

- 球拍与桌面的净空；
- 右臂 IK 可达性和关节余量；
- COM/支撑区和腰踝余量；
- 到达目标位置、法向和速度所需的最小时间；
- command 太低、太晚或需要关节饱和时拒绝、降速或重规划。

### P2：再调整训练

- 接触奖励必须由位置、时刻、速度/法向和身体健康共同 gating，避免奖励偶然早碰。
- 对踝和腰增加接近部署可用范围时的连续 soft margin，不只依赖硬 clip。
- strike 阶段保留右臂与手腕自由度，但要求躯干、COM 和关节余量健康。
- recovery 奖励只在有效触球后结算，并要求回到下一球可执行状态。
- 不应先全局加 stiff、action-rate 或 anchor；它们会同时压低正确拍速。

## 9. 产物

- 离线审计：`offline_audit/POLICY_SESSION_AUDIT.md`
- 完整 JSON：`offline_audit/policy_session_audit.json`
- 每任务统计：`offline_audit/task_summary.csv`
- 实机触球动力学：`offline_audit/real_contact_kinematics.csv`
- 关节诊断：`offline_audit/joint_diagnostics.csv`
- command 反事实：`offline_audit/counterfactual_action_sensitivity.csv`
- 可复用数据集：`offline_audit/recorded_policy_replay.npz`
- MuJoCo 汇总：`mujoco_all_tasks_final/result.json`
- 实机/MuJoCo 联表：`comparison_by_task.csv`
- closed-loop 视频：
  `mujoco_video_tasks23_31_43/closed-loop/closed-loop_recorded_ball_command.mp4`
- recorded-action 视频：
  `mujoco_video_tasks23_31_43/open-loop-action/open-loop-action_recorded_ball_command.mp4`

脚本：

- `hope_training/whole_body_tracking/scripts/analyze_recorded_policy_session.py`
- `hope_training/whole_body_tracking/scripts/mujoco_replay_recorded_session.py`
- `hope_training/whole_body_tracking/tests/test_recorded_policy_session_tools.py`
