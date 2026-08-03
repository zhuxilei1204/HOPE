# model25991 执行器约束因果 A/B 结果

日期：2026-07-31

## 实验问题

该实验只回答一个局部问题：在不破坏 `model25991` 已有正手能力的前提下，用户提供的
A3 电机/减速器参数能否通过轻量执行器约束减少不可实现动作。它不是最终正反手、平移和
全桌策略训练。

## 冻结协议

- 起点：同一个 `model_25991.pt`，checkpoint iteration `25991`；
- 两组：`4096 env x 500 iter`，seed `0`，完整恢复同一 optimizer；
- Control：`HOPEPingPongM25991ReplayControlV1`；
- Robust：只增加串联/并联 armature 随机化和腰、右臂、腿三组 phase-aware
  torque-speed feasibility 软惩罚；
- 两组使用同一个经过 SHA 核对的单正手 motion；
- 评估同时比较原始 `model25991`、Control `model26490` 和 Robust `model26490`；
- 评估使用相同 Planner V4 快照、seed `17`、one-bounce、正手、动态 station
  `x=0, y=+-0.10 m`；
- Isaac 为 10 球安全/contact gate；MuJoCo 为连续 40 球物理比较。

## 训练末尾 100 轮

| 指标 | Control | Robust |
|---|---:|---:|
| analytic `return_success` | 0.2604 | 0.2556 |
| impact position error | 0.0567 m | 0.0604 m |
| impact velocity error | 0.6881 m/s | 0.7166 m/s |
| impact velocity angle | 11.55 deg | 12.13 deg |
| action clamp fraction | 9.70% | 10.13% |
| waist overflow RMS | 0.793 | 0.932 |
| leg overflow RMS | 0.600 | 0.681 |
| base-too-low termination | 0.125% | 0.625% |
| base-tilted termination | 0.083% | 0.292% |

Robust 的 waist feasibility episode reward 只有约 `-1.1e-4`，右臂和腿项接近零。
静态 torque-speed 软奖励对当前策略的梯度很弱；主要差异来自 armature 随机化和其引发的
闭环适应。训练末段没有观察到 overflow 或 termination 改善。

## 统一物理评估

| 模型 | Isaac contact/success/fall | MuJoCo contact | net/success | fall |
|---|---:|---:|---:|---:|
| 原始 `model25991` | 6/10, 4/10, 0 | 29/40 | 21/40 (52.5%) | 0 |
| Control `model26490` | 2/10, 2/10, 0 | 25/40 | 13/40 (32.5%) | 0 |
| Robust `model26490` | 2/10, 0/10, 0 | 29/40 | 15/40 (37.5%) | 0 |

Robust 比 Control 多保留了 4 次 contact、2 次成功，但没有通过相对原始模型的正手
no-regression gate。Isaac 只有 10 个样本，且碰撞结果和 MuJoCo 不一致，因此不能用
Robust 的 Isaac `0/10` 单独否定策略；三组均通过 no-fall gate。

## Planner 执行

以下均为 MuJoCo 实际 contact 时相对同一个 Planner command 的均值：

| 模型 | position error | velocity error | velocity angle | speed ratio | normal error |
|---|---:|---:|---:|---:|---:|
| 原始 | 0.105 m | 1.226 m/s | 23.39 deg | 0.692 | 10.92 deg |
| Control | 0.130 m | 1.308 m/s | 18.86 deg | 0.575 | 11.12 deg |
| Robust | 0.107 m | 1.425 m/s | 28.43 deg | 0.647 | 13.75 deg |

Robust 基本保留了击球位置和 contact，但速度方向、速度误差和 normal 都比原始模型差。
这解释了 `29/40` contact 最终只有 `15/40` 过网成功：主要退化发生在接触质量，而不是
是否碰到球。

## 动作可实现性

| 模型 | waist clamp | waist overflow p95 | leg clamp | leg overflow p95 |
|---|---:|---:|---:|---:|
| 原始 | 22.77% | 1.001 | 13.35% | 2.602 |
| Control | 27.24% | 2.066 | 15.42% | 1.733 |
| Robust | 26.28% | 1.271 | 15.47% | 2.232 |

Robust 相对 Control 降低了 waist overflow 和右臂绝对 5--20 Hz 能量，但相对原始模型
仍增加了 waist/leg clamp，且腿 overflow 高于 Control。约束没有消除不可实现动作，只是
改变了腰腿之间的补偿分配。

## 结论与决策

1. 当前 `ActuatorRobustLightV1` **未通过**，不继续训练，也不按原权重接入最终多技能任务。
2. 用户提供的 armature、额定/峰值力矩和速度仍有价值：保留为统一物理合同、domain
   randomization 和诊断 gate，但静态 torque-speed reward 不是现有问题的主控制量。
3. 当前主问题是 position target/action overflow、机械限位依赖、raw last-action 闭环和
   恢复段躯干动态；下一版必须直接约束 operational q-des margin 和可执行 action。
4. Control 在 analytic 训练指标上继续提高，但 MuJoCo 从原始 `52.5%` 降到 `32.5%`。
   这再次证明旧单正手 analytic objective 与真实 Planner/碰撞结果不一致，不能继续通过
   该任务长训来寻找最终模型。
5. `model25991` 已完成其用途：提供正手能力 no-regression gate。最终模型不从该单正手
   分布直接外推，而应在统一物理闭环中分别解锁正手、反手、正手平移、反手平移和切换。

## 工具修复

- `export_onnx.py` 改为 actor-only、device-mapped checkpoint load；
- `isaac_physical_eval.py` 同样改为 portable actor-only load；
- A/B evaluator 增加结果文件和 attempts 硬检查；
- 并行 Isaac 评估使用 `CUDA_VISIBLE_DEVICES` 隔离，避免两个进程各自建立两张卡的
  PhysX context。

## 产物

- 训练对齐：`aligned_final100.md`、`aligned_final200.md`；
- checkpoint SHA：`checkpoint_sha256.txt`；
- 三模型评估：`eval/{original,control,robust}/`；
- 每个评估目录包含 ONNX、Isaac/MuJoCo JSON、Planner alignment 和逐关节 action
  feasibility 报告。
