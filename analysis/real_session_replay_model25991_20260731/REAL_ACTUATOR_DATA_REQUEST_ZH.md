# A3 实机执行器合同补充清单

## 1. 当前数据包已经包含

- ONNX 50 Hz 推理与 500 Hz 关节命令发布。
- 发布侧一阶滤波：`y[k] = 0.5*y[k-1] + 0.5*x[k]`。
- 31 DoF 的 `q`、`dq`、估计 `tau`、raw/applied action 和 `q_des`，但
  `policy_trace.csv` 只在 50 Hz policy tick 留一行，不是完整 500 Hz 驱动记录。
- MDU 希望使用的逐关节 stiffness/damping（配置中的 `simulation.pd_gains` 与同源
  参考实现的 policy gains 一致）。
- ActionAdapter 的 default position、action scale 和机械位置 clamp。

用户后续提供的 A3 规格截图还补充了：执行器型号与整机关节映射、电机高速端理论
惯量、减速比、额定/峰值扭矩及输出速度，以及腰踝并联机构的标量折算。逐项审计见
`A3_ACTUATOR_SPEC_AUDIT_ZH.md`。这些 nominal armature 已经存在于当前 Isaac 和
MuJoCo 配置中。

当前数据不能证明：EtherCAT/电机驱动最终实际采用了哪些增益、限制和内部滤波。
`tau` 在数据说明中是估计关节力矩；没有电流字段，也没有力矩/电流饱和标志。

## 2. 需要部署侧直接提供的文件

按以下目录打包，原始 YAML/JSON/C++/日志均可，不要求重新整理格式：

```text
A3_actuator_contract_<date>/
  README.md
  config/
    policy_joint_pd.*
    low_level_servo_pd.*
    motor_current_torque_limits.*
    joint_position_velocity_limits.*
    ethercat_or_motor_config.*
  source/
    joint_command_message_definition.*
    policy_command_publisher.*
    low_level_command_subscriber.*
    command_filter_interpolation.*
  logs/
    steady_ready_10s.csv
    step_<joint_name>.csv
    chirp_<joint_name>.csv
    base_imu.csv
  checksums.sha256
```

`README.md` 至少说明：

- 机器人和执行器固件版本、减速比；
- 控制模式是 position/impedance/torque 中的哪一种；
- stiffness/damping 是否随每条 `JointCommand` 下发，驱动是否会缩放或覆盖；
- `JointState.effort` 是电流换算、观测器估计还是传感器测量，单位和正方向；
- 电流、力矩限制是 motor-side 还是 joint-side，峰值与持续值分别是多少；
- 位置命令从 MDU 到电机之间全部滤波、插值、限速、死区和饱和顺序；
- 本次日志实际加载的配置路径、Git commit 和运行二进制 SHA-256。

## 3. 建议的 500 Hz 长表格式

每个时间戳、每个关节一行。无法取得的列保留为空，不能用 0 代替未知值。

```csv
monotonic_ns,system_ns,mode,joint_name,q_cmd_pre_filter_rad,q_cmd_published_rad,q_motor_received_rad,q_meas_rad,dq_meas_rad_s,tau_ff_cmd_nm,tau_meas_or_est_nm,current_a,kp_cmd,kd_cmd,position_saturated,velocity_saturated,torque_saturated,current_saturated
```

另存 `base_imu.csv`：

```csv
monotonic_ns,system_ns,pelvis_qw,pelvis_qx,pelvis_qy,pelvis_qz,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z
```

必须使用同一单调时钟，或提供两个日志时钟之间的同步映射。

## 4. 无球辨识实验

所有测试由熟悉 A3 的实机操作员执行，保留急停并使用机械保护。腿、踝、腰测试应
在可靠防跌倒支撑下进行。先做静态记录，再决定是否执行扰动；禁止一次扰动多个关节。

### 4.1 静态 READY

- 无球、关闭 RL policy 输出，使用部署时相同的低层控制模式和增益。
- 保持实际 READY 10 s。
- 以至少 500 Hz 记录上述 command/state/current/torque/IMU 字段。

用途：测量静态偏置、噪声、命令到状态延迟、重力负载和自然抖动。

### 4.2 单关节 step

- 其余关节保持 READY。
- 从当前目标叠加一个小正阶跃，保持约 0.3 s 后回零；负方向重复；每方向 3 次。
- 建议初始幅度仅作为上限参考：腕部 `0.01-0.02 rad`，肩肘 `0.01 rad`，
  腰/踝 `0.003-0.005 rad`。最终幅度必须由实机安全负责人确认。
- 优先顺序：右腕/右肩肘，然后腰 pitch/roll，最后踝 pitch/roll。

用途：识别纯延迟、上升时间、阻尼、稳态误差和饱和。

### 4.3 低幅 chirp

- 只有对应关节 step 正常后才执行。
- 单关节、低幅正弦扫频，先从 `0.2 Hz` 到 `2 Hz`，持续约 10 s。
- 幅度不高于同关节已验证安全的 step 幅度，其他关节保持 READY。

用途：识别带宽、相位延迟和未建模共振。若 0.2-2 Hz 已出现异常，不继续提高频率。

## 5. 第一批最小交付

不需要一次拿齐全部 31 关节。第一批只需：

1. 实际 policy 模式 Kp/Kd 表和低层是否覆盖它的说明；
2. 腰 pitch/roll、左右踝 pitch/roll、右肩 pitch/roll、右肘、三轴右腕的实际电流限制、
   峰值持续时间和 torque-speed 限制逻辑；
3. 10 s 静态 READY 500 Hz 日志；
4. 右腕 pitch、腰 pitch、左/右踝 roll 各一组安全 step；
5. 对应运行二进制和配置的 SHA-256。

这一批足够先判断 MuJoCo 快速发散来自增益、延迟、滤波、力矩饱和还是模型惯量/接触。
