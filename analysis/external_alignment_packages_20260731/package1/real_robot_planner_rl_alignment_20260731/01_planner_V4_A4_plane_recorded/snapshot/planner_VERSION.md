# v4_latest_729

2026-07-29 HDU 当前代码和配置的完整快照。增加按时间截取拟合窗口、XY/Z
分阶拟合和 bounce 抗抖，并使用优化后的 drag/revision 参数。

- `x_hit=0.20`
- `fit_window_s=0.14`，`fit_window=67` 上限
- `poly_order_xy=1`，`poly_order_z=2`
- `drag_k=0.28`
- `revision_gate_freeze_tts_s=0.08`
- 保留为可恢复的今天版本；尚需单独完成实机回归验证
