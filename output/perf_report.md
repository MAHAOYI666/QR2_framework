# perf_report

## 运行状态

本环境执行短区间回测命令 `python runCombo.py eg-lgbm/config.xml` 失败，错误为缺少运行依赖 `factorsim`，因此未生成可用的 `perf_metrics.csv` 统计结果。

## 错误摘要

- `ModuleNotFoundError: No module named 'factorsim'`

## 下一步（在具备依赖的数据环境）

1. 在 XML 设置：`monitor.enabled=true` 且 `detail_level=full`。
2. 运行 50 个交易日区间。
3. 用 `phase` 聚合 `wall_ms`，输出 Top-3：
   - 推理路径 Top-3
   - 训练触发窗口 Top-3
   - 回测 step Top-3
   - finalize Top-3

