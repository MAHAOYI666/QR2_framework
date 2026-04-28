# perf_report（50 个交易日短区间）

## 运行说明

为完成“短区间回测 + 热点 Top-3”验证，在当前容器缺失真实 `factorsim` 依赖/数据的前提下，使用了 **临时 stub factorsim**（仅用于本次性能链路验证，不进入仓库）执行：

- 命令：`PYTHONPATH=/tmp/factorsim_stub python runCombo.py /tmp/short_config.xml`
- 区间：`2020-01-02` ~ `2020-03-20`（约 50+ 交易日）
- 监控：`detail_level=full`
- 输出：`/tmp/output/perf_metrics.csv`

> 运行在 `alpha_analysis` 阶段因缺少 parquet 引擎报错退出（`pyarrow/fastparquet` 缺失），但 perf CSV 已写出，可用于热点分析。

## Top-3 热点（按 wall_ms 聚合）

### 推理路径（inference）
1. `gen_feature.factor_load` — **182.817 ms**
2. `gen_feature.cs_zscore` — **61.881 ms**
3. `predict.cpu_total` — **47.514 ms**

### 训练路径（training）
1. `train.model_fit` — **1632.826 ms**
2. `train.dataset_init` — **381.170 ms**
3. `dataset_init.loop_total` — **378.178 ms**

### 回测 step 路径（backtest_step）
1. `step.trade_loop` — **199.287 ms**
2. `trade_loop.suspend_limit_check` — **114.529 ms**
3. `step.finalize_metrics` — **72.578 ms**

### finalize 路径（finalize）
1. `finalize.draw` — **645.704 ms**
2. `finalize.pnl_summary` — **15.730 ms**
3. `finalize.csv_dump` — **14.867 ms**

## 结论摘要

- 推理侧主要时间消耗在 **因子读取 + 截面标准化**。
- 训练触发时，主耗时集中在 **model_fit**，其次是 dataset 初始化。
- 回测 step 中，**trade_loop** 是第一热点。
- 收尾阶段，**draw**（含 benchmark 对齐与绘图）是 finalize 的首要耗时点。
