# Profiling 指南（细粒度性能埋点）

## 1) section 层级树

> 命名规则：`snake_case` + 层级前缀。

- 主流程（原有）
  - `setup`
  - `combine`
  - `alpha_convert`
  - `backtest_step`
  - `backtest_finalize`
  - `alpha_analysis`
  - `combo_load_checkpoint`
  - `combo_train`
  - `combo_save_checkpoint`
  - `combo_gen_pos`

- 推理（A）
  - `buffer_load.total`
  - `buffer_load.gen_feature_total`
  - `buffer_load.buffer_append`
  - `predict.feature_window_index`
  - `predict.refill`
  - `predict.gpu_total` / `predict.cpu_total`
  - `gen_combo_pos.predict_new`
  - `gen_combo_pos.predict_old`
  - `gen_combo_pos.smooth`
  - `gen_combo_pos.valid_mask`
  - `gen_combo_pos.set_alpha`

- 数据加载（C）
  - `mmap_feature.load`
  - `mmap_feature.stack`
  - `mmap_label.load`
  - `mmap_mask.load`
  - `gen_feature.factor_load`
  - `gen_feature.cube_load`
  - `gen_feature.cs_zscore`
  - `gen_feature.truncate_nan_to_num`
  - `gen_label.load_returns`
  - `gen_label.aggregate`
  - `gen_label.valid_mask`

- 训练（B）
  - `train.build_plan`
  - `train.save_old_model`
  - `train.load_old_model`
  - `train.dataset_init`
  - `train.before_fit`
  - `train.model_init`
  - `train.model_fit`
  - `train.after_fit`
  - `dataset_init.build_validinsts`
  - `dataset_init.alloc_xyw`
  - `dataset_init.loop_total`
  - `dataset_init.loop.gen_feature`（accumulator）
  - `dataset_init.loop.gen_label`（accumulator）
  - `dataset_init.loop.tensor_assign`（accumulator）

- 回测（D）
  - `step.advance_close`
  - `step.coerce_alpha`
  - `step.generate_positions`
  - `step.diff_index`
  - `step.trade_loop`
  - `step.finalize_metrics`
  - `step.csv_append`
  - `trade_loop.suspend_limit_check`（accumulator）
  - `trade_loop.buy_branch`（accumulator）
  - `trade_loop.sell_branch`（accumulator）
  - `finalize.concat_position`
  - `finalize.concat_holdings`
  - `finalize.pnl_summary`
  - `finalize.draw`
  - `finalize.fetch_benchmark`
  - `finalize.csv_dump`

- 主循环 alpha 转换（E）
  - `alpha_convert.detach`
  - `alpha_convert.cpu_transfer`
  - `alpha_convert.numpy`

## 2) detail_level 三档

在 XML 里：

```xml
<monitor enabled="true" output_path="output/perf_metrics.csv"
         collect_gpu="true" sync_cuda="false"
         detail_level="full" />
```

- `coarse`（默认）
  - 只保留主流程和原有 coarse section。
- `standard`
  - 增加一级子阶段（如 `train.*`, `step.*`, `gen_combo_pos.predict_new` 等）。
- `full`
  - 打开所有细粒度埋点与循环累加埋点。

## 3) CSV 输出列说明

`vendor/perf_monitor.py` 输出列：
- `phase`：阶段名（支持层级）
- `date`：交易日（若适用）
- `wall_ms`：墙钟时间（毫秒）
- `cpu_ms`：CPU process 时间（毫秒）
- `gpu_ms`：GPU Event 时间（毫秒；非 GPU 段为空）
- `count`：聚合次数（accumulator 会 > 1）
- `success`：是否成功
- `error`：异常文本（失败时）

## 4) 如何读结果（建议）

1. 先按 `phase` groupby，取 `sum(wall_ms)`，看总热点。  
2. 对 `step.trade_loop` 看占 `step` 的比例。  
3. 对 `train.dataset_init` vs `train.model_fit` 看“数据准备 vs 训练”占比。  
4. 对 `predict.gpu_total`（若有）与 `buffer_load.*` 比较“GPU前向 vs CPU准备”。  
5. 对 `finalize.*` 看尾部耗时，重点关注 `finalize.fetch_benchmark` 和 `finalize.draw`。

## 5) 测量精度与限制

- `cuda_section` 使用 `torch.cuda.Event`，能测 GPU 核函数时间；但仅对可见的 GPU work 准确。  
- 若模型把 HtoD/Forward/DtoH 都封装在 `ResearchModel.predict` 内，外层只能记录 `predict.gpu_total`，无法稳定拆分三段。  
- `mixed_section` 会在边界 `cuda.synchronize()`，反映端到端“CPU入队+GPU执行+同步”耗时，但会改变时间线（仅建议用于诊断）。  
- `accumulator` 通过循环内本地累加 + 末尾 flush，减少高频 loop 的监控开销。
