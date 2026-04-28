# QR2_framework 成熟度与流程性能评估（分析版）

> 评估范围：`runCombo.py + vendor/comb2 + vendor/comb2-pcmaster + 示例模型`。
> 
> 本文仅做诊断与定位建议，不包含代码优化改造。

## 1. 架构成熟度评估

### 1.1 分层与可扩展性

**优点（中高成熟）**
- 已形成“研究模型接口 + 统一调度入口 + 回测引擎”三段式结构：
  - 统一入口 `runCombo.py` 负责配置加载、训练/推理、回测串联。
  - `ResearchModel` 通过 `fit/predict/save/load` 接口接入（`eg-lgbm`、`eg-torch` 已示例）。
  - 回测引擎通过 `comb2-pcmaster` 独立封装。
- `selection` 模块化，训练计划与数据窗口逻辑独立于模型本体。
- 支持可选 `PerfMonitor`，具备流程级可观测性基础。

**不足（中等成熟）**
- 训练、推理、回测均在单进程串行主循环中，缺少流水线并发。
- 缺少统一的“阶段 SLA/阈值”与自动告警（例如：某阶段 wall time 环比暴涨）。
- 运行依赖 `factorsim + Memmaper2` 与本地缓存路径，部署移植性中等。

**结论**：架构达到“可研究、可回测”的**工程化初级到中级**水平，模块边界清晰，但距离专业量化平台常见的“并行调度 + 资源编排 + 自动性能回归”仍有明显差距。

---

### 1.2 工程可靠性与可维护性

**优点**
- 配置默认值较完整，路径解析规范（相对路径以 config 所在目录解析）。
- 训练/推理日历对齐逻辑较完备（date2didx / didx2date / align_date）。
- 日志输出覆盖训练检查点、alpha 摘要、回测指标。

**风险点**
- 性能监控默认 `sync_cuda=false`，GPU 耗时容易低估（异步执行导致）。
- `draw/_pnl_summary` 依赖外部基准数据查询，可能引入非本地网络延迟与不确定性。
- `alpha_history` 全量保存在内存字典后再落盘，长区间回测存在内存增长风险。

**结论**：可维护性中等偏上，但可观测性与稳定性（尤其 GPU 时间真实性、外部依赖抖动）还有改进空间。

---

## 2. 流程级性能评估（不改代码）

## 2.1 当前主流程

日级循环核心路径：
1. `combo.Combine(date)`：可能触发 `GenComboPos` 与 `Train`。
2. `alpha` 从 torch tensor 转 numpy。
3. `backtest.step` 执行逐股票交易撮合。
4. 全部日期完成后：`finalize + alpha_analysis`。

该路径中最可能的重阶段是：
- `combo_train`（训练数据构建 + 模型拟合）
- `combo_gen_pos`（逐日特征加载+归一化+预测）
- `backtest_step`（逐标的 Python 循环撮合）
- `alpha_analysis`（全历史因子与 label 相关性计算）

---

## 2.2 潜在时间瓶颈（按优先级）

### 瓶颈 A：训练数据集构建的 I/O 与重复计算

`ComboTrainDataset` 初始化时会按 `ndays` 遍历，每天调用：
- `gen_feature(feature_ds)`：逐因子路径读取 memmap + 归一化等变换
- `gen_label(label_ds)`：读取 label 并做 winsorize/标准化

该过程是 Python for-loop + 多次小粒度 I/O，随 `ndays × 因子数 × 标的数` 放大，通常是首要耗时来源之一。

---

### 瓶颈 B：日频推理阶段的频繁数据搬运与同步边界不清

在日循环中：
- `feature_window` 由 loader/buffer 产生（CPU 侧）；
- 若模型在 GPU（如 `eg-torch device=cuda`），`predict` 中会把输入 `.to(self.device)`；
- 返回时再 `detach().cpu()`，随后在 `runCombo` 又转 numpy 送入回测。

这意味着存在 **CPU→GPU（输入）+ GPU→CPU（输出）** 的每日往返。
当 `tsDays × inst × features` 较大时，通信时间可与前向计算同量级甚至更高。

---

### 瓶颈 C：回测 `step` 中逐股票 Python 循环

`DailyBacktest.step` 在每个交易日按股票逐个计算买卖手数与现金变动。
这部分高度依赖 Python 标量循环，对大股票池（几千只）会产生显著解释器开销。

---

### 瓶颈 D：`alpha_analysis` 的全历史计算

结束阶段会将 `alpha_history` 组装成全历史 DataFrame，再加载 1d/5d label，掩码处理后计算多种 IC。
该阶段包含大矩阵构建与多次相关系数计算，区间长时可能出现一次性“尾部大耗时”。

---

### 瓶颈 E：回测收尾中的 benchmark 拉取与绘图

`finalize` 中 `_pnl_summary/draw` 会查询 benchmark 并生成多张图片。
若外部服务响应慢，会让总耗时出现不稳定尾巴（并非纯计算瓶颈，但影响端到端时延）。

---

## 3. CPU/GPU 视角的瓶颈分解

### 3.1 CPU 时间

主要来自：
- Memmap 数据读取与预处理（feature/label 的逐日加载、标准化、截断）。
- Python 层循环（dataset 构建、回测撮合）。
- Pandas 对齐、DataFrame 构建、CSV/Parquet 输出。

### 3.2 GPU 运算时间

主要来自：
- `eg-torch` 的训练 `forward/backward`。
- `predict` 前向（单日单 batch）。

注意：若 batch 很小（例如单日推理 batch=1），GPU 利用率可能偏低，核函数启动开销占比会上升。

### 3.3 CPU↔GPU 通信时间

主要来自：
- 训练：DataLoader 产出 CPU tensor，batch 内 `.to(device)`。
- 推理：`x_window.unsqueeze(0).to(device)` 与 `pred.detach().cpu()`。
- 回测输入要求 numpy/Series，导致最终必须回到 CPU。

通信是否“主瓶颈”取决于：
- 输入张量体积
- 每日推理调用频次
- GPU 前向计算复杂度（轻模型更容易被通信主导）

---

## 4. 这些瓶颈能否通过代码定位？

**可以，而且当前代码已经具备“第一层定位能力”。**

### 4.1 已有定位能力（开箱可用）
- `PerfMonitor` 已对 `setup/combine/alpha_convert/backtest_step/backtest_finalize/alpha_analysis` 计时。
- `ComboBase` 进一步细化到 `combo_gen_pos/combo_train/combo_load_checkpoint/combo_save_checkpoint`。

这能先回答：**慢在训练、推理、回测，还是分析收尾**。

### 4.2 当前定位盲区
- 对 GPU 时间缺少“核函数真实执行时间”拆分（默认不同步）。
- 对 CPU↔GPU 拷贝时间缺少单独指标（例如 H2D、D2H 分别计时）。
- 对 I/O 时间缺少分项（读取 factor、读取 label、写文件）。
- 对回测循环缺少“每日报单股票数/成交股票数”与耗时关联指标。

### 4.3 结论
- **能定位到“阶段级”瓶颈**：是的。
- **能精确定位到“GPU计算 vs GPU通信 vs I/O”**：目前还不够精细，但代码结构允许继续下钻。

---

## 5. 与专业量化框架对比下的成熟度评级

给出一个工程化视角的粗粒度评分（10 分制）：

- 架构清晰度：**7.5/10**（模块边界清楚，入口统一）
- 可扩展性：**7/10**（ResearchModel 插拔较友好）
- 可观测性：**6/10**（有 monitor，但缺 GPU 通信/内核精细拆分）
- 性能工程成熟度：**5.5/10**（串行流程、Python 循环较多）
- 生产稳健性：**6/10**（外部依赖/尾部耗时抖动风险存在）

**总体成熟度：约 6.4/10（研究可用，生产化前夜）**。

---

## 6. 建议的“分析优先级”路线（仍不改代码）

1. 先开 `monitor.enabled=true` 跑一段代表性区间（例如 6-12 个月），确认最大耗时阶段。
2. 对最大阶段再拆：
   - 若是 `combo_train`：观察数据构建与模型 fit 的占比。
   - 若是 `combo_gen_pos`：观察单日推理时长是否受模型复杂度影响显著。
   - 若是 `backtest_step`：观察股票池规模与 step 时间线性关系。
3. 在 GPU 任务上做一次同步对照试验：`sync_cuda=false` vs `true`，估算“被异步掩盖”的真实 GPU 时间。
4. 单独统计 `alpha_analysis` 与 `finalize`，避免把尾部批处理误判为日循环性能问题。

---

## 7. 最终回答（针对你的问题）

- 这个框架相对“专业量化框架”的定位是：**研究到准生产之间**，可用但性能工程尚未系统化。
- 流程优化评估上，最大风险点通常在：
  1) 训练数据构建 I/O + 预处理，
  2) 推理阶段 CPU/GPU 往返，
  3) 回测逐股票 Python 循环，
  4) 收尾阶段全历史 IC/绘图。
- 你关心的 **CPU、GPU 通信、GPU 运算** 三类瓶颈：
  - 目前代码可定位到“阶段级”瓶颈；
  - 还不能一步到位精确拆出通信与核函数占比；
  - 但通过现有 monitor + 针对性补充埋点，完全可以定位出来。

