# comb2-organize

`comb2-organize` 用来把 research 模型接到内置的 `comb2` 和 `comb2-pcmaster` 源码上，完成训练、信号生成和回测。

## 安装方式

下载这个仓库即可。当前仓库已经临时内置所需源码：

```text
vendor/
  comb2/
    comb2/
    src/
  comb2-pcmaster/
    comb2_pcmaster/
```

`vendor/` 只包含运行所需源码，不包含原仓库 git history、构建产物、缓存和历史输出。

## research 流程示例

下面以 `comb2-example-lgbm` 为例说明完整流程。

### 1. 准备 research 目录

示例目录结构如下：

```text
comb2-example-lgbm/
  model.py
  config.xml
  output/
  checkpoints/
```

其中：
- `model.py`：research 模型实现
- `config.xml`：运行配置
- `output/`：输出目录
- `checkpoints/`：模型 checkpoint 目录

### 2. 编写 research 模型

可以参考 `comb2-example-lgbm/model.py`。模型需要提供一个 `ResearchModel` 类，并实现以下接口：

- `fit(dataset)`
- `predict(x_window)`
- `save(path_or_buffer)`
- `load(path_or_buffer)`

### 3. 编写配置文件

可以参考 `comb2-example-lgbm/config.xml`。

关键配置包括：
- `combo.paths.model_path`：指向 research 目录下的 `model.py`
- `combo.paths.checkpoint_root`：checkpoint 输出目录
- `combo.paths.output_dir`：日志和中间结果输出目录
- `backtest.output_path`：回测结果输出目录

示例里：
- `model_path="model.py"`
- `checkpoint_root="checkpoints"`
- `output_dir="output"`
- `output_path="output/backtest"`

### 4. 运行 organize 入口

可以从任意目录执行：

```bash
python3 /root/autodl-tmp/comb2-organize/runCombo.py /root/autodl-tmp/comb2-example-lgbm/config.xml
```

或者：

```bash
python3 /root/autodl-tmp/comb2-organize/runCombo.py --config /root/autodl-tmp/comb2-example-lgbm/config.xml
```

研究员只需要把命令里的 `runCombo.py` 和 `config.xml` 换成自己的实际路径。

## 运行结果

执行后通常会产生这些输出：

- `output/train.log`
- `output/alpha_history.pt`
- `output/backtest/daily_pnl.csv`
- `checkpoints/` 下的模型文件

## 当前目录说明

- `runCombo.py`：research 运行入口，负责加载配置、调用 `comb2`、再接入 `comb2-pcmaster` 回测
- `config.py`：配置解析与默认参数
- `alpha_strategy.py`：把信号转换为回测仓位
- `vendor/comb2`：临时内置的 `comb2` 源码
- `vendor/comb2-pcmaster`：临时内置的 `comb2-pcmaster` 源码
