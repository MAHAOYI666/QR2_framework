# comb2-organize

`comb2-organize` 用来把 research 模型接到 `comb2` 和 `comb2-pcmaster` 上，完成训练、信号生成和回测。

## 准备依赖仓库

先下载两个依赖仓库并安装：

```bash
git clone git@10.20.4.19:factorsimteam/combo/comb2.git
git clone git@10.20.4.19:factorsimteam/combo/comb2_pcmaster.git
```

### 编译并安装 `comb2`

```bash
cd comb2
python3 setup.py build_ext --inplace
python3 -m pip install .
```

### 安装 `comb2-pcmaster`

```bash
cd ../comb2_pcmaster
python3 -m pip install .
```

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

示例命令：

```bash
cd /root/autodl-tmp/comb2-example-lgbm
ls
```

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

在 `comb2-organize` 目录下执行：

```bash
cd /root/autodl-tmp/comb2-organize
python3 runCombo.py /root/autodl-tmp/comb2-example-lgbm/config.xml
```

或者：

```bash
python3 runCombo.py --config /root/autodl-tmp/comb2-example-lgbm/config.xml
```

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
