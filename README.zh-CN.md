# Context Perturbation Discrepancy (CPD)

[English](README.md) | **简体中文**

《Beyond Fixed-Context Likelihood: Zero-Shot AI-Generated Text Detection via Context Perturbation Discrepancy》的主要实验代码。CPD 用同一个 Falcon-7B 比较真实 token 在完整前缀和短前缀下的对数概率，扣除模型期望的增益，再按参考标准差归一化。仓库覆盖 GEN、DetectRL-X 和 HART 的直接检测、HART 草稿来源、GEN / DetectRL-X 改写对比，以及固定阈值评估。

主实验的统一入口是 `scripts/run_experiments.py`。在仓库根目录运行以下命令，可以在准备数据和模型前预览三个基准的评分命令。[实验脚本](scripts/run_experiments.py)

```bash
python scripts/run_experiments.py main --dry-run
```

## 先看主结果

在仓库根目录运行以下命令。它只用 Python 标准库，从随仓库发布的汇总重建八种方法的主表。生成的 `outputs/main_table.csv` 中，CPD 的三个数据集宏平均 AUROC 为 **94.80%**，TPR@5% 为 **79.60%**。完整九方法表还需在下文单独运行 Single-Fast。[参考结果及字段说明（英文）](docs/results.md)

```bash
python report_main_results.py
```

## 配置评分环境

复现环境为 Linux、Python 3.10.16、PyTorch 2.4.1（CUDA 12.4）和 Transformers 4.51.1。论文的 BF16 推理需要支持 BF16 的 NVIDIA GPU。CPD 自身加载一个模型，完整主表中的双模型基线同时加载两个模型，原实验使用 48 GB RTX A6000。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

下载以下两个 Falcon 检查点到本地。版本按原实验目录中的权重分片、分词器和配置文件核对。已有 Hugging Face 格式权重时，可直接把含有 `falcon-7b/` 和 `falcon-7b-instruct/` 的父目录传给 `--model-dir`。只跑 CPD 或 Single-Fast 时，只需第一个。[Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) · [Falcon-7B-Instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)

```bash
huggingface-cli download tiiuae/falcon-7b \
  --revision ec89142b67d748a1865ea4451372db8313ada0d8 --local-dir models/falcon-7b \
  --include '*.json' '*.safetensors' '*.py'
huggingface-cli download tiiuae/falcon-7b-instruct \
  --revision 8782b5c5d8c9290412416618f36a133653e85285 --local-dir models/falcon-7b-instruct \
  --include '*.json' '*.safetensors' '*.py'
```

## 从原始数据重跑

下面的第一步准备 GEN 开发集和测试集、DetectRL-X 英语测试集、HART 测试集，并检查它们与论文输入的 SHA-256。已有划分时，使用 `python prepare_data.py verify --data-dir /path/to/benchmark_2026`，再向评分命令传同一目录。[数据来源与离线准备（英文）](docs/data.md)

```bash
python prepare_data.py build
python scripts/run_experiments.py main --methods CPD --sample-size 2 --output-root outputs/smoke
```

小规模命令每个数据集抽两条 H 和两条 A，适合检查模型加载和输出。生成论文主表请运行完整数据和方法集。

```bash
python scripts/run_experiments.py main
python scripts/run_experiments.py single-fast
python report_main_results.py --main-csv outputs/cpd_results_main/summary.csv \
  --single-fast-csv outputs/cpd_results_single_fast/summary.csv \
  --output outputs/main_table_all_methods.csv
```

其他主要对比可独立运行，阈值校准依赖上面的 `main` 和下面的 `dev`。[实验设置和输出位置（英文）](docs/experiments.md)

| 实验 | 命令 |
| --- | --- |
| GEN 开发评分和阈值跨数据集迁移 | `python scripts/run_experiments.py dev`，随后运行 `python scripts/run_experiments.py calibrate` |
| HART 草稿来源和最终表达对比 | `python scripts/run_experiments.py hart` |
| GEN 人类草稿的 AI 改写 | `python prepare_data.py build gen-edit`，随后运行 `python scripts/run_experiments.py editing` |
| DetectRL-X 对 AI 草稿的三种改写 | `python prepare_data.py build rewriting`，随后运行 `python scripts/run_experiments.py rewriting` |

## 代码和产物

| 位置 | 用途 |
| --- | --- |
| `CPD.py`、`cpd_cache.py`、`cpd_paths.py` | 核心评分、可选逐样本缓存和文件命名 |
| `prepare_data.py`、`data_prep/` | 官方数据转换、GEN 主实验划分索引、文件哈希与类别计数 |
| `scripts/run_experiments.py` | 七个实验入口，共享数据、模型和输出路径参数 |
| `evaluate_cpd_thresholds.py`、`report_main_results.py` | 在开发集选择固定阈值，生成跨数据集报告 |
| `reference_results/`、`docs/`、`tests/` | 五份历史聚合表、复现说明和回归测试 |
| `outputs/` | 每次运行写入的参数、逐样本 CSV 和汇总，默认由 Git 忽略 |

在已安装依赖的环境中，可用 `python -m unittest discover -s tests` 运行 50 项主流程回归测试。主流程的四个 JSONL 已从固定版本来源重建，与历史输入逐字节一致。八方法主表的 24 组指标已用 20,000 条历史测试分数复算。发布版还以 Falcon-7B 对三个基准各抽一条 H 和一条 A 做了真实推理，六条 CPD 分数均与归档一致。完整九方法的全量推理由上述命令启动。

## 基线方法来源

主实验沿用论文中的方法名称与方向，分数越高表示越偏向 AI 草稿来源。Loglikelihood、LRR、TempTest、NTS 使用 Falcon-7B。Binoculars、双模型 Fast-DetectGPT 和 Lastde++ 按各自方法分配 Falcon-7B 与 Falcon-7B-Instruct 的角色。[GLTR](https://github.com/HendrikStrobelt/detecting-fake-text) · [DetectLLM](https://github.com/mbzuai-nlp/DetectLLM) · [TempTest](https://github.com/TMKempton/TempTest) · [NTS](https://github.com/Shixuan-Ma/NTS) · [Binoculars](https://github.com/ahans30/Binoculars) · [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) · [Lastde++](https://github.com/TrustMedia-zju/Lastde_Detector)
