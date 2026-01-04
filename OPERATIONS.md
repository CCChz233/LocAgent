# LocAgent 操作指南

本文档汇总常用操作命令（环境、下载、索引、基线、评估、排错）。默认在仓库根目录执行。

## 目录结构

```
/workspace/LocAgent/
├── config/                                  # 配置文件
│   ├── .env                                # 环境变量（API Key 等）
│   └── .env.example                        # 配置模板
├── data/                                    # 本地数据集
│   └── Loc-Bench_V1_dataset.jsonl          # Loc-Bench 数据集（560条）
├── index_data/Loc-Bench_V1/
│   ├── graph_index_v2.3/                   # 图索引 (*.pkl)
│   └── BM25_index/                         # BM25 索引
├── playground/locbench_repos/              # 克隆的仓库
├── outputs/                                # 实验输出
│   ├── bm25_locbench/                      # BM25 基线结果
│   └── locagent_*/                         # LocAgent 结果
└── run_locagent.sh                         # LocAgent 启动脚本
```

## 环境准备（macOS 推荐）
1) 使用 Python 3.11（3.12 会依赖冲突）
```
conda create -n locagent python=3.11
conda activate locagent
pip install -r requirements-macos-min.txt
```

## 环境依赖说明
- `requirements.txt`：完整依赖（含 Azure/LLM 相关组件与 CUDA 12 的 `nvidia-*` 包），更适合 **Linux + GPU**。
- `requirements-macos-min.txt`：最小可运行集合（BM25/图索引/基础流程），但**不包含 Azure embedding 组件**。

若遇到 `ModuleNotFoundError: llama_index.embeddings.azure_openai`，可补装：
```
pip install llama-index-embeddings-azure-openai llama-index-llms-azure-openai azure-core azure-identity
```

若遇到 `llama-index` 版本不兼容问题，确保安装正确版本：
```
pip install llama-index==0.11.22 llama-index-core==0.11.22 llama-index-retrievers-bm25==0.4.0
```

## Linux 环境安装（服务器）
推荐 Python 3.11：
```
conda create -n locagent python=3.11
conda activate locagent
```
GPU（CUDA 12）环境：
```
pip install -r requirements.txt
```
CPU/轻量环境（建议先跑通 BM25/图索引）：
```
pip install -r requirements-macos-min.txt
pip install llama-index-embeddings-azure-openai llama-index-llms-azure-openai azure-core azure-identity faiss-cpu
```
提示：如果只做 BM25/图索引，不用 Azure/OpenAI embedding，也可以暂时跳过补装，遇到报错再补。

## 数据集

### 在线模式（需要网络）
数据会缓存到 `~/.cache/huggingface/datasets`（可通过 `HF_HOME` 或 `HF_DATASETS_CACHE` 改路径）。

### 离线模式（无网络环境）
将数据集导出为 JSONL 文件放在 `data/` 目录：
```bash
mkdir -p data
# 在有网络的机器上导出：
python -c "
from datasets import load_dataset
import json
ds = load_dataset('czlll/Loc-Bench_V1', split='test')
with open('Loc-Bench_V1_dataset.jsonl', 'w') as f:
    for item in ds:
        f.write(json.dumps(item) + '\n')
"
# 将文件拷贝到 data/Loc-Bench_V1_dataset.jsonl
```

## 克隆仓库（只下载、不建索引）
脚本：`scripts/clone_locbench_repos.py`  
默认走 SSH，目录结构为 `output_dir/<rank>/<org_repo>`（与原始脚本一致）。
如果你要用 SSH，先确保本机能通过：`ssh -T git@github.com`。

基础命令（建议先小样本验证）：
```
python scripts/clone_locbench_repos.py \
  --dataset czlll/Loc-Bench_V1 \
  --split test \
  --output_dir playground/locbench_repos \
  --num_processes 2 \
  --limit 10
```

常用选项：
- `--use_https`：强制 HTTPS（默认 SSH）
- `--skip_lfs`：跳过 Git LFS 大文件（节省空间）
- `--limit N`：仅下载前 N 个实例
- `--num_processes`：并发数（建议 1–4，过高易被限速）

提示：如果要最大化复用已有 clone，后续索引阶段建议使用**相同的 `--num_processes`**。  
若想完全避免重复下载，最稳妥的是 clone 与索引都用 `--num_processes 1`。

## 完整版下载（SSH + LFS）
以下是“完整仓库+含 LFS”的推荐命令（速度取决于网络与 GitHub 限速，无法保证 1 小时内完成）：
```
python scripts/clone_locbench_repos.py \
  --dataset czlll/Loc-Bench_V1 \
  --split test \
  --output_dir playground/locbench_repos \
  --num_processes 2
```
如果 SSH 仍然慢，可切到 HTTPS 试试：
```
python scripts/clone_locbench_repos.py \
  --dataset czlll/Loc-Bench_V1 \
  --split test \
  --output_dir playground/locbench_repos \
  --num_processes 2 \
  --use_https
```

## 下载进度与数量
统计已完成的 repo 数量（以 `.git` 目录计）：
```
find playground/locbench_repos -name ".git" -type d | wc -l
```
查看总占用空间：
```
du -sh playground/locbench_repos
```
查看体积最大的几个仓库：
```
du -sh playground/locbench_repos/*/* | sort -h | tail
```

## 生成图索引（Graph Index）

**重要**：运行前必须设置 `PYTHONPATH`！

```bash
cd /workspace/LocAgent
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 从本地仓库生成（推荐，不需要网络）
python dependency_graph/batch_build_graph.py \
  --dataset 'czlll/Loc-Bench_V1' \
  --split test \
  --repo_path playground/locbench_repos \
  --num_processes 60

# 或者自动下载仓库（需要网络）
python dependency_graph/batch_build_graph.py \
  --dataset 'czlll/Loc-Bench_V1' \
  --split test \
  --num_processes 30 \
  --download_repo \
  --repo_path playground/build_graph
```

说明：
- `--download_repo`：自动克隆仓库
- `--num_processes`：并行进程数（根据 CPU 核心数调整，128核机器可用 60-80）
- 索引输出到 `index_data/Loc-Bench_V1/graph_index_v2.3/*.pkl`

## 生成 BM25 索引

```bash
cd /workspace/LocAgent
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 从本地仓库生成
python build_bm25_index.py \
  --dataset 'czlll/Loc-Bench_V1' \
  --split test \
  --repo_path playground/locbench_repos \
  --num_processes 60
```

索引输出到 `index_data/Loc-Bench_V1/BM25_index/<repo_name>/`

## 运行 LocAgent

> ⚠️ **重要提示**：运行前必须设置环境变量 `GRAPH_INDEX_DIR` 和 `BM25_INDEX_DIR`，否则程序会尝试从 GitHub 克隆仓库并失败！
>
> 确保以下环境变量已设置：
> ```bash
> export GRAPH_INDEX_DIR="/workspace/LocAgent/index_data/Loc-Bench_V1/graph_index_v2.3"
> export BM25_INDEX_DIR="/workspace/LocAgent/index_data/Loc-Bench_V1/BM25_index"
> ```

### 1. 配置 API Key

创建并编辑配置文件：

```bash
# 复制模板
cp config/.env.example config/.env

# 编辑配置
nano config/.env
```

**config/.env 示例（火山引擎 DeepSeek）**：
```env
# 火山引擎 API
OPENAI_API_KEY=your-volcengine-api-key
OPENAI_API_BASE=https://ark.cn-beijing.volces.com/api/v3

# 索引目录
GRAPH_INDEX_DIR=index_data/Loc-Bench_V1/graph_index_v2.3
BM25_INDEX_DIR=index_data/Loc-Bench_V1/BM25_index
```

**其他 API 配置示例**：
```env
# OpenAI 官方
OPENAI_API_KEY=sk-your-openai-key
# OPENAI_API_BASE 不需要设置

# Azure OpenAI
AZURE_API_KEY=your-azure-key
AZURE_API_BASE=https://your-endpoint.openai.azure.com/

# DeepSeek 官方
DEEPSEEK_API_KEY=your-deepseek-key
```

### 2. 使用启动脚本（推荐）

```bash
cd /workspace/LocAgent

# 测试运行（5条样本，2并行）
./run_locagent.sh 5 2

# 正式运行（300条样本，10并行）
./run_locagent.sh 300 10

# 完整数据集
./run_locagent.sh 560 10
```

启动脚本会自动：
- 加载 `config/.env` 配置
- 激活 conda 环境
- 设置 PYTHONPATH
- 创建带时间戳的输出目录

### 3. 使用 vLLM 部署本地模型

如果你有本地模型（如 Qwen2.5-7B-Instruct），可以用 vLLM 部署后运行 LocAgent。

#### 安装 vLLM
```bash
pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 启动 vLLM 服务
```bash
# 单卡部署 Qwen2.5-7B-Instruct (端口 8000)
CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/model/Qwen__Qwen2.5-7B-Instruct/Qwen/Qwen2___5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen2.5-7b \
  --dtype auto --max-model-len 16384 \
  --gpu-memory-utilization 0.9 > /workspace/vllm.log 2>&1 &

# 双卡部署（吞吐量翻倍）
CUDA_VISIBLE_DEVICES=6,7 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/model/Qwen__Qwen2.5-7B-Instruct/Qwen/Qwen2___5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen2.5-7b \
  --tensor-parallel-size 2 \
  --dtype auto --max-model-len 32768 > /workspace/vllm.log 2>&1 &

# 验证服务是否启动
curl http://localhost:8000/v1/models
```

#### 配置 LocAgent 使用 vLLM
```bash
# 创建配置文件
cat > /workspace/LocAgent/config/.env << 'EOF'
OPENAI_API_KEY=EMPTY
OPENAI_API_BASE=http://localhost:8000/v1
GRAPH_INDEX_DIR=/workspace/LocAgent/index_data/Loc-Bench_V1/graph_index_v2.3
BM25_INDEX_DIR=/workspace/LocAgent/index_data/Loc-Bench_V1/BM25_index
EOF
```

#### 运行 LocAgent（使用本地模型）

> ⚠️ **必须先设置环境变量**，否则会报 `UnboundLocalError: cannot access local variable 'G'` 错误！

```bash
cd /workspace/LocAgent
source /root/miniconda3/etc/profile.d/conda.sh && conda activate locagent

# 方法1: 从 config/.env 加载环境变量
export $(grep -v '^#' config/.env | xargs)
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 方法2: 手动设置环境变量（如果没有 config/.env）
# export OPENAI_API_KEY=EMPTY
# export OPENAI_API_BASE=http://localhost:8000/v1
# export GRAPH_INDEX_DIR="/workspace/LocAgent/index_data/Loc-Bench_V1/graph_index_v2.3"
# export BM25_INDEX_DIR="/workspace/LocAgent/index_data/Loc-Bench_V1/BM25_index"
# export PYTHONPATH=$PYTHONPATH:$(pwd)

# 验证环境变量（必须显示正确路径，不能为空！）
echo "GRAPH_INDEX_DIR: $GRAPH_INDEX_DIR"
echo "BM25_INDEX_DIR: $BM25_INDEX_DIR"
ls $GRAPH_INDEX_DIR/*.pkl 2>/dev/null | wc -l  # 应该显示 165

# 清理之前失败产生的临时目录（可选）
rm -rf playground/[0-9a-f]*-*-*-*-*/

# 测试运行（10条样本）
python auto_search_main.py \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --model 'openai/qwen2.5-7b' \
  --localize --merge \
  --output_folder outputs/locagent_qwen25_7b \
  --eval_n_limit 10 \
  --num_processes 4 \
  --use_function_calling \
  --simple_desc

# 完整运行（560条样本）
python auto_search_main.py \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --model 'openai/qwen2.5-7b' \
  --localize --merge \
  --output_folder outputs/locagent_qwen25_7b \
  --eval_n_limit 560 \
  --num_processes 8 \
  --use_function_calling \
  --simple_desc
```

#### 其他可用本地模型
| 模型 | 路径 | 显存需求 | 兼容性 |
|------|------|----------|--------|
| Qwen2.5-7B-Instruct | `/workspace/model/Qwen__Qwen2.5-7B-Instruct/Qwen/Qwen2___5-7B-Instruct` | ~14GB | ✅ 推荐 |
| Qwen3-8B | `/workspace/model/Qwen__Qwen3-8B/Qwen/Qwen3-8B` | ~16GB | ⚠️ 需关闭 thinking |
| DeepSeek-R1-Distill-7B | `/workspace/model/deepseek-ai__DeepSeek-R1-Distill-Qwen-7B/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | ~14GB | ⚠️ 需测试 |

> ⚠️ **Qwen3 兼容性问题**：Qwen3 默认启用 thinking 模式（输出 `<think>...</think>`），会导致 LocAgent 的 function calling 解析失败。解决方法：
> 1. 在 vLLM 启动时添加 `--override-generation-config '{"enable_thinking": false}'` 关闭 thinking 模式
> 2. 或者直接使用 Qwen2.5-7B-Instruct（推荐）

#### vLLM 常用命令
```bash
# 查看日志
tail -f /workspace/vllm.log

# 停止服务
pkill -f "vllm.entrypoints"

# 查看 GPU 使用
nvidia-smi
```

### 4. 手动运行（远程 API）

```bash
cd /workspace/LocAgent
source /root/miniconda3/etc/profile.d/conda.sh && conda activate locagent
export $(grep -v '^#' config/.env | xargs)
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 使用自定义模型（火山引擎 DeepSeek）
python auto_search_main.py \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --model 'openai/deepseek-v3-1-terminus' \
  --localize --merge \
  --output_folder outputs/locagent_deepseek \
  --eval_n_limit 300 \
  --num_processes 10 \
  --use_function_calling \
  --simple_desc

# 使用 Azure OpenAI
python auto_search_main.py \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --model 'azure/gpt-4o' \
  --localize --merge \
  --output_folder outputs/locagent_azure \
  --eval_n_limit 300 \
  --num_processes 30 \
  --use_function_calling \
  --simple_desc

# 使用 OpenAI 官方
python auto_search_main.py \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --model 'openai/gpt-4o-2024-05-13' \
  --localize --merge \
  --output_folder outputs/locagent_openai \
  --eval_n_limit 300 \
  --num_processes 20 \
  --use_function_calling \
  --simple_desc
```

### 5. 参数说明

#### 必选参数
| 参数 | 说明 |
|------|------|
| `--output_folder` | 输出目录（必须） |
| `--localize` | 启动定位流程（必须） |

#### 推荐参数
| 参数 | 说明 |
|------|------|
| `--merge` | 合并多个样本的结果（**`num_samples > 1` 时强烈推荐**） |
| `--dataset_path` | 本地数据集路径（离线模式） |
| `--model` | 模型名，格式：`provider/model-name` |
| `--use_function_calling` | 使用 function calling（推荐） |
| `--simple_desc` | 简化函数描述（Qwen/DeepSeek 推荐开启） |

#### 可选参数
| 参数 | 说明 |
|------|------|
| `--eval_n_limit` | 评估样本数量（0=全部，测试时建议 10-50） |
| `--num_samples` | 每个实例采样次数（默认 2，>1 时需要 `--merge`） |
| `--num_processes` | 并行进程数（默认 -1=自动，取决于 API rate limit） |
| `--ranking_method` | merge 时的排序方法（`'mrr'` 或 `'majority'`，默认 `'mrr'`） |
| `--log_level` | 日志级别（`INFO`/`DEBUG`） |
| `--timeout` | 超时时间（秒，默认 900） |

> ⚠️ **重要提示**：
> - 当 `--num_samples > 1` 时，**必须使用 `--merge`** 合并结果，否则评估时会出现格式错误
> - `--merge` 可以在运行时一起执行（推荐），也可以后续单独运行
> - 评估时必须使用 `merged_loc_outputs_mrr.jsonl`，不要使用 `loc_outputs.jsonl`

### 6. 输出文件

```
outputs/locagent_<timestamp>/
├── args.json                    # 运行参数
├── loc_outputs.jsonl            # 原始定位结果（列表的列表格式）
├── loc_trajs.jsonl              # 推理轨迹（含 token 用量）
├── merged_loc_outputs_mrr.jsonl # 合并后的结果（MRR 排序，单个列表格式）
└── localize.log                 # 运行日志
```

**文件格式说明**：
- **`loc_outputs.jsonl`**：原始结果，当 `num_samples=2` 时，`found_files` 格式为 `[[sample1_files], [sample2_files]]`
- **`merged_loc_outputs_mrr.jsonl`**：合并后的结果，`found_files` 格式为 `[file1, file2, ...]`（单个列表）
- **评估时必须使用 `merged_loc_outputs_mrr.jsonl`**，否则评估结果会出错或全为 0

## 运行 BM25 基线

```bash
cd /workspace/LocAgent
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 在线模式（需要网络访问 HuggingFace）
python scripts/run_bm25_baseline.py \
  --dataset czlll/Loc-Bench_V1 \
  --split test \
  --output_folder outputs/bm25_locbench \
  --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \
  --bm25_index_dir index_data/Loc-Bench_V1/BM25_index

# 离线模式（使用本地数据集文件）
python scripts/run_bm25_baseline.py \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --output_folder outputs/bm25_locbench \
  --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \
  --bm25_index_dir index_data/Loc-Bench_V1/BM25_index
```

## 评估

### Merge 操作说明

当 `--num_samples > 1` 时，每个实例会生成多个样本，`loc_outputs.jsonl` 中的格式为：
```json
{
  "instance_id": "xxx",
  "found_files": [["file1.py", "file2.py"], ["file2.py", "file3.py"]],  // 列表的列表
  "found_modules": [[...], [...]],
  "found_entities": [[...], [...]]
}
```

**Merge 操作**会将多个样本合并为单个排序列表，生成 `merged_loc_outputs_mrr.jsonl`：
```json
{
  "instance_id": "xxx",
  "found_files": ["file2.py", "file1.py", "file3.py"],  // 单个列表（按 MRR 权重排序）
  "found_modules": [...],
  "found_entities": [...]
}
```

**Merge 方法**：
- **MRR (Mean Reciprocal Rank)**：考虑文件在每个样本中的排名位置，排名越靠前权重越大（默认）
- **Majority**：按出现频率排序，不考虑位置

**使用建议**：
- 推荐在运行时加 `--merge` 参数（一次完成）
- 也可以后续单独运行 merge（灵活性更高）
- 评估时**必须使用 merged 文件**

### 评测指标说明

| 指标 | 说明 |
|------|------|
| `acc` | Accuracy@k - 前 k 个结果是否完全命中所有 ground truth |
| `recall` | Recall@k - 前 k 个结果中命中的比例 |
| `ndcg` | NDCG@k - 归一化折损累积增益，考虑排序质量 |
| `precision` | Precision@k - 精确率 |
| `map` | MAP@k - 平均精确率 |

评测级别：
- **File**: 文件级定位，k 值默认 [1, 3, 5]
- **Module**: 模块级定位（类），k 值默认 [5, 10]
- **Function**: 函数级定位，k 值默认 [5, 10]

### 评估 LocAgent

> ⚠️ **重要提示**：评估时**必须使用 `merged_loc_outputs_mrr.jsonl` 文件**，不要使用 `loc_outputs.jsonl`！
> - `loc_outputs.jsonl` 格式为列表的列表（`[[sample1], [sample2]]`），评估函数无法正确处理
> - `merged_loc_outputs_mrr.jsonl` 格式为单个列表（`[file1, file2, ...]`），可直接用于评估
> - 如果评估结果全为 0，通常是使用了错误的文件导致的

```bash
cd /workspace/LocAgent
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 替换为你的实际输出目录
OUTPUT_DIR="outputs/locagent_qwen25_7b"

# 基础评测（仅 Accuracy）
python - <<PY
from evaluation.eval_metric import evaluate_results
level2key = {"file":"found_files","module":"found_modules","function":"found_entities"}
result = evaluate_results("${OUTPUT_DIR}/merged_loc_outputs_mrr.jsonl", level2key,
                          metrics=['acc'],
                          dataset_path="data/Loc-Bench_V1_dataset.jsonl")
print(result.to_string())
PY

# 完整评测（所有指标）
python - <<PY
from evaluation.eval_metric import evaluate_results
level2key = {"file":"found_files","module":"found_modules","function":"found_entities"}
result = evaluate_results("${OUTPUT_DIR}/merged_loc_outputs_mrr.jsonl", level2key,
                          metrics=['acc', 'recall', 'ndcg', 'precision', 'map'],
                          dataset_path="data/Loc-Bench_V1_dataset.jsonl")
print(result.to_string())
result.to_csv("${OUTPUT_DIR}/eval_results.csv")
PY
```

### 评估 BM25 基线

```bash
cd /workspace/LocAgent
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 离线模式（推荐）
python - <<'PY'
from evaluation.eval_metric import evaluate_results
level2key = {"file":"found_files","module":"found_modules","function":"found_entities"}
result = evaluate_results("outputs/bm25_locbench/loc_outputs.jsonl", level2key,
                          metrics=['acc'],
                          dataset_path="data/Loc-Bench_V1_dataset.jsonl")
print(result.to_string())
result.to_csv("outputs/bm25_locbench/eval_results.csv")
PY

# 在线模式
python - <<'PY'
from evaluation.eval_metric import evaluate_results
level2key = {"file":"found_files","module":"found_modules","function":"found_entities"}
print(evaluate_results("outputs/bm25_locbench/loc_outputs.jsonl", level2key,
                       dataset="czlll/Loc-Bench_V1", split="test"))
PY
```

### 手动合并结果（如 merge 步骤未完成）

如果运行时没有加 `--merge` 参数，或者 `merged_loc_outputs_mrr.jsonl` 文件缺失/不完整，可以后续手动执行 merge：

**方法1：使用命令行参数（推荐）**
```bash
cd /workspace/LocAgent
export PYTHONPATH=$PYTHONPATH:$(pwd)
OUTPUT_DIR="outputs/locagent_qwen25_7b"

python auto_search_main.py \
    --merge \
    --output_folder "${OUTPUT_DIR}" \
    --ranking_method mrr
```

**方法2：使用 Python 脚本（如方法1不可用）**

如果 `merged_loc_outputs_mrr.jsonl` 记录数少于 `loc_outputs.jsonl`，可手动执行：

```bash
cd /workspace/LocAgent
OUTPUT_DIR="outputs/locagent_qwen25_7b"

python - <<'PY'
import json
import collections
from collections import Counter

def merge_sample_locations(found_files, found_modules, found_entities, ranking_method='mrr'):
    def rank_locs(found_locs, ranking_method='mrr'):
        flat_locs = [loc for sublist in found_locs for loc in sublist]
        locs_weights = collections.defaultdict(float)
        if ranking_method == 'majority':
            loc_counts = Counter(flat_locs)
            for loc, count in loc_counts.items():
                locs_weights[loc] = count
        elif ranking_method == 'mrr':
            for sample_locs in found_locs:
                for rank, loc in enumerate(sample_locs, start=1):
                    locs_weights[loc] += 1 / rank
        ranked_loc_weights = sorted(locs_weights.items(), key=lambda x: x[1], reverse=True)
        return [loc for loc, _ in ranked_loc_weights]
    return (rank_locs(found_files, ranking_method),
            rank_locs(found_modules, ranking_method),
            rank_locs(found_entities, ranking_method))

OUTPUT_DIR = "${OUTPUT_DIR}"
merge_file = f'{OUTPUT_DIR}/merged_loc_outputs_mrr.jsonl'
output_file = f'{OUTPUT_DIR}/loc_outputs.jsonl'

with open(merge_file, 'w') as f:
    pass

with open(output_file, 'r') as file:
    count = 0
    for line in file:
        loc_data = json.loads(line)
        if loc_data['found_files'] == [[]] or loc_data['found_files'] == [[], []]:
            loc_data['found_files'] = []
            loc_data['found_modules'] = []
            loc_data['found_entities'] = []
        else:
            ranked = merge_sample_locations(
                loc_data['found_files'], loc_data['found_modules'],
                loc_data['found_entities'], ranking_method='mrr')
            loc_data['found_files'], loc_data['found_modules'], loc_data['found_entities'] = ranked
        with open(merge_file, 'a') as f:
            f.write(json.dumps(loc_data) + '\n')
        count += 1
print(f'成功合并 {count} 条记录到 {merge_file}')
PY
```

### 查看定位结果统计

```bash
cd /workspace/LocAgent
OUTPUT_DIR="outputs/locagent_qwen25_7b"

python - <<'PY'
import json
total = has_files = has_modules = has_entities = empty = 0
with open("${OUTPUT_DIR}/merged_loc_outputs_mrr.jsonl", 'r') as f:
    for line in f:
        data = json.loads(line)
        total += 1
        if data['found_files']: has_files += 1
        if data['found_modules']: has_modules += 1
        if data['found_entities']: has_entities += 1
        if not any([data['found_files'], data['found_modules'], data['found_entities']]): empty += 1
print(f'总实例数: {total}')
print(f'有文件级结果: {has_files} ({has_files/total*100:.1f}%)')
print(f'有模块级结果: {has_modules} ({has_modules/total*100:.1f}%)')
print(f'有函数级结果: {has_entities} ({has_entities/total*100:.1f}%)')
print(f'空结果: {empty} ({empty/total*100:.1f}%)')
PY
```

### BM25 Baseline 参考结果

| 级别 | Acc@1 | Acc@3 | Acc@5 | Acc@10 |
|------|-------|-------|-------|--------|
| File | 36.96% | 46.07% | 53.39% | - |
| Module | - | - | 35.89% | 41.43% |
| Function | - | - | 23.93% | 27.86% |

### Claude-3.5 参考结果（论文基准）

| 级别 | Acc@1 | Acc@3 | Acc@5 | Acc@10 |
|------|-------|-------|-------|--------|
| File | 77.74% | 91.97% | 94.16% | - |
| Module | - | - | 86.50% | 87.59% |
| Function | - | - | 73.36% | 77.37% |

## 输出与路径
- 本地数据集：`data/Loc-Bench_V1_dataset.jsonl`
- 图索引：`index_data/Loc-Bench_V1/graph_index_v2.3/*.pkl`
- BM25 索引：`index_data/Loc-Bench_V1/BM25_index/<repo_name>/`
- 运行输出：`outputs/<experiment>/loc_outputs.jsonl`

## 清理临时目录

运行 LocAgent 时，程序会在 `playground/` 下创建 UUID 格式的临时目录（如 `44e6e332-86a4-416a-b576-584e5f21d846/`）。

**产生原因**：每处理一个实例，程序都会创建临时目录用于存放仓库数据。正常情况下处理完成后会自动删除，但如果程序异常终止（如环境变量未设置、网络超时等），这些目录会残留。

**清理命令**：
```bash
# 清理所有 UUID 格式的临时目录
rm -rf /workspace/LocAgent/playground/[0-9a-f]*-*-*-*-*/

# 查看临时目录占用空间
du -sh /workspace/LocAgent/playground/*/

# 只保留 locbench_repos 和 build_graph
cd /workspace/LocAgent/playground
ls | grep -E '^[0-9a-f]{8}-' | xargs rm -rf
```

**建议**：运行 LocAgent 前先清理临时目录，避免占用过多磁盘空间。

## Git 提交注意事项
下载的仓库默认在 `playground/` 下，`.gitignore` 已忽略 `playground/`、`index_data/`、`outputs/` 和 `data/`，不会被提交。  
推送前可用 `git status` 再确认，避免把大文件加入版本库。

## 常见问题

### UnboundLocalError: cannot access local variable 'G'
**原因**：环境变量 `GRAPH_INDEX_DIR` 或 `BM25_INDEX_DIR` 未设置，程序找不到索引文件  
**解决**：
```bash
# 设置环境变量（使用绝对路径）
export GRAPH_INDEX_DIR="/workspace/LocAgent/index_data/Loc-Bench_V1/graph_index_v2.3"
export BM25_INDEX_DIR="/workspace/LocAgent/index_data/Loc-Bench_V1/BM25_index"

# 或者从 config/.env 加载
export $(grep -v '^#' config/.env | xargs)

# 验证
echo $GRAPH_INDEX_DIR  # 不能为空
ls $GRAPH_INDEX_DIR/*.pkl | wc -l  # 应该显示 165
```

### ModuleNotFoundError: dependency_graph
**解决**：设置 PYTHONPATH
```bash
cd /workspace/LocAgent
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### ModuleNotFoundError: llama_index.embeddings.azure_openai
**解决**：安装缺失的包
```bash
pip install llama-index-embeddings-azure-openai faiss-cpu
```

### llama_index 版本不兼容
**解决**：降级到兼容版本
```bash
pip install llama-index==0.11.22 llama-index-core==0.11.22
```

### Network is unreachable（无法访问 HuggingFace）
**解决**：使用离线模式
1. 提前下载数据集为 JSONL 文件
2. 使用 `--dataset_path` 参数指定本地文件

### corpus size should be larger than top-k
**原因**：某些小仓库的文件数少于 top_k  
**解决**：已在脚本中自动跳过这类情况

### git clone 失败/限速
**解决**：
- 降低 `--num_processes`
- 切换 SSH/HTTPS
- 使用代理或热点

### 仓库过大
**解决**：加 `--skip_lfs`（不影响代码评测）

### KeyError: 'openai/xxx' (未知模型的成本计算)
**原因**：自定义模型不在成本字典中  
**解决**：已自动处理，成本显示为 0

### FileNotFoundError: xxx.pkl (索引文件找不到)
**原因**：instance_id 与索引文件名格式不匹配  
**解决**：已修复，代码会自动转换 `REPO__name-123` → `REPO_name`

### git clone 网络超时
**原因**：无网络或 GitHub 访问受限  
**解决**：LocAgent 现在支持完全离线运行，只需：
1. 提前准备好本地仓库 (`playground/locbench_repos/`)
2. 提前生成好索引 (`index_data/`)
3. 使用本地数据集 (`--dataset_path`)

### playground 下出现大量 UUID 目录
**原因**：程序异常终止时，临时目录未被清理  
**解决**：
```bash
rm -rf /workspace/LocAgent/playground/[0-9a-f]*-*-*-*-*/
```

### 评估结果全为 0
**原因**：使用了 `loc_outputs.jsonl` 而不是 `merged_loc_outputs_mrr.jsonl` 进行评估  
**症状**：评估时所有指标（acc, recall, ndcg 等）都是 0  
**解决**：
```bash
# ✅ 正确：使用 merged 文件
python - <<PY
from evaluation.eval_metric import evaluate_results
level2key = {"file":"found_files","module":"found_modules","function":"found_entities"}
result = evaluate_results("outputs/xxx/merged_loc_outputs_mrr.jsonl", level2key,
                          dataset_path="data/Loc-Bench_V1_dataset.jsonl")
print(result.to_string())
PY

# ❌ 错误：使用原始文件（会导致结果全为0）
# evaluate_results("outputs/xxx/loc_outputs.jsonl", ...)
```

如果 merged 文件不存在，可以手动执行 merge：
```bash
python auto_search_main.py --merge --output_folder outputs/xxx --ranking_method mrr
```

### Qwen3 模型运行无结果输出
**症状**：日志健康但没有 `loc_outputs.jsonl`，大量 `convert none fncall messages failed` 错误  
**原因**：Qwen3 默认启用 thinking 模式，输出 `<think>...</think>` 格式，LocAgent 无法解析  
**解决**：
```bash
# 方法1: 关闭 thinking 模式启动 vLLM
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/model/Qwen__Qwen3-8B/Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8001 \
  --served-model-name qwen3-8b \
  --override-generation-config '{"enable_thinking": false}' \
  --dtype auto --max-model-len 16384

# 方法2: 使用 Qwen2.5-7B（推荐，无此问题）
```

## 监控命令

```bash
# 查看索引生成进度
ls index_data/Loc-Bench_V1/graph_index_v2.3 | wc -l
ls index_data/Loc-Bench_V1/BM25_index | wc -l

# 查看 BM25 基线进度
wc -l outputs/bm25_locbench/loc_outputs.jsonl

# 查看 LocAgent 进度
LATEST=$(ls -td outputs/locagent_* 2>/dev/null | head -1)
echo "目录: $LATEST"
wc -l "$LATEST"/*.jsonl 2>/dev/null
tail -20 "$LATEST/localize.log" 2>/dev/null

# 查看系统资源
htop

# 查看运行中的进程
ps aux | grep -E "(auto_search_main|run_bm25)" | grep -v grep
```

## 快速开始（完整流程）

```bash
cd /workspace/LocAgent

# 1. 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate locagent

# 2. 准备数据（确保以下目录/文件存在）
#    - data/Loc-Bench_V1_dataset.jsonl
#    - index_data/Loc-Bench_V1/graph_index_v2.3/ (165 个 .pkl 文件)
#    - index_data/Loc-Bench_V1/BM25_index/ (164 个目录)

# 3. 配置环境变量（⚠️ 必须！）
cat > config/.env << 'EOF'
OPENAI_API_KEY=EMPTY
OPENAI_API_BASE=http://localhost:8000/v1
GRAPH_INDEX_DIR=/workspace/LocAgent/index_data/Loc-Bench_V1/graph_index_v2.3
BM25_INDEX_DIR=/workspace/LocAgent/index_data/Loc-Bench_V1/BM25_index
EOF

# 4. 加载环境变量并验证
export $(grep -v '^#' config/.env | xargs)
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "GRAPH_INDEX_DIR: $GRAPH_INDEX_DIR"
ls $GRAPH_INDEX_DIR/*.pkl | wc -l  # 应显示 165

# 5. 启动 vLLM 服务（如使用本地模型）
CUDA_VISIBLE_DEVICES=0 nohup python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/model/Qwen__Qwen2.5-7B-Instruct/Qwen/Qwen2___5-7B-Instruct \
  --host 0.0.0.0 --port 8000 --served-model-name qwen2.5-7b \
  --dtype auto --max-model-len 16384 > /workspace/vllm.log 2>&1 &
sleep 30 && curl http://localhost:8000/v1/models  # 验证服务启动

# 6. 运行 LocAgent
python auto_search_main.py \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --model 'openai/qwen2.5-7b' \
  --localize --merge \
  --output_folder outputs/locagent_qwen25_7b \
  --eval_n_limit 10 \
  --num_processes 4 \
  --use_function_calling \
  --simple_desc

# 7. 评估结果
python - <<'PY'
from evaluation.eval_metric import evaluate_results
level2key = {"file":"found_files","module":"found_modules","function":"found_entities"}
LATEST = "outputs/locagent_qwen25_7b"  # 替换为实际目录
result = evaluate_results(f"{LATEST}/merged_loc_outputs_mrr.jsonl", level2key,
                          dataset_path="data/Loc-Bench_V1_dataset.jsonl")
print(result.to_string())
PY
```
