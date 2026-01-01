# LocAgent 操作指南

本文档汇总常用操作命令（环境、下载、索引、基线、评估、排错）。默认在仓库根目录执行。

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
pip install llama-index-embeddings-azure-openai llama-index-llms-azure-openai azure-core azure-identity
```
提示：如果只做 BM25/图索引，不用 Azure/OpenAI embedding，也可以暂时跳过补装，遇到报错再补。

## 数据集与缓存
Loc-Bench 只包含问题与定位标注，不含源码。数据会缓存到：
`~/.cache/huggingface/datasets`（可通过 `HF_HOME` 或 `HF_DATASETS_CACHE` 改路径）。

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
```
python -m dependency_graph.batch_build_graph \
  --dataset 'czlll/Loc-Bench_V1' \
  --split test \
  --num_processes 2 \
  --download_repo \
  --repo_path playground/locbench_repos
```
说明：`--download_repo` 会调用 `setup_repo`，如果本地已存在同名 repo，会跳过重复 clone。

## 生成 BM25 索引
```
python -m build_bm25_index \
  --dataset 'czlll/Loc-Bench_V1' \
  --split test \
  --num_processes 2 \
  --download_repo \
  --repo_path playground/locbench_repos
```

## 运行 LocAgent
```
export GRAPH_INDEX_DIR="index_data/Loc-Bench_V1/graph_index_v2.3"
export BM25_INDEX_DIR="index_data/Loc-Bench_V1/BM25_index"

python auto_search_main.py \
  --dataset 'czlll/Loc-Bench_V1' \
  --split test \
  --model 'azure/gpt-4o' \
  --localize --merge \
  --output_folder outputs/locagent \
  --eval_n_limit 300 \
  --num_processes 50 \
  --use_function_calling \
  --simple_desc
```

## 运行 BM25 基线
```
python scripts/run_bm25_baseline.py \
  --dataset czlll/Loc-Bench_V1 \
  --split test \
  --output_folder outputs/bm25_locbench \
  --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \
  --bm25_index_dir index_data/Loc-Bench_V1/BM25_index
```

## 评估（统一指标）
```
python - <<'PY'
from evaluation.eval_metric import evaluate_results
level2key = {"file":"found_files","module":"found_modules","function":"found_entities"}
print(evaluate_results("outputs/bm25_locbench/loc_outputs.jsonl", level2key,
                       dataset="czlll/Loc-Bench_V1", split="test"))
PY
```

## 输出与路径
- 图索引：`index_data/Loc-Bench_V1/graph_index_v2.3/*.pkl`
- BM25 索引：`index_data/Loc-Bench_V1/BM25_index/<instance_id>/`
- 运行输出：`outputs/<experiment>/loc_outputs.jsonl`

## Git 提交注意事项
下载的仓库默认在 `playground/` 下，`.gitignore` 已忽略 `playground/` 和 `index_data/`，不会被提交。  
推送前可用 `git status` 再确认，避免把大文件加入版本库。

## 常见问题
- `ModuleNotFoundError: dependency_graph`：用 `python -m dependency_graph.batch_build_graph` 或设置 `PYTHONPATH`.
- `git clone` 失败/限速：降低 `--num_processes`，切换 SSH，或使用热点。
- 仓库过大：加 `--skip_lfs`（不影响代码评测）。
