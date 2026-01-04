# 方法评测框架（设计对齐）

## 核心目标
为不同代码定位算法提供统一的评测框架，便于横向对比和复现。

## 三层结构
- 统一接口层（预期：`method/base.py`）：定义标准输出 `LocResult` 与可选基类 `BaseMethod`。
- 共享工具层（预期：`method/utils.py`）：数据集加载（在线/离线）、结果保存（JSONL）、索引加载（图/BM25/稠密）、通用参数解析。
- 方法实现层（`method/{method_name}/`）：每个方法独立目录，包含 `run.py` 入口、索引构建脚本（如需）和方法特有逻辑。

## 标准三阶段流程
1) 索引构建（按需）
   - LocAgent：图索引 + BM25 索引（预构建）。
   - BM25 基线：复用 LocAgent 索引。
   - RLCoder：稠密向量索引（UniXcoder 编码 + 余弦相似度，不使用 BM25）。
2) 运行评测
   - 加载索引与数据集（在线 HuggingFace / 离线 JSONL）。
   - 对每个实例运行定位算法。
   - 输出标准 `loc_outputs.jsonl`（`found_files/modules/entities`）。
3) 评估结果
   - 使用 `evaluation/eval_metric.py` 计算 Acc/Recall/NDCG/MAP@k。
   - 横向对比各方法。

## 关键决策
- 各方法使用各自适配的索引：LocAgent/BM25 用图+BM25；RLCoder 用稠密检索。
- 统一输出格式，内部提取路径可不同；需支持在线/离线数据。

## 当前状态
- 架构思路与规范明确；BM25 基线可用；LocAgent 既有流程保留。
- RLCoder 已提供稠密索引脚本 `method/RLCoder/build_dense_index.py`（基于 `nov3630/RLRetriever`，UniXcoder 编码 + 余弦相似度）。
- 已知问题：RLCoder 的模块/函数级提取仍依赖简单规则，需改进（建议接入图索引或 AST 解析）。

## 扩展指引
- 新方法：新建 `method/{name}`，提供 `run.py`，输出 `loc_outputs.jsonl` 的 `LocResult` 字段；如需自有索引，可添加构建脚本。
- 框架（base/utils）负责数据加载、结果落盘、评测调用、跳过已处理实例等通用逻辑。

## 命令速查（当前可用流程）
> 路径与数据集按需调整；建议先设置 `export PYTHONPATH=$(pwd)`。

- LocAgent 生成索引（图 + BM25）：
  ```bash
  bash scripts/gen_graph_index.sh      # 生成 graph_index_v2.3
  bash scripts/gen_bm25_index.sh       # 生成 BM25_index
  ```

- RLCoder 稠密索引（UniXcoder/nov3630/RLRetriever，无 BM25）：
  ```bash
  python method/RLCoder/build_dense_index.py \
    --repo_path playground/locbench_repos/SomeRepo \
    --output_dir index_data/RLCoder/SomeRepo_dense \
    --model_name nov3630/RLRetriever \
    --max_length 512 --batch_size 8 --block_size 15
  # 输出：embeddings.pt + metadata.jsonl
  ```

- BM25 基线运行（复用 LocAgent 索引）：
  ```bash
  python scripts/run_bm25_baseline.py \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/bm25_locbench \
    --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \
    --bm25_index_dir index_data/Loc-Bench_V1/BM25_index
  ```

- 评测（通用，适用于任意 `loc_outputs.jsonl` 或 merged 文件）：
  ```bash
  python - <<'PY'
  from evaluation.eval_metric import evaluate_results
  level2key = {"file":"found_files","module":"found_modules","function":"found_entities"}
  print(evaluate_results("outputs/bm25_locbench/loc_outputs.jsonl",
                         level2key,
                         dataset_path="data/Loc-Bench_V1_dataset.jsonl"))
  PY
  ```

## RepoCoder 定位（稠密检索版本）
- 单仓库文件级定位（输出 `loc_outputs.jsonl`，暂不产模块/函数）：
  ```bash
  export TOKENIZERS_PARALLELISM=false
  python method/RepoCoder/run_locator.py \
    --repo_path /path/to/repo \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/repocoder_locator \
    --model_name /Users/chz/code/LocAgent/models/rlretriever \
    --mode dense \
    --block_size 15 --max_length 512 --batch_size 8 \
    --top_k_blocks 50 --top_k_files 10
  # 仅做冒烟可加: --max_blocks_per_file 2 --eval_n_limit 5
  ```
  多仓库（自动按 instance_id 推导仓库名，根目录下需有相应 repo 文件夹）：
  ```bash
  export TOKENIZERS_PARALLELISM=false
  python method/RepoCoder/run_locator.py \
    --repos_root playground/locbench_repos \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --output_folder outputs/repocoder_locator \
    --model_name /Users/chz/code/LocAgent/models/rlretriever \
    --mode dense \
    --block_size 15 --max_length 512 --batch_size 8 \
    --top_k_blocks 50 --top_k_files 10 \
    --eval_n_limit 10        # 先小样本冒烟
  # 需要限制块规模可加 --max_blocks_per_file 2
  ```
- 若需论文版 Jaccard（BoW）检索，改用 `--mode jaccard`，可不加载模型（其他参数相同）。

## RLCoder 模型与本地化
- 推荐缓存路径：`/Users/chz/code/LocAgent/models/rlretriever`
- 下载/续传（含权重）：
  ```bash
  huggingface-cli download nov3630/RLRetriever \
    --local-dir /Users/chz/code/LocAgent/models/rlretriever \
    --resume-download \
    --local-dir-use-symlinks False
  ```
- 使用本地模型跑索引：
  ```bash
  python method/RLCoder/build_dense_index.py \
    --repo_path /actual/repo/path \
    --output_dir index_data/RLCoder/<name>_dense \
    --model_name /Users/chz/code/LocAgent/models/rlretriever \
    --block_size 15 --max_length 512 --batch_size 8
  ```
- 本地快速冒烟（避免全库跑太久）：
  ```bash
  export TOKENIZERS_PARALLELISM=false
  python method/RLCoder/build_dense_index.py \
    --repo_path dependency_graph \
    --output_dir index_data/RLCoder/smoke_depgraph \
    --model_name /Users/chz/code/LocAgent/models/rlretriever \
    --block_size 15 --max_length 512 --batch_size 4 \
    --max_blocks_per_file 2
  # 生成 embeddings.pt + metadata.jsonl 后，再在服务器跑全量
  ```

## 对比评测流水线（分阶段）
### 第 1 阶段：准备索引
- 图索引（若缺）：`bash scripts/gen_graph_index.sh`
- BM25 索引（Loc-Bench V1，全量示例）：
  ```bash
  export PYTHONPATH=$(pwd)
  python build_bm25_index.py \
    --dataset czlll/Loc-Bench_V1 \
    --split test \
    --repo_path playground/locbench_repos \
    --index_dir index_data \
    --num_processes 4
  ```
  生成到 `index_data/Loc-Bench_V1/BM25_index/`。若只跑子集，可调 `--num_processes` 并在对比配置中设置 `eval_n_limit`。
- 稠密/Jaccard locator 无需预建索引（运行时现场切块编码/分词），仅需模型（dense 模式）。

### 第 2 阶段：运行对比与评测
- 使用配置化运行器（默认配置在 `configs/retrieval_benchmark.json`，含 bm25/dense/jaccard）：
  ```bash
  export PYTHONPATH=$(pwd)
  python scripts/run_retrieval_benchmark.py --config configs/retrieval_benchmark.json
  ```
  - 若未准备 BM25 索引，可在配置文件里移除 bm25 方法，或先完成第 1 阶段再运行。
  - 配置项关键字段：
    - `dataset_path`: `data/Loc-Bench_V1_dataset.jsonl`
    - `repos_root`: `playground/locbench_repos`
    - `methods`: 方法列表（bm25/locator-dense/locator-jaccard 等），各自 `output_folder`、参数。
    - `eval_n_limit`: 小样本冒烟（设 0 为全量）。
  - 输出：每方法的 `loc_outputs.jsonl` + `eval_results.csv`；汇总表 `outputs/compare_*.csv`。

## 支持的基线与位置
- BM25 基线：`scripts/run_bm25_baseline.py`（需 graph_index + BM25_index）
- Dense 定位（RLRetriever）：`method/RepoCoder/run_locator.py --mode dense`（模型默认 `models/rlretriever`）
- Jaccard/BoW 定位：`method/RepoCoder/run_locator.py --mode jaccard`（无模型依赖）
- 稠密索引构建（固定行、不重叠）：`method/RLCoder/build_dense_index.py`
- 滑窗稠密索引构建（RepoCoder 风格 window_size/slice_size）：`method/RepoCoder/build_sliding_index.py`
- RLCoder 原论文分块（未编码，输出块清单）：`method/RLCoder/build_fixed_blocks.py`

### 简化入口（拆分到独立子目录）
- BM25 运行：`method/bm25/run.py`（包装 `scripts/run_bm25_baseline.py`）
- Dense 索引：`method/dense/build_index.py`（包装 `method/RLCoder/build_dense_index.py`）
- Dense 检索：`method/dense/run.py`（包装 `run_locator.py --mode dense`）
- Jaccard 检索：`method/jaccard/run.py`（包装 `run_locator.py --mode jaccard`）
- 滑窗稠密索引：`method/sliding/build_index.py`（包装 `method/RepoCoder/build_sliding_index.py`）

### 常用索引构建命令
- 统一稠密索引（多策略可选：fixed/sliding/rl_fixed/rl_mini）：
  ```bash
  export TOKENIZERS_PARALLELISM=false
  python method/index/build_index.py \
    --repo_path /path/to/repo \
    --output_dir index_data/index_fixed_rlretriever \
    --model_name models/rlretriever \
    --strategy fixed \           # 或 sliding / rl_fixed / rl_mini
    --block_size 15 \            # strategy=fixed 时使用
    --window_size 20 --slice_size 2 \  # strategy=sliding 时使用
    --max_length 512 --batch_size 8
  ```
- RLCoder 分块+索引一步到位示例：
  ```bash
  # 固定块（12 非空行）
  python method/index/build_index.py \
    --repo_path /path/to/repo \
    --output_dir index_data/index_rl_fixed \
    --model_name models/rlretriever \
    --strategy rl_fixed \
    --max_length 512 --batch_size 8
  # mini_block（空行分段、≤15 行）
  python method/index/build_index.py \
    --repo_path /path/to/repo \
    --output_dir index_data/index_rl_mini \
    --model_name models/rlretriever \
    --strategy rl_mini \
    --max_length 512 --batch_size 8
  ```
- 固定行稠密索引（RLRetriever，示例）：
  ```bash
  export TOKENIZERS_PARALLELISM=false
  python method/RLCoder/build_dense_index.py \
    --repo_path /path/to/repo \
    --output_dir index_data/RLCoder/<name>_dense \
    --model_name models/rlretriever \
    --block_size 15 --max_length 512 --batch_size 8
  ```
- 滑窗稠密索引（RepoCoder 风格）：
  ```bash
  export TOKENIZERS_PARALLELISM=false
  python method/RepoCoder/build_sliding_index.py \
    --repo_path /path/to/repo \
    --output_dir index_data/RepoCoder/<name>_sliding \
    --model_name models/rlretriever \
    --window_size 20 --slice_size 2 \
    --max_length 512 --batch_size 8
  ```
- RLCoder 原论文分块清单（未编码）：
  ```bash
  # fixed_block（12 非空行一块）
  python method/RLCoder/build_fixed_blocks.py \
    --repo_path /path/to/repo \
    --output_path index_data/RLCoder/blocks_fixed.jsonl \
    --enable_fixed_block
  # mini_block（默认，空行分段再拼 ≤15 行）
  python method/RLCoder/build_fixed_blocks.py \
    --repo_path /path/to/repo \
    --output_path index_data/RLCoder/blocks_mini.jsonl
  ```
- BM25 索引（Loc-Bench V1 示例）：
  ```bash
  export PYTHONPATH=$(pwd)
  python build_bm25_index.py \
    --dataset czlll/Loc-Bench_V1 \
    --split test \
    --repo_path playground/locbench_repos \
    --index_dir index_data \
    --num_processes 4
  ```
