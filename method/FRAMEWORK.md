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
