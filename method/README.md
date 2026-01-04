# method/ 现有算法概览

本目录放置外部/自研算法实现。当前可用条目：

## RLCoder（repository-level code completion + 检索）
- 位置：`method/RLCoder`
- 主要入口：`main.py`（原论文训练/推理流程），`retriever.py`（UniXcoder/RL 语义检索），`generator.py`（跨文件补全）。
- LocAgent 侧的索引生成：使用稠密检索模型 `nov3630/RLRetriever`，不再依赖 BM25。
  - 脚本：`method/RLCoder/build_dense_index.py`
  - 功能：遍历代码仓库，按行数切块（默认 15 行），用 `nov3630/RLRetriever` 编码为向量并保存。
  - 示例：
    ```bash
    python method/RLCoder/build_dense_index.py \
      --repo_path /path/to/repo \
      --output_dir /path/to/index_dir \
      --model_name nov3630/RLRetriever \
      --max_length 512 --batch_size 8 --block_size 15
    # 输出：embeddings.pt（块向量），metadata.jsonl（文件/行号/类型）
    ```
  - 说明：运行时会自动从 Hugging Face 下载模型；可加 `--force_cpu` 在无 GPU 下运行。

## RepoCoder（迭代检索-生成补全框架）
- 位置：`method/RepoCoder`
- 主要组件：`run_pipeline.py`（补全流水线）、`build_vector.py`/`search_code.py`（向量构建与检索）、`build_prompt.py`（提示构造）、`compute_score.py`（评估）。自带数据集样例与仓库快照。
- 用途：复现实验/基线，对 LocAgent 的直接适配尚未完成，如需使用请参考子目录内 `README.md`。

> 添加新算法时，建议在各自子目录内保持独立 README/入口脚本，并按 LocAgent 的评测格式输出 `loc_outputs.jsonl`，便于复用评测脚本。
