# method/ 现有算法概览

本目录放置外部/自研算法实现。当前可用条目：

## RLCoder（repository-level code completion + 检索）
- 位置：`method/RLCoder`
- 保留入口：`build_dense_index.py`（使用 `nov3630/RLRetriever` 生成稠密索引），其余训练/补全源码已移除。
- 功能：遍历代码仓库，按行数切块（默认 15 行），编码为向量并保存。
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
- 保留入口：`run_locator.py`（文件级定位，支持 dense/jaccard），其余补全流水线源码已移除。
- 用途：作为检索定位基线，对 LocAgent 的直接适配产出 `loc_outputs.jsonl`。

> 添加新算法时，建议在各自子目录内保持独立 README/入口脚本，并按 LocAgent 的评测格式输出 `loc_outputs.jsonl`，便于复用评测脚本。
