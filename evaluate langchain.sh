set -euo pipefail

python method/dense/run_with_index.py \
  --index_dir /workspace/LocAgent/index_data/new_langchain_fixed/dense_index_langchain_fixed \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --output_folder outputs/new_langchain_fixed_coderank \
  --model_name /workspace/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --gpu_id 0 \
  --top_k_blocks 50 \
  --top_k_files 15 \
  --top_k_modules 15 \
  --top_k_entities 15 \
  --batch_size 16

python method/dense/run_with_index.py \
  --index_dir /workspace/LocAgent/index_data/new_langchain_recursive/dense_index_langchain_recursive \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --output_folder outputs/new_langchain_recursive_coderank \
  --model_name /workspace/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --gpu_id 0 \
  --top_k_blocks 50 \
  --top_k_files 15 \
  --top_k_modules 15 \
  --top_k_entities 15 \
  --batch_size 16

python method/dense/run_with_index.py \
  --index_dir /workspace/LocAgent/index_data/new_langchain_token/dense_index_langchain_token \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --output_folder outputs/new_langchain_token_coderank \
  --model_name /workspace/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --gpu_id 0 \
  --top_k_blocks 50 \
  --top_k_files 15 \
  --top_k_modules 15 \
  --top_k_entities 15 \
  --batch_size 16