set -euo pipefail


python method/dense/run_with_index.py \
  --index_dir /workspace/LocAgent/index_data/new_llamaindex_code_index/dense_index_llamaindex_code \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --output_folder outputs/new_llama_code_CodeRankEmbed \
  --model_name /workspace/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --gpu_id 1 \
  --batch_size 16

python method/dense/run_with_index.py \
  --index_dir /workspace/LocAgent/index_data/new_llamaindex_sentence_CodeRankEmbed/dense_index_llamaindex_sentence \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --output_folder outputs/new_llama_sentence_CodeRankEmbed \
  --model_name /workspace/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --gpu_id 1 \
  --batch_size 16

python method/dense/run_with_index.py \
  --index_dir /workspace/LocAgent/index_data/new_llamaindex_token_CodeRankEmbed/dense_index_llamaindex_token \
  --dataset_path data/Loc-Bench_V1_dataset.jsonl \
  --output_folder outputs/new_llama_token_CodeRankEmbed \
  --model_name /workspace/LocAgent/models/CodeRankEmbed \
  --trust_remote_code \
  --mapper_type ast \
  --gpu_id 1 \
  --batch_size 16