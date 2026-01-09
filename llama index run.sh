set -euo pipefail
python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --strategy llamaindex_code \
    --index_dir index_data/new_llamaindex_code_index \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 4,5,6,7 \
    --batch_size 32

python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --strategy llamaindex_sentence \
    --index_dir index_data/new_llamaindex_sentence_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 4,5,6,7 \
    --batch_size 32

python method/index/batch_build_index.py \
    --repo_path playground/locbench_repos \
    --strategy llamaindex_token \
    --index_dir index_data/new_llamaindex_token_CodeRankEmbed \
    --model_name models/CodeRankEmbed \
    --trust_remote_code \
    --num_processes 4 \
    --gpu_ids 4,5,6,7 \
    --batch_size 32