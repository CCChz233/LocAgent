#!/usr/bin/env bash
set -euo pipefail

python method/index/batch_build_index.py \
  --repo_path playground/locbench_repos \
  --strategy langchain_recursive \
  --index_dir index_data/new_langchain_recursive \
  --model_name models/CodeRankEmbed \
  --trust_remote_code \
  --num_processes 4 \
  --gpu_ids 0,1,2,3 \
  --batch_size 32

python method/index/batch_build_index.py \
  --repo_path playground/locbench_repos \
  --strategy langchain_token \
  --index_dir index_data/new_langchain_token \
  --model_name models/CodeRankEmbed \
  --trust_remote_code \
  --num_processes 4 \
  --gpu_ids 0,1,2,3 \
  --batch_size 32

python method/index/batch_build_index.py \
  --repo_path playground/locbench_repos \
  --strategy langchain_fixed \
  --index_dir index_data/new_langchain_fixed \
  --model_name models/CodeRankEmbed \
  --trust_remote_code \
  --num_processes 4 \
  --gpu_ids 0,1,2,3 \
  --batch_size 32
