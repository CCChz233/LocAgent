#!/bin/bash
# LocAgent 启动脚本
# 使用方法: ./run_locagent.sh [eval_n_limit] [num_processes]

set -e

cd /workspace/LocAgent

# 加载环境变量
if [ -f config/.env ]; then
    echo "📦 加载配置: config/.env"
    export $(grep -v '^#' config/.env | xargs)
else
    echo "❌ 错误: 请先创建 config/.env 文件"
    echo "   cp config/.env.example config/.env"
    echo "   然后编辑填写你的 API Key"
    exit 1
fi

# 激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate locagent

# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 参数默认值
EVAL_LIMIT=${1:-5}
NUM_PROCESSES=${2:-2}
MODEL=${MODEL_NAME:-"openai/deepseek-v3-1-terminus"}
OUTPUT_DIR="outputs/locagent_$(date +%Y%m%d_%H%M%S)"

echo "🚀 启动 LocAgent"
echo "   模型: $MODEL"
echo "   样本数: $EVAL_LIMIT"
echo "   并行数: $NUM_PROCESSES"
echo "   输出目录: $OUTPUT_DIR"
echo ""

python auto_search_main.py \
    --dataset_path data/Loc-Bench_V1_dataset.jsonl \
    --model "$MODEL" \
    --localize \
    --merge \
    --output_folder "$OUTPUT_DIR" \
    --eval_n_limit "$EVAL_LIMIT" \
    --num_processes "$NUM_PROCESSES" \
    --use_function_calling \
    --simple_desc

echo ""
echo "✅ 完成! 结果保存在: $OUTPUT_DIR"



