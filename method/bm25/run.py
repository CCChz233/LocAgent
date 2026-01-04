#!/usr/bin/env python
"""
BM25 基线方法 CLI 入口

使用 BM25 算法进行代码定位，输出标准 loc_outputs.jsonl 格式。

Usage:
    python method/bm25/run.py \\
        --dataset czlll/Loc-Bench_V1 \\
        --split test \\
        --output_folder outputs/bm25_results \\
        --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \\
        --bm25_index_dir index_data/Loc-Bench_V1/BM25_index
    
    # 离线模式
    python method/bm25/run.py \\
        --dataset_path data/Loc-Bench_V1_dataset.jsonl \\
        --output_folder outputs/bm25_results \\
        --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3 \\
        --bm25_index_dir index_data/Loc-Bench_V1/BM25_index
"""

import argparse
import logging
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method.bm25.retriever import BM25Method
from method.utils import (
    load_dataset_instances,
    load_processed_ids,
    save_result,
    add_common_args,
    resolve_index_dirs,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="BM25 基线代码定位方法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 添加通用参数
    add_common_args(parser)
    
    # BM25 特定参数（如有）
    parser.add_argument("--build_if_missing", action="store_true",
                        help="如果索引缺失则构建（需要仓库路径）")
    parser.add_argument("--repo_base_dir", type=str, default="playground/bm25_baseline",
                        help="仓库基础目录（用于构建缺失索引）")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    
    # 创建输出目录
    os.makedirs(args.output_folder, exist_ok=True)
    output_file = os.path.join(args.output_folder, args.output_file)
    
    # 解析索引目录
    graph_index_dir, bm25_index_dir = resolve_index_dirs(args)
    logger.info(f"Graph index dir: {graph_index_dir}")
    logger.info(f"BM25 index dir: {bm25_index_dir}")
    
    # 加载数据集
    instances = load_dataset_instances(
        dataset_path=args.dataset_path or None,
        dataset=args.dataset if not args.dataset_path else None,
        split=args.split,
        limit=args.eval_n_limit or None,
    )
    logger.info(f"Loaded {len(instances)} instances")
    
    # 加载已处理的实例（断点续跑）
    processed_ids = load_processed_ids(output_file)
    if processed_ids:
        logger.info(f"Found {len(processed_ids)} processed instances, skipping...")
    
    # 初始化方法
    method = BM25Method(
        graph_index_dir=graph_index_dir,
        bm25_index_dir=bm25_index_dir,
        top_k_files=args.top_k_files,
        top_k_modules=args.top_k_modules,
        top_k_entities=args.top_k_entities,
    )
    
    # 运行定位
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for instance in instances:
        instance_id = instance.get("instance_id")
        if not instance_id:
            logger.warning("Instance missing instance_id, skipping")
            continue
        
        if instance_id in processed_ids:
            skip_count += 1
            continue
        
        try:
            result = method.localize(instance)
            save_result(result, output_file)
            
            if result.is_empty():
                fail_count += 1
                logger.warning(f"[{instance_id}] Empty result")
            else:
                success_count += 1
                logger.info(f"[{instance_id}] Found {len(result.found_files)} files, "
                           f"{len(result.found_modules)} modules, "
                           f"{len(result.found_entities)} entities")
        
        except Exception as e:
            fail_count += 1
            logger.error(f"[{instance_id}] Error: {e}")
    
    # 输出统计
    logger.info("=" * 60)
    logger.info(f"Completed: {success_count} success, {fail_count} fail, {skip_count} skipped")
    logger.info(f"Output: {output_file}")


if __name__ == "__main__":
    main()

