#!/usr/bin/env python
"""
使用预建的稠密索引进行代码定位检索

用法:
    python method/dense/run_with_index.py \
        --index_dir index_data/dense_index_fixed \
        --dataset_path data/Loc-Bench_V1_dataset.jsonl \
        --output_folder outputs/dense_locator_fixed \
        --model_name models/rlretriever \
        --top_k_blocks 50 --top_k_files 10
"""

import argparse
import json
import logging
import os
import os.path as osp
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目根目录到Python路径
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from method.mapping import ASTBasedMapper, GraphBasedMapper
from method.utils import instance_id_to_repo_name as utils_instance_id_to_repo_name, clean_file_path


def instance_id_to_repo_name(instance_id: str) -> str:
    """将 instance_id 转换为 repo_name（去掉 issue 编号后缀）"""
    # 使用 utils 中的统一实现
    return utils_instance_id_to_repo_name(instance_id)


def get_problem_text(instance: dict) -> str:
    """从实例中提取问题描述"""
    for key in ("problem_statement", "issue", "description", "prompt", "text"):
        val = instance.get(key)
        if val:
            return val
    return ""


def embed_texts(
    texts: List[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """编码查询文本"""
    from torch.utils.data import Dataset, DataLoader
    
    class TextDataset(Dataset):
        def __init__(self, items: List[str]):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx: int):
            encoded = tokenizer(
                self.items[idx],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    ds = TextDataset(texts)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for input_ids, attn_mask in loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            token_embeddings = outputs[0]
            mask = attn_mask.unsqueeze(-1)
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            sent_emb = summed / counts
            sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)
            outs.append(sent_emb.cpu())
    return torch.cat(outs, dim=0)


def load_index(repo_name: str, index_dir: str) -> Tuple[torch.Tensor, List[dict]]:
    """加载预建的索引，支持标准格式和替换格式"""
    
    # 尝试1: 标准格式（单下划线，转换后的格式）
    repo_index_dir = Path(index_dir) / repo_name
    embeddings_file = repo_index_dir / "embeddings.pt"
    metadata_file = repo_index_dir / "metadata.jsonl"
    
    if embeddings_file.exists() and metadata_file.exists():
        try:
            embeddings = torch.load(embeddings_file, weights_only=True, map_location='cpu')
            metadata = []
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    metadata.append(json.loads(line))
            
            # 验证索引有效性
            if embeddings.shape[0] == len(metadata) and embeddings.shape[0] > 0:
                return embeddings, metadata
        except Exception as e:
            logging.warning(f"Failed to load index for {repo_name}: {e}")
    
    # 尝试2: 替换格式（处理连字符等特殊情况）
    # 如果 repo_name 包含连字符，尝试替换为下划线
    if '-' in repo_name:
        alt_repo_name = repo_name.replace('-', '_')
        alt_repo_index_dir = Path(index_dir) / alt_repo_name
        alt_embeddings_file = alt_repo_index_dir / "embeddings.pt"
        alt_metadata_file = alt_repo_index_dir / "metadata.jsonl"
        
        if alt_embeddings_file.exists() and alt_metadata_file.exists():
            try:
                embeddings = torch.load(alt_embeddings_file, weights_only=True, map_location='cpu')
                metadata = []
                with open(alt_metadata_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        metadata.append(json.loads(line))
                
                if embeddings.shape[0] == len(metadata) and embeddings.shape[0] > 0:
                    logging.info(f"Found index using alternative naming: {alt_repo_name}")
                    return embeddings, metadata
            except Exception as e:
                logging.warning(f"Failed to load alternative index for {repo_name}: {e}")
    
    logging.warning(f"Index not found for {repo_name} in {index_dir}")
    return None, None


def rank_files(
    block_scores: List[Tuple[int, float]],
    metadata: List[dict],
    top_k_files: int,
    repo_name: str,
) -> List[str]:
    """根据代码块分数聚合到文件级别"""
    file_scores: Dict[str, float] = {}
    for block_idx, score in block_scores:
        block_meta = metadata[block_idx]
        file_path = block_meta['file_path']
        # 清理文件路径，使其与 GT 格式一致（相对路径）
        cleaned_path = clean_file_path(file_path, repo_name)
        file_scores[cleaned_path] = file_scores.get(cleaned_path, 0.0) + float(score)
    
    # 按分数排序
    ranked = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    return [f for f, _ in ranked[:top_k_files]]


def run(args: argparse.Namespace) -> None:
    # 加载模型（只用于编码查询）
    if args.force_cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device("cpu")
    
    trust_remote_code = getattr(args, 'trust_remote_code', False)
    print(f"Loading model from {args.model_name}, trust_remote_code={trust_remote_code}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=trust_remote_code)
        model = AutoModel.from_pretrained(args.model_name, trust_remote_code=trust_remote_code).to(device)
    except ValueError as e:
        if "trust_remote_code" in str(e).lower():
            print(f"❌ Error: Model requires trust_remote_code=True. Please add --trust_remote_code to your command.")
            raise
        else:
            raise
    model.eval()
    print(f"Model loaded on {device}")
    
    # 加载数据集
    with open(args.dataset_path, "r") as f:
        instances = [json.loads(line) for line in f]
    if args.eval_n_limit:
        instances = instances[: args.eval_n_limit]
    
    print(f"Loaded {len(instances)} instances")
    
    # 提取查询文本
    queries = [get_problem_text(ins) for ins in instances]
    
    # 编码所有查询（批量处理）
    print("Encoding queries...")
    query_embeddings = embed_texts(queries, model, tokenizer, args.max_length, args.batch_size, device)
    print(f"Encoded {len(query_embeddings)} queries")
    
    # 创建输出目录
    os.makedirs(args.output_folder, exist_ok=True)
    output_file = osp.join(args.output_folder, "loc_outputs.jsonl")
    
    # 根据mapper_type选择映射器
    if args.mapper_type == "graph":
        if not args.graph_index_dir:
            raise ValueError(
                "使用Graph映射器时必须提供 --graph_index_dir 参数。\n"
                "示例: --mapper_type graph --graph_index_dir index_data/Loc-Bench_V1/graph_index_v2.3"
            )
        mapper = GraphBasedMapper(graph_index_dir=args.graph_index_dir)
        print(f"✓ 使用 Graph映射器 (graph_index_dir: {args.graph_index_dir})")
    else:  # args.mapper_type == "ast"
        if not args.repos_root:
            raise ValueError(
                "使用AST映射器时必须提供 --repos_root 参数。\n"
                "示例: --mapper_type ast --repos_root playground/locbench_repos"
            )
        mapper = ASTBasedMapper(repos_root=args.repos_root)
        print(f"✓ 使用 AST映射器 (repos_root: {args.repos_root})")
    
    # 缓存索引（避免重复加载）
    index_cache: Dict[str, Tuple[torch.Tensor, List[dict]]] = {}
    
    def get_cached_index(repo_name: str):
        if repo_name not in index_cache:
            embeddings, metadata = load_index(repo_name, args.index_dir)
            # 索引保留在 CPU，避免 GPU 内存不足
            index_cache[repo_name] = (embeddings, metadata)
        return index_cache[repo_name]
    
    # 处理每个实例
    print("Running retrieval...")
    index_found = 0
    index_missing = 0
    missing_repos = []
    
    with open(output_file, "w") as fout, torch.no_grad():
        for ins, query_emb in tqdm(zip(instances, query_embeddings), total=len(instances), desc="Retrieving"):
            instance_id = ins.get("instance_id", "")
            repo_name = instance_id_to_repo_name(instance_id)
            
            # 加载索引（在 CPU 上）
            embeddings, metadata = get_cached_index(repo_name)
            
            if embeddings is None or metadata is None:
                # 索引不存在，返回空结果
                index_missing += 1
                missing_repos.append(repo_name)
                record = {
                    "instance_id": instance_id,
                    "found_files": [],
                    "found_modules": [],
                    "found_entities": [],
                    "raw_output_loc": [],
                }
                fout.write(json.dumps(record) + "\n")
                continue
            
            index_found += 1
            
            # 计算相似度（临时移到 GPU）
            query_emb_gpu = query_emb.to(device)
            embeddings_gpu = embeddings.to(device)  # 临时移到 GPU
            scores = torch.matmul(query_emb_gpu.unsqueeze(0), embeddings_gpu.t()).squeeze(0)  # (num_blocks,)
            scores = scores.cpu()  # 移回 CPU 以便后续处理
            
            # 获取 Top-K 代码块
            topk = min(args.top_k_blocks, scores.numel())
            if topk == 0:
                found_files = []
                found_modules = []
                found_entities = []
            else:
                topk_scores, topk_idx = torch.topk(scores, k=topk)
                block_scores = list(zip(topk_idx.tolist(), topk_scores.tolist()))
                found_files = rank_files(block_scores, metadata, args.top_k_files, repo_name)
                
                # 映射代码块到函数/模块
                # 清理 top_blocks 中的 file_path，使其与 GT 格式一致
                top_blocks = []
                for idx, _ in block_scores:
                    block = metadata[idx].copy()  # 复制以避免修改原始 metadata
                    original_path = block.get('file_path', '')
                    if original_path:
                        block['file_path'] = clean_file_path(original_path, repo_name)
                    top_blocks.append(block)
                
                found_modules, found_entities = mapper.map_blocks_to_entities(
                    blocks=top_blocks,
                    instance_id=instance_id,
                    top_k_modules=args.top_k_modules,
                    top_k_entities=args.top_k_entities,
                )
            
            # 保存结果
            record = {
                "instance_id": instance_id,
                "found_files": found_files,
                "found_modules": found_modules,
                "found_entities": found_entities,
                "raw_output_loc": [],
            }
            fout.write(json.dumps(record) + "\n")
    
    # 输出索引查找统计
    print(f"\nIndex Statistics:")
    print(f"  Found: {index_found}/{len(instances)}")
    print(f"  Missing: {index_missing}/{len(instances)}")
    if missing_repos:
        unique_missing = list(set(missing_repos))[:10]
        print(f"  Missing repos (sample): {unique_missing}")
    
    print(f"\nResults saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用预建的稠密索引进行代码定位检索")
    parser.add_argument("--index_dir", type=str, required=True,
                        help="预建索引目录（如 index_data/dense_index_fixed）")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="数据集 JSONL 文件路径")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--model_name", type=str, default="models/rlretriever",
                        help="模型路径（用于编码查询）")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="允许执行模型仓库中的自定义代码（CodeRankEmbed 等模型需要）")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="指定使用的 GPU ID（默认 0）")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大 token 长度")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批量大小")
    parser.add_argument("--top_k_blocks", type=int, default=50,
                        help="Top-K 代码块")
    parser.add_argument("--top_k_files", type=int, default=15,
                        help="Top-K 文件")
    parser.add_argument("--top_k_modules", type=int, default=15,
                        help="返回的模块数量")
    parser.add_argument("--top_k_entities", type=int, default=15,
                        help="返回的实体数量")
    parser.add_argument(
        "--mapper_type",
        type=str,
        choices=["ast", "graph"],
        default="ast",
        help="映射器类型: 'ast' (AST解析, 默认) 或 'graph' (Graph索引+span_ids)"
    )
    parser.add_argument(
        "--graph_index_dir",
        type=str,
        default=None,
        help="Graph索引目录（使用 --mapper_type graph 时必需）"
    )
    parser.add_argument(
        "--repos_root",
        type=str,
        default="playground/locbench_repos",
        help="代码仓库根目录（使用 --mapper_type ast 时必需，默认: playground/locbench_repos）"
    )
    parser.add_argument("--eval_n_limit", type=int, default=0,
                        help="限制处理的实例数量（0 表示全部）")
    parser.add_argument("--force_cpu", action="store_true",
                        help="强制使用 CPU")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

