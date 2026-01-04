"""
BM25 检索器核心逻辑

基于 BM25 算法的代码定位方法。
"""

import logging
from typing import List, Optional

from dependency_graph import RepoEntitySearcher
from dependency_graph.build_graph import NODE_TYPE_CLASS, NODE_TYPE_FUNCTION

from method.base import LocResult, BaseMethod
from method.utils import (
    get_problem_text,
    instance_id_to_repo_name,
    load_graph_index,
    load_bm25_retriever,
    dedupe_append,
    clean_file_path,
)

logger = logging.getLogger(__name__)


def _module_id(entity_id: str) -> str:
    """从实体 ID 提取模块 ID"""
    file_path, name = entity_id.split(":", 1)
    if "." in name:
        name = name.split(".")[0]
    return f"{file_path}:{name}"


class BM25Method(BaseMethod):
    """
    BM25 基线方法
    
    使用 BM25 算法检索与问题描述最相关的代码片段，
    然后从检索结果中提取文件、模块和实体信息。
    """
    
    def __init__(
        self,
        graph_index_dir: str,
        bm25_index_dir: str,
        top_k_files: int = 10,
        top_k_modules: int = 10,
        top_k_entities: int = 10,
    ):
        """
        初始化 BM25 方法
        
        Args:
            graph_index_dir: 图索引目录
            bm25_index_dir: BM25 索引目录
            top_k_files: 返回的文件数量
            top_k_modules: 返回的模块数量
            top_k_entities: 返回的实体数量
        """
        self.graph_index_dir = graph_index_dir
        self.bm25_index_dir = bm25_index_dir
        self.top_k_files = top_k_files
        self.top_k_modules = top_k_modules
        self.top_k_entities = top_k_entities
    
    @property
    def name(self) -> str:
        return "BM25"
    
    def localize(self, instance: dict) -> LocResult:
        """
        使用 BM25 进行代码定位
        
        Args:
            instance: 数据集实例
        
        Returns:
            LocResult: 定位结果
        """
        instance_id = instance["instance_id"]
        repo_name = instance_id_to_repo_name(instance_id)
        
        # 加载图索引和 BM25 检索器
        graph = load_graph_index(instance_id, self.graph_index_dir)
        searcher = RepoEntitySearcher(graph) if graph else None
        
        retriever = load_bm25_retriever(instance_id, self.bm25_index_dir)
        if retriever is None:
            logger.warning(f"Missing BM25 index for {instance_id}. Returning empty result.")
            return LocResult.empty(instance_id)
        
        # 设置检索数量
        max_k = max(self.top_k_files, self.top_k_modules, self.top_k_entities)
        if hasattr(retriever, "similarity_top_k"):
            retriever.similarity_top_k = max_k
        
        # 获取查询文本
        query = get_problem_text(instance)
        if not query:
            logger.warning(f"No problem statement for {instance_id}. Returning empty result.")
            return LocResult.empty(instance_id)
        
        # 执行检索
        found_files: List[str] = []
        found_modules: List[str] = []
        found_entities: List[str] = []
        
        try:
            retrieved_nodes = retriever.retrieve(query)
        except ValueError as e:
            if "corpus size should be larger than top-k" in str(e):
                logger.warning(f"Corpus too small for {instance_id}: {e}")
                return LocResult.empty(instance_id)
            raise
        
        # 处理检索结果
        for node in retrieved_nodes:
            file_path = node.metadata.get("file_path")
            if file_path:
                file_path = clean_file_path(file_path, repo_name)
                dedupe_append(found_files, file_path, self.top_k_files)
            
            # 使用图索引提取模块和实体
            if searcher and file_path:
                span_ids = node.metadata.get("span_ids", [])
                for span_id in span_ids:
                    entity_id = f"{file_path}:{span_id}"
                    if not searcher.has_node(entity_id):
                        continue
                    
                    node_data = searcher.get_node_data([entity_id])[0]
                    if node_data["type"] == NODE_TYPE_FUNCTION:
                        dedupe_append(found_entities, entity_id, self.top_k_entities)
                        dedupe_append(found_modules, _module_id(entity_id), self.top_k_modules)
                    elif node_data["type"] == NODE_TYPE_CLASS:
                        dedupe_append(found_modules, entity_id, self.top_k_modules)
            
            # 检查是否已达到所需数量
            if (
                len(found_files) >= self.top_k_files
                and len(found_modules) >= self.top_k_modules
                and len(found_entities) >= self.top_k_entities
            ):
                break
        
        return LocResult(
            instance_id=instance_id,
            found_files=found_files,
            found_modules=found_modules,
            found_entities=found_entities,
        )


def run_bm25_localization(
    instance: dict,
    graph_index_dir: str,
    bm25_index_dir: str,
    top_k_files: int = 10,
    top_k_modules: int = 10,
    top_k_entities: int = 10,
) -> LocResult:
    """
    运行 BM25 定位（函数式接口）
    
    Args:
        instance: 数据集实例
        graph_index_dir: 图索引目录
        bm25_index_dir: BM25 索引目录
        top_k_files: 返回的文件数量
        top_k_modules: 返回的模块数量
        top_k_entities: 返回的实体数量
    
    Returns:
        LocResult: 定位结果
    """
    method = BM25Method(
        graph_index_dir=graph_index_dir,
        bm25_index_dir=bm25_index_dir,
        top_k_files=top_k_files,
        top_k_modules=top_k_modules,
        top_k_entities=top_k_entities,
    )
    return method.localize(instance)

