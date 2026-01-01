import argparse
import logging
import os
import os.path as osp
import pickle
from typing import Iterable, List, Optional

from datasets import load_dataset

from dependency_graph import RepoEntitySearcher
from dependency_graph.build_graph import NODE_TYPE_CLASS, NODE_TYPE_FUNCTION, VERSION
from plugins.location_tools.retriever.bm25_retriever import (
    build_code_retriever_from_repo,
    build_retriever_from_persist_dir,
)
from util.benchmark.setup_repo import setup_repo
from util.utils import append_to_jsonl, load_jsonl


def _iter_instances(data, limit: Optional[int]) -> Iterable[dict]:
    if hasattr(data, "select"):
        if limit:
            limit = min(limit, len(data))
            data = data.select(range(limit))
        for instance in data:
            yield instance
    else:
        if limit:
            data = data[:limit]
        for instance in data:
            yield instance


def _problem_text(instance: dict) -> str:
    for key in ("problem_statement", "issue", "description", "prompt", "text"):
        value = instance.get(key)
        if value:
            return value
    return ""


def _load_graph(graph_index_file: str, instance: dict, repo_base_dir: str, build_if_missing: bool):
    if osp.exists(graph_index_file):
        return pickle.load(open(graph_index_file, "rb"))
    if not build_if_missing:
        return None
    repo_dir = setup_repo(instance_data=instance, repo_base_dir=repo_base_dir, dataset=None, split=None)
    from dependency_graph.build_graph import build_graph

    os.makedirs(osp.dirname(graph_index_file), exist_ok=True)
    graph = build_graph(repo_dir, global_import=True)
    with open(graph_index_file, "wb") as f:
        pickle.dump(graph, f)
    return graph


def _load_retriever(persist_dir: str, instance: dict, repo_base_dir: str, build_if_missing: bool):
    if osp.exists(osp.join(persist_dir, "corpus.jsonl")):
        return build_retriever_from_persist_dir(persist_dir)
    if not build_if_missing:
        return None
    repo_dir = setup_repo(instance_data=instance, repo_base_dir=repo_base_dir, dataset=None, split=None)
    os.makedirs(persist_dir, exist_ok=True)
    return build_code_retriever_from_repo(repo_dir, persist_path=persist_dir)


def _module_id(entity_id: str) -> str:
    file_path, name = entity_id.split(":", 1)
    if "." in name:
        name = name.split(".")[0]
    return f"{file_path}:{name}"


def _dedupe_append(target: List[str], item: str, limit: int) -> None:
    if item not in target and len(target) < limit:
        target.append(item)


def run_instance(
    instance: dict,
    graph_index_dir: str,
    bm25_index_dir: str,
    repo_base_dir: str,
    build_if_missing: bool,
    top_k_files: int,
    top_k_modules: int,
    top_k_entities: int,
):
    instance_id = instance["instance_id"]
    graph_index_file = osp.join(graph_index_dir, f"{instance_id}.pkl")
    retriever_dir = osp.join(bm25_index_dir, instance_id)

    graph = _load_graph(graph_index_file, instance, repo_base_dir, build_if_missing)
    searcher = RepoEntitySearcher(graph) if graph else None

    retriever = _load_retriever(retriever_dir, instance, repo_base_dir, build_if_missing)
    if retriever is None:
        logging.warning("Missing BM25 index for %s. Skipping.", instance_id)
        return None

    if hasattr(retriever, "similarity_top_k"):
        retriever.similarity_top_k = max(top_k_files, top_k_modules, top_k_entities)

    query = _problem_text(instance)
    if not query:
        logging.warning("No problem statement for %s. Skipping.", instance_id)
        return None

    found_files: List[str] = []
    found_modules: List[str] = []
    found_entities: List[str] = []

    retrieved_nodes = retriever.retrieve(query)
    for node in retrieved_nodes:
        file_path = node.metadata.get("file_path")
        if file_path:
            _dedupe_append(found_files, file_path, top_k_files)

        if searcher and file_path:
            span_ids = node.metadata.get("span_ids", [])
            for span_id in span_ids:
                entity_id = f"{file_path}:{span_id}"
                if not searcher.has_node(entity_id):
                    continue
                node_data = searcher.get_node_data([entity_id])[0]
                if node_data["type"] == NODE_TYPE_FUNCTION:
                    _dedupe_append(found_entities, entity_id, top_k_entities)
                    _dedupe_append(found_modules, _module_id(entity_id), top_k_modules)
                elif node_data["type"] == NODE_TYPE_CLASS:
                    _dedupe_append(found_modules, entity_id, top_k_modules)

        if (
            len(found_files) >= top_k_files
            and len(found_modules) >= top_k_modules
            and len(found_entities) >= top_k_entities
        ):
            break

    return {
        "instance_id": instance_id,
        "found_files": found_files,
        "found_modules": found_modules,
        "found_entities": found_entities,
        "raw_output_loc": [],
    }


def main():
    parser = argparse.ArgumentParser(description="Run BM25 baseline localization.")
    parser.add_argument("--dataset", type=str, default="czlll/Loc-Bench_V1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument("--graph_index_dir", type=str, default="")
    parser.add_argument("--bm25_index_dir", type=str, default="")
    parser.add_argument("--repo_base_dir", type=str, default="playground/bm25_baseline")
    parser.add_argument("--eval_n_limit", type=int, default=0)
    parser.add_argument("--top_k_files", type=int, default=10)
    parser.add_argument("--top_k_modules", type=int, default=10)
    parser.add_argument("--top_k_entities", type=int, default=10)
    parser.add_argument("--build_if_missing", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset_name or args.dataset.split("/")[-1]
    if args.dataset_path and not args.dataset_name:
        dataset_name = "Loc-Bench_V1"

    graph_index_dir = (
        args.graph_index_dir
        or os.environ.get("GRAPH_INDEX_DIR")
        or f"index_data/{dataset_name}/graph_index_{VERSION}"
    )
    bm25_index_dir = (
        args.bm25_index_dir
        or os.environ.get("BM25_INDEX_DIR")
        or f"index_data/{dataset_name}/BM25_index"
    )

    os.makedirs(args.output_folder, exist_ok=True)
    output_file = osp.join(args.output_folder, args.output_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.dataset_path:
        data = load_jsonl(args.dataset_path)
    else:
        data = load_dataset(args.dataset, split=args.split)

    processed = set()
    if osp.exists(output_file):
        for row in load_jsonl(output_file):
            processed.add(row.get("instance_id"))

    for instance in _iter_instances(data, args.eval_n_limit or None):
        instance_id = instance.get("instance_id")
        if not instance_id or instance_id in processed:
            continue
        result = run_instance(
            instance=instance,
            graph_index_dir=graph_index_dir,
            bm25_index_dir=bm25_index_dir,
            repo_base_dir=args.repo_base_dir,
            build_if_missing=args.build_if_missing,
            top_k_files=args.top_k_files,
            top_k_modules=args.top_k_modules,
            top_k_entities=args.top_k_entities,
        )
        if result:
            append_to_jsonl(result, output_file)


if __name__ == "__main__":
    main()
