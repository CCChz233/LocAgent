import argparse
import json
import logging
import os
import os.path as osp
import multiprocessing as mp
import sys

from datasets import load_dataset

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from util.utils import load_jsonl
from util.benchmark.git_repo_manager import maybe_clone, checkout_commit, get_repo_dir_name


def repo_url(repo: str, use_ssh: bool) -> str:
    if use_ssh:
        return f"git@github.com:{repo}.git"
    return f"https://github.com/{repo}.git"


def load_instances(dataset: str, split: str, dataset_path: str, limit: int, instance_id_path: str):
    if dataset_path:
        instances = load_jsonl(dataset_path)
    else:
        instances = list(load_dataset(dataset, split=split))

    if instance_id_path:
        with open(instance_id_path, "r") as f:
            selected_ids = set(json.loads(f.read()))
        instances = [ins for ins in instances if ins.get("instance_id") in selected_ids]

    if limit and limit > 0:
        instances = instances[:limit]
    return instances


def ensure_repo(
    instance: dict,
    output_dir: str,
    rank: int,
    use_ssh: bool,
    skip_lfs: bool,
    skip_checkout: bool,
):
    repo = instance["repo"]
    base_commit = instance.get("base_commit")

    repo_base_dir = osp.join(output_dir, str(rank))
    target_dir = osp.join(repo_base_dir, get_repo_dir_name(repo))
    if skip_lfs:
        os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
    os.makedirs(repo_base_dir, exist_ok=True)
    maybe_clone(repo_url(repo, use_ssh), target_dir)

    if not skip_checkout and base_commit:
        checkout_commit(target_dir, base_commit)


def worker(
    rank: int,
    queue,
    output_dir: str,
    use_ssh: bool,
    skip_lfs: bool,
    skip_checkout: bool,
):
    while True:
        try:
            instance = queue.get_nowait()
        except Exception:
            break

        instance_id = instance.get("instance_id")
        try:
            logging.info("Cloning %s", instance_id)
            ensure_repo(instance, output_dir, rank, use_ssh, skip_lfs, skip_checkout)
            logging.info("Done %s", instance_id)
        except Exception as exc:
            logging.warning("Failed %s: %s", instance_id, exc)


def main():
    parser = argparse.ArgumentParser(description="Clone Loc-Bench repos without building indexes.")
    parser.add_argument("--dataset", type=str, default="czlll/Loc-Bench_V1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--instance_id_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="playground/locbench_repos")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--use_ssh", action="store_true")
    parser.add_argument("--use_https", action="store_true")
    parser.add_argument("--skip_lfs", action="store_true")
    parser.add_argument("--skip_checkout", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.use_https:
        use_ssh = False
    elif args.use_ssh:
        use_ssh = True
    else:
        use_ssh = True

    instances = load_instances(
        dataset=args.dataset,
        split=args.split,
        dataset_path=args.dataset_path,
        limit=args.limit,
        instance_id_path=args.instance_id_path,
    )

    manager = mp.Manager()
    queue = manager.Queue()
    for instance in instances:
        queue.put(instance)

    num_processes = max(1, args.num_processes)
    workers = []
    for rank in range(num_processes):
        proc = mp.Process(
            target=worker,
            args=(
                rank,
                queue,
                args.output_dir,
                use_ssh,
                args.skip_lfs,
                args.skip_checkout,
            ),
        )
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()


if __name__ == "__main__":
    main()
