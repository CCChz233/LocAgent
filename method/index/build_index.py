"""
Unified dense index builder with multiple block strategies.
- fixed: non-overlapping blocks of N lines (default 15)
- sliding: overlapping window (window_size, slice_size controls step)
- rl_fixed: RLCoder fixed_block (12 non-empty lines, max 5000 lines)
- rl_mini: RLCoder mini_block (empty-line segments, stitched to <=15 lines, max 5000 lines)

Outputs: embeddings.pt + metadata.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class Block:
    def __init__(self, file_path: str, start: int, end: int, content: str, block_type: str):
        self.file_path = file_path
        self.start = start
        self.end = end
        self.content = content
        self.block_type = block_type


def iter_files(repo_root: Path) -> List[Path]:
    return [
        p for p in repo_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".py", ".java", ".js", ".ts", ".go", ".rs", ".cpp", ".c", ".hpp", ".h"}
    ]


def blocks_fixed_lines(text: str, rel: str, block_size: int) -> List[Block]:
    lines = text.splitlines()
    blocks: List[Block] = []
    start = 0
    while start < len(lines):
        end = min(start + block_size, len(lines))
        chunk = lines[start:end]
        if any(l.strip() for l in chunk):
            blocks.append(Block(rel, start, end - 1, "\n".join(chunk), "fixed"))
        start = end
    return blocks


def blocks_sliding(text: str, rel: str, window_size: int, slice_size: int) -> List[Block]:
    lines = text.splitlines()
    delta = window_size // 2
    step = max(1, window_size // slice_size) if slice_size > 0 else window_size
    blocks: List[Block] = []
    for line_no in range(0, len(lines), step):
        start = max(0, line_no - delta)
        end = min(len(lines), line_no + window_size - delta)
        chunk = lines[start:end]
        if any(l.strip() for l in chunk):
            blocks.append(Block(rel, start, end - 1, "\n".join(chunk), "sliding"))
    return blocks


def blocks_rl_fixed(text: str, rel: str, max_lines: int = 5000) -> List[Block]:
    lines = [line for line in text.split("\n") if line.strip()]
    blocks: List[Block] = []
    for i in range(0, min(len(lines), max_lines), 12):
        start = i
        end = min(i + 12, len(lines))
        chunk = lines[start:end]
        blocks.append(Block(rel, start, end - 1, "\n".join(chunk), "rl_fixed"))
    return blocks


def blocks_rl_mini(text: str, rel: str, max_lines: int = 5000) -> List[Block]:
    mini_blocks = []
    cur = []
    for line in text.splitlines():
        if line.strip():
            cur.append(line)
        else:
            if cur:
                mini_blocks.append(cur)
                cur = []
    if cur:
        mini_blocks.append(cur)

    temp = []
    for mb in mini_blocks:
        if len(mb) > 15:
            for idx in range(0, len(mb), 15):
                temp.append(mb[idx: idx + 15])
        else:
            temp.append(mb)
    mini_blocks = temp

    blocks: List[Block] = []
    current = []
    total = 0
    for block in mini_blocks:
        if total >= max_lines:
            break
        if len(current) + len(block) <= 15:
            current.extend(block)
            total += len(block)
        else:
            if current:
                blocks.append(Block(rel, total - len(current) + 1, total, "\n".join(current), "rl_mini"))
            current = block
            total += len(block)
    if current:
        blocks.append(Block(rel, total - len(current) + 1, total, "\n".join(current), "rl_mini"))
    return blocks


def collect_blocks(repo_path: str, strategy: str, block_size: int, window_size: int, slice_size: int) -> List[Block]:
    repo_root = Path(repo_path)
    blocks: List[Block] = []
    for p in iter_files(repo_root):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="utf-8", errors="ignore")
        rel = str(p.relative_to(repo_root))
        if strategy == "fixed":
            blocks.extend(blocks_fixed_lines(text, rel, block_size))
        elif strategy == "sliding":
            blocks.extend(blocks_sliding(text, rel, window_size, slice_size))
        elif strategy == "rl_fixed":
            blocks.extend(blocks_rl_fixed(text, rel))
        elif strategy == "rl_mini":
            blocks.extend(blocks_rl_mini(text, rel))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    if not blocks:
        raise RuntimeError(f"No blocks produced under {repo_path}")
    return blocks


class BlockDataset(Dataset):
    def __init__(self, blocks: List[Block], tokenizer: AutoTokenizer, max_length: int):
        self.blocks = blocks
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx: int):
        b = self.blocks[idx]
        text = f"file path: {b.file_path}\nlines: {b.start}-{b.end}\n\n{b.content}"
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def embed_blocks(blocks: List[Block], model: AutoModel, tokenizer: AutoTokenizer, max_length: int, batch_size: int, device: torch.device) -> torch.Tensor:
    ds = BlockDataset(blocks, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    outs = []
    with torch.no_grad():
        for input_ids, attn_mask in tqdm(loader, desc="Embedding blocks"):
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


def save_index(output_dir: Path, embeddings: torch.Tensor, blocks: List[Block], strategy: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_dir / "embeddings.pt")
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for idx, b in enumerate(blocks):
            f.write(json.dumps({
                "block_id": idx,
                "file_path": b.file_path,
                "start_line": b.start,
                "end_line": b.end,
                "block_type": b.block_type,
                "strategy": strategy,
            }) + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Build dense index with multiple block strategies.")
    ap.add_argument("--repo_path", required=True, help="Repository root to index.")
    ap.add_argument("--output_dir", required=True, help="Where to save embeddings + metadata.")
    ap.add_argument("--model_name", default="nov3630/RLRetriever", help="HF model id or local path.")
    ap.add_argument("--strategy", choices=["fixed", "sliding", "rl_fixed", "rl_mini"], default="fixed")
    ap.add_argument("--block_size", type=int, default=15, help="Used for strategy=fixed.")
    ap.add_argument("--window_size", type=int, default=20, help="Used for strategy=sliding.")
    ap.add_argument("--slice_size", type=int, default=2, help="Used for strategy=sliding.")
    ap.add_argument("--max_length", type=int, default=512, help="Max tokens for encoding.")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for encoding.")
    ap.add_argument("--force_cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    blocks = collect_blocks(args.repo_path, args.strategy, args.block_size, args.window_size, args.slice_size)
    embeddings = embed_blocks(blocks, model, tokenizer, args.max_length, args.batch_size, device)
    save_index(Path(args.output_dir), embeddings, blocks, args.strategy)


if __name__ == "__main__":
    main()
