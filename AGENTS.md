# Repository Guidelines

## Project Structure & Module Organization
Key entrypoints live at the repo root: `auto_search_main.py` (localization pipeline), `build_bm25_index.py` (BM25 indexing), and `sft_train.py` (training). Core modules are organized by responsibility: `dependency_graph/` builds and traverses code graphs, `repo_index/` manages repository indexing and workspace abstractions, `plugins/` hosts tool integrations, and `util/` contains shared helpers and prompts. Supporting assets live in `assets/`, while evaluation artifacts are in `evaluation/` (metrics and `run_evaluation.ipynb`). Shell helpers are in `scripts/`.

## Build, Test, and Development Commands
Install dependencies with:
```
pip install -r requirements.txt
```
Generate graph and BM25 indexes:
```
bash scripts/gen_graph_index.sh
bash scripts/gen_bm25_index.sh
```
Run localization end-to-end (expects env vars and index paths):
```
bash scripts/run.sh
```
For ad-hoc runs, call the entrypoint directly:
```
python auto_search_main.py --dataset ... --localize --merge ...
```

## Coding Style & Naming Conventions
This is a Python-first codebase. Use 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and lowercase module filenames. No formatter config is present, so keep changes consistent with nearby style; add short docstrings when behavior is non-obvious.

## Testing Guidelines
No dedicated test framework or coverage gates are configured. Validation is typically done via `evaluation/eval_metric.py` and `evaluation/run_evaluation.ipynb`. If you add automated tests, document the command and location (e.g., a new `tests/` directory) in this file.

## Commit & Pull Request Guidelines
Recent commits favor short, imperative, lowercase messages (e.g., “update readme”, “refactor dependency_graph”). Keep commits focused and descriptive. Pull requests should explain the dataset and index paths used, include the exact commands run, link relevant issues, and attach evaluation metrics or sample outputs when localization or indexing behavior changes.

## Configuration & Secrets
`scripts/run.sh` expects `OPENAI_API_KEY` and `OPENAI_API_BASE`. Set `GRAPH_INDEX_DIR` and `BM25_INDEX_DIR` to point at generated indexes before running. Avoid committing credentials or local paths.
