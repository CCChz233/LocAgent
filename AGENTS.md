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

## Skills
These skills are discovered at startup from multiple local sources. Each entry includes a name, description, and file path so you can open the source for full instructions.
- skill-creator: Guide for creating effective skills. This skill should be used when users want to create a new skill (or update an existing skill) that extends Codex's capabilities with specialized knowledge, workflows, or tool integrations. (file: /Users/chz/.codex/skills/.system/skill-creator/SKILL.md)
- skill-installer: Install Codex skills into $CODEX_HOME/skills from a curated list or a GitHub repo path. Use when a user asks to list installable skills, install a curated skill, or install a skill from another repo (including private repos). (file: /Users/chz/.codex/skills/.system/skill-installer/SKILL.md)
- Discovery: Available skills are listed in project docs and may also appear in a runtime "## Skills" section (name + description + file path). These are the sources of truth; skill bodies live on disk at the listed paths.
- Trigger rules: If the user names a skill (with $SkillName or plain text) OR the task clearly matches a skill's description, you must use that skill for that turn. Multiple mentions mean use them all. Do not carry skills across turns unless re-mentioned.
- Missing/blocked: If a named skill isn't in the list or the path can't be read, say so briefly and continue with the best fallback.
- How to use a skill (progressive disclosure):
  1) After deciding to use a skill, open its SKILL.md. Read only enough to follow the workflow.
  2) If SKILL.md points to extra folders such as references/, load only the specific files needed for the request; don't bulk-load everything.
  3) If scripts/ exist, prefer running or patching them instead of retyping large code blocks.
  4) If assets/ or templates exist, reuse them instead of recreating from scratch.
- Description as trigger: The YAML description in SKILL.md is the primary trigger signal; rely on it to decide applicability. If unsure, ask a brief clarification before proceeding.
- Coordination and sequencing:
  - If multiple skills apply, choose the minimal set that covers the request and state the order you'll use them.
  - Announce which skill(s) you're using and why (one short line). If you skip an obvious skill, say why.
- Context hygiene:
  - Keep context small: summarize long sections instead of pasting them; only load extra files when needed.
  - Avoid deeply nested references; prefer one-hop files explicitly linked from SKILL.md.
  - When variants exist (frameworks, providers, domains), pick only the relevant reference file(s) and note that choice.
- Safety and fallback: If a skill can't be applied cleanly (missing files, unclear instructions), state the issue, pick the next-best approach, and continue.
