# PR Preface (Read Before Reviewing)

This repository uses a dual-runtime approach (Python via `uv`, JS/TS CLIs via `bunx`) and a provider abstraction (Google via `google-genai`, Anthropic, OpenAI, and optional Hugging Face). The authoritative plan is defined in [PR #4](https://github.com/poisontr33s/Multi-PlatformAIOrchestrationImplementation/pull/4).

Acceptance criteria at a glance:
- `uv` based environment and fast setup (see `scripts/jules-setup.sh` and `scripts/dev-setup.sh`).
- Providers implemented using current SDKs with environment-driven model selection (no hardcoded deprecated IDs).
- Node CLI bridge prefers `bunx` with `npx` fallback.
- Minimal FastAPI server exposed (if included), with OpenAPI kept in sync.
- Tests and linters pass without API keys for import-only tests.

Agent alignment:
- If an agent authored changes, it must emit file blocks like:
  ```python name=src/path/file.py
  # contents
  ```
  or
  ````markdown name=docs/FILE.md
  # contents
  ````
  Apply blocks using: `python scripts/apply_file_blocks.py < plan.md`.

Local verification commands:
- Lint: `uv run ruff check .`
- Format check: `uv run black --check .`
- Tests: `uv run pytest -q`
- Run server: `uv run uvicorn ai_orchestration.api.server:app --reload --port 8000`

For deviations, request the author/agent to reconcile to PR #4's acceptance criteria before substantive review.