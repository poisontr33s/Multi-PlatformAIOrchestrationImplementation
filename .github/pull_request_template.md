# Pull Request Checklist (Preflight)

This repository follows the modernization plan introduced in PR #4.

Before requesting review, confirm all that apply:

- [ ] This PR aligns with the uv + Bun dual-runtime approach (no global npm installs required).
- [ ] If touching providers, use: google-genai (Google), anthropic (Claude), openai (OpenAI), and optional Hugging Face. No deprecated model IDs are hardcoded.
- [ ] If adding runtime code, tests and linters pass locally:
      - `uv run ruff check .`
      - `uv run black --check .`
      - `uv run pytest -q`
- [ ] If adding/altering server surface, update openapi/openapi.yaml accordingly.
- [ ] If an AI agent produced code, file blocks are provided and can be applied via `python scripts/apply_file_blocks.py`.

Reference: PR #4 – Modernize orchestration with uv + Bun and multi-provider adapters.