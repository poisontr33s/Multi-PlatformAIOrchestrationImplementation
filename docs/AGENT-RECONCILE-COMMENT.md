Please reconcile this branch with the modernization plan in PR #4.

Acceptance criteria (must all be met):
- uv-based setup; `scripts/jules-setup.sh` completes on fresh VM (with ruff/black/pytest).
- Providers: Google (`google-genai`), Anthropic, OpenAI, Hugging Face.
- Node CLI bridge via `bunx` with `npx` fallback.
- Minimal FastAPI server at `src/ai_orchestration/api/server.py` and `openapi/openapi.yaml`.
- `docs/INTEGRATIONS.md` and `AGENTS.md` added and accurate.
- `tests/test_providers_import.py` and `tests/test_apply_file_blocks.py` pass.
- No hardcoded deprecated model IDs; env-driven configuration.

If you generate code in chat, emit file blocks like:
```python name=src/path/file.py
# contents
```
or
````markdown name=docs/FILE.md
# contents
````
We will apply them with `scripts/apply_file_blocks.py`.

Report back with:
- Which criteria you satisfied,
- Which files you created/modified (paths),
- Exact commands and exit codes for ruff, black, pytest.