# AGENTS.md

Context for AI coding assistants (Claude Code, Codex, Cursor, Copilot, Gemini CLI) working in this repository.

**laya-multimodal** adds native text, image, audio and video decisions plus single-box grounding to Laya. These rules mirror [CONTRIBUTING.md](CONTRIBUTING.md); the assistant-facing summary lives here because coding agents read this file automatically.

## Do NOT

- Introduce a dependency on a hosted service or external API. Features must run in the user's own process or on their own hardware; a feature that only works against a hosted backend belongs in a separate integration, not here.
- Change the public API without updating `tests/test_hooks_api.py` in the same pull request — that suite is the API contract.
- Skip the CI gates before declaring a change done:

  ```bash
  ruff check laya_multimodal/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
  python -m compileall -q laya_multimodal/ tests/
  ```

- Reformat files wholesale, reorder imports, or "modernize" surrounding code. Match the style of the file being edited; a diff full of formatting noise gets a change rejected.
- Commit secrets, tokens, or large binary files.
- Open pull requests in a language other than English. The project is triaged in English.
- Invent conventions. Use conventional commit prefixes matching the history (`feat(agent):`, `fix(router):`, `perf(common):`, `docs(hooks):`, `test(batch):`), and keep one logical change per commit.

## Where to look

| Editing | Read | Check |
|---|---|---|
| `laya_multimodal/` core | — | plain script suites: `python tests/test_router.py`, `tests/test_criteria.py`, `tests/test_hooks.py`, `tests/test_hooks_api.py` |
| native model or training | [Native multimodal guide](docs/multimodal.md) | `python -m pytest -q tests/test_multimodal.py tests/test_grounding.py tests/test_multimodal_cli.py tests/test_multimodal_export.py` |
| server path | `tests/test_serve.py` | `python -m pytest tests/test_serve.py` (skips when the `serve` extra is missing) |
| docs, docstrings in `laya_multimodal/` | `docs/` and `docs_zh/`, with `.nav.yml` for page order | `pip install -r requirements-docs.txt`, then `python scripts/docs_site.py build` with no `griffe:` lines in the output |

Optional extras are declared in `pyproject.toml` (`multimodal`, `serve`, `swanlab`, `vega`, `langchain`, `langgraph`); install only what the change needs.

## Pull requests

- Rebase onto the latest `main` so the diff is only your change.
- Keep it focused; split unrelated work into another PR.
- If a change moves numbers, report the before and after. The maintainer verifies decisions against real checkpoints, and measured deltas (probability changes, latency, memory) are what gets a change merged.
- Leave one runnable check behind for non-trivial logic; an assert-based script is enough.
