# Contributing

Thanks for helping improve laya-multimodal. This guide keeps reviews fast and the history clean.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Scope

Laya is a fast, local, on-device decision engine. Keep additions in that spirit: they should run
in the user's own process or on their own hardware, with no dependency on a hosted service or an
external API. A feature that only works against a hosted backend is out of scope for the core
package; it belongs in a separate integration or a community project.

Keep reusable multimodal code in `laya_multimodal/`, `scripts/multimodal/` and `configs/`.
Store source-specific preparation, intermediate outputs and experiment logs under `_local/`.

## Ways to contribute

- Report a bug with the bug report template.
- Request or discuss a feature with the feature request template.
- Improve the docs under [`docs/`](docs/index.md), including the [hooks guide](docs/hooks/index.md).
- Fix a bug or add a feature with a focused pull request.
- Share benchmarks, evaluations, or integration reports, which the project treats as first class.

## Development setup

Use Python 3.11 or newer for the native runtime. Install the extras for the code you are testing.

```bash
python -m pip install -e '.[multimodal,serve,mcp,langchain]' pytest httpx ruff
ruff check laya_multimodal/ --select=E9,F63,F7,F82,F401,F811 --line-length=120
python -m compileall -q laya_multimodal/ tests/
python scripts/test_suites.py
```

The shared suite exercises native CPU fixtures, the serving API and training resumes.
Run a real checkpoint with `verify/multimodal_smoke_test.py` when changing model loading,
media preprocessing or decision decoding.

## Documentation

Describe the command or API the reader should call and the checkpoint it needs. English pages
live under `docs/`, with corresponding Simplified Chinese pages under `docs_zh/`.
Update both versions when an example, interface or metric changes. Keep shared v6 images,
configuration and sample indexes in `docs/project-readme/assets/`.

Build and check the documentation before submitting changes.

```bash
python -m pip install -r requirements-docs.txt
python scripts/docs_site.py build
python tests/test_packaging.py
```

## Style

- Keep the public API stable. If a change must move it, update `tests/test_hooks_api.py` in the same
  pull request, because that suite is the API contract.
- Prefer the standard library and what is already a dependency over a new one.
- Leave one runnable check behind for non-trivial logic: an assert based script is enough.
- Comment only where the code cannot say it, usually a non-obvious reason or a hardware caveat.
- Match the surrounding style rather than a personal preference.

## Commits

Use conventional commit prefixes, matching the history:

```
feat(agent): ...
fix(router): ...
perf(common): ...
docs(hooks): ...
test(batch): ...
```

Keep one logical change per commit. A focused pull request is much easier to review and merge than a
large one, and it is fine to split a big change into several PRs.

## Pull requests

The template asks for four things; filling them in is what makes review quick:

1. **What** changed.
2. **Why**, ideally the concrete use case, not only the mechanism.
3. **How it was verified**: the exact commands you ran.
4. Any follow-ups you deliberately left out.

Before you open one:

- Rebase onto the latest `main`, so the diff is only your change.
- Keep it focused; split unrelated work into another PR.
- Update docs or examples when the public API changes.
- Do not commit secrets, tokens, or large binary files.

If your change moves numbers, report the before and after: the maintainer verifies decisions against
real checkpoints, and measured deltas (probability changes, latency, memory) are what gets a change
merged.

## Reporting issues

The project is maintained in English, so please write issues in English when you can; it is the
common language for everyone reading and triaging.

Use the issue templates and include:

- What you expected and what you got.
- A minimal reproduction, ideally a short script.
- Your Laya version (`python -c "import laya_multimodal; print(laya_multimodal.__version__)"`), Python version, OS, and
  device (CPU, CUDA or MPS).

## Reviews

Reviewers may ask you to rebase, to split a change, or to add a regression test. Those are the
normal asks in this repository, not a rejection. If you disagree with a request, say so on the PR
and explain the tradeoff.

## License

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
