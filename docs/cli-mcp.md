# CLI and MCP

Set `LAYA_MULTIMODAL_MODEL` to a trained native checkpoint directory to use it with both
the CLI and MCP. The setup script installs the merged Qwen3.5 checkpoint by default.
[Local setup](https://github.com/2351548518/laya/blob/main/LOCAL_SETUP.md)

```bash
laya --preset triage --json "I was charged twice."
laya --preset triage --route-only "请帮我退款"
laya --multimodal models/jev-multimodal-qwen3.5-4b --help
laya train --help
laya eval --help
```

Pass `--questions` with a JSON question definition, and use `--image`, `--audio` or `--video`
for local media. Use `--max-len` to set a request token budget.
[Native inputs](multimodal.md) · [Fine-tuning](finetune.md#fine-tune-with-laya-train) · [Evaluation](evals.md)

## MCP Server

Install `.[multimodal,mcp]` for local inference, then launch `laya-mcp-server` as a stdio
server from your MCP client. Choose the checkpoint and device in the process environment.

```bash
LAYA_MULTIMODAL_MODEL=/path/to/checkpoint LAYA_DEVICE=cpu laya-mcp-server
```

To share a running inference process, set `LAYA_BASE_URL=http://127.0.0.1:8000` in the MCP process.
Set `LAYA_API_KEY` there if the service requires bearer authentication. The model stays in the
service process. [HTTP API](http-api.md)

`LAYA_EXTRA_MODELS` accepts the same checkpoint registry as the HTTP service. Set
`LAYA_PRELOAD=0` to load on the first prediction. For native requests use `max_len`
for the token budget. A separate `head_max_len` requires an attached backend that supports it.

| Tool | Result | Parameters |
| --- | --- | --- |
| `laya_status` | Resident checkpoints and device | None |
| `laya_route` | Checkpoint selection | `state`, `questions`, `model`, `task`, `lang`, `lang_guess`, `explain` |
| `laya_predict` | Typed decisions | `state`, `questions`, `model`, `task`, `lang`, `lang_guess`, `max_len`, `head_max_len`, `min_confidence` |
| `laya_predict_batch` | Decisions in request order | `requests` containing `{state, questions, model?, task?, lang?, lang_guess?, max_len?, head_max_len?}`, `batch_size`, `hooks_timeout`, `min_confidence`, `sort_by_length` |
| `laya_route_batch` | Routing in request order | `requests` containing `{state, questions, model?, task?, lang?, lang_guess?}`, `hooks_timeout`, `explain` |
| `laya_shortlist` | Typed decisions after filtering choice options | `state`, `questions`, `model`, `k`, `task`, `lang`, `lang_guess`, `max_len`, `head_max_len`, `min_confidence` |
| `laya_tournament` | Typed decisions after grouped elimination | `state`, `questions`, `model`, `group_size`, `task`, `lang`, `lang_guess`, `max_len`, `head_max_len`, `min_confidence` |
| `laya_preset` | Decisions using a built-in question set | `preset`, `state`, `task`, `lang`, `lang_guess`, `max_len`, `head_max_len`, `min_confidence` |
| `laya_decide` | Values using a decision schema | `state`, `schema`, `model`, `min_confidence` |
| `laya_predict_long` | Windowed decisions with a compatible backend | `state`, `questions`, `model`, `window`, `stride`, `task`, `lang` |

A shortlist includes `labels` for the retained options, `scores` for their selection scores,
`k` for the number retained, `n` for the original option count and `passthrough` when the
choice was passed through without narrowing.
