# CLI 与 MCP

[English](../cli-mcp.md) · **简体中文**

将 `LAYA_MULTIMODAL_MODEL` 指向训练好的原生检查点目录，CLI 和 MCP 都可以加载它。
项目的安装脚本默认下载合并后的 Qwen3.5 模型。
[本机安装](https://github.com/2351548518/laya/blob/main/LOCAL_SETUP.md)

```bash
laya-multimodal --preset triage --json "I was charged twice."
laya-multimodal --json --explain "请帮我退款"
laya-multimodal --multimodal models/jev-multimodal-qwen3.5-4b --help
laya-multimodal train --help
laya-multimodal eval --help
```

用 `--questions` 读取 JSON 题目定义，用 `--image`、`--audio` 或 `--video` 传入本地文件。
用 `--max-len` 设置单个请求的 token 预算。
[原生输入](multimodal.md) · [微调](finetune.md#run-the-joint-trainer) · [评测](evals.md)

<a name="mcp-server"></a>

## MCP 服务

为本地推理安装 `.[multimodal,mcp]`，在 MCP 客户端中以 stdio 方式启动
`laya-multimodal-mcp-server`。通过该进程的环境变量选择检查点与设备。

```bash
LAYA_MULTIMODAL_MODEL=/path/to/checkpoint LAYA_DEVICE=cpu laya-multimodal-mcp-server
```

如果已经启动 HTTP 推理服务，在 MCP 进程中设置 `LAYA_BASE_URL=http://127.0.0.1:8000`
即可转发请求。服务启用 Bearer 认证时，同时配置 `LAYA_API_KEY`。模型权重由 HTTP 服务持有。
[HTTP API](http-api.md)

`LAYA_EXTRA_MODELS` 使用与 HTTP 服务相同的检查点注册格式。`LAYA_PRELOAD=0`
将加载时机延迟到第一次预测。原生请求以 `max_len` 控制 token 预算。`head_max_len`
用于支持独立候选项预算的后端。

| 工具 | 返回 | 参数 |
| --- | --- | --- |
| `laya_status` | 已加载检查点与运行设备 | 无 |
| `laya_route` | 检查点路由结果 | `state`、`questions`、`model`、`task`、`lang`、`lang_guess`、`explain` |
| `laya_predict` | 类型化决策 | `state`、`questions`、`model`、`task`、`lang`、`lang_guess`、`max_len`、`head_max_len`、`min_confidence` |
| `laya_predict_batch` | 按请求顺序返回的答案 | `requests` 中的 `{state, questions, model?, task?, lang?, lang_guess?, max_len?, head_max_len?}`，以及 `batch_size`、`hooks_timeout`、`min_confidence`、`sort_by_length` |
| `laya_route_batch` | 按请求顺序返回的路由 | `requests` 中的 `{state, questions, model?, task?, lang?, lang_guess?}`，以及 `hooks_timeout`、`explain` |
| `laya_shortlist` | 缩小候选集合后的决策 | `state`、`questions`、`model`、`k`、`task`、`lang`、`lang_guess`、`max_len`、`head_max_len`、`min_confidence` |
| `laya_tournament` | 分组淘汰后的决策 | `state`、`questions`、`model`、`group_size`、`task`、`lang`、`lang_guess`、`max_len`、`head_max_len`、`min_confidence` |
| `laya_preset` | 用内置题目集所得的答案 | `preset`、`state`、`task`、`lang`、`lang_guess`、`max_len`、`head_max_len`、`min_confidence` |
| `laya_decide` | 按决策 schema 返回的值 | `state`、`schema`、`model`、`min_confidence` |
| `laya_predict_long` | 兼容后端上的滑窗决策 | `state`、`questions`、`model`、`window`、`stride`、`task`、`lang` |

候选缩减结果包含保留项标签 `labels`、筛选分数 `scores`、保留数 `k`、原始选项数
`n` 和是否保留全部选项的 `passthrough`。
