# LlamaIndex 集成

[English](../docs/llamaindex.md) · **简体中文**

`LayaSingleSelector` 选择一个查询引擎，`LayaMultiSelector` 可选择多个，
`LayaQueryRouter` 可直接向注册的引擎或函数分派。使用本地原生模型或自托管 HTTP 服务。

<a name="installation"></a>

## 安装

```bash
python -m pip install -e '.[multimodal,llamaindex]'
```

<a name="1-single-choice-routing-with-routerqueryengine"></a>

## RouterQueryEngine 单选

用 `LayaSingleSelector` 对注册的查询引擎评分。

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from laya_multimodal.integrations.llamaindex import LayaSingleSelector

# Define query engine tools
docs_tool = QueryEngineTool(
    query_engine=vector_index.as_query_engine(),
    metadata=ToolMetadata(
        name="vector_documentation",
        description="Semantic search over technical user documentation and API guides.",
    ),
)
sql_tool = QueryEngineTool(
    query_engine=sql_index.as_query_engine(),
    metadata=ToolMetadata(
        name="sql_database",
        description="Structured SQL database containing customer accounts, billing, and orders.",
    ),
)

# Initialize Laya selector with confidence fallback
selector = LayaSingleSelector(
    confidence_threshold=0.80,   # If confidence < 0.80, fall back to index 0
    fallback_index=0,
)

router_engine = RouterQueryEngine(
    selector=selector,
    query_engine_tools=[docs_tool, sql_tool],
)

response = router_engine.query("What is the shipping address for order #4912?")
print(response)
```

<a name="2-multi-choice-selection-for-composite-queries"></a>

## 为组合查询多选

跨多个索引的查询可用 `LayaMultiSelector` 选出相关工具。

```python
from laya_multimodal.integrations.llamaindex import LayaMultiSelector

multi_selector = LayaMultiSelector(
    probability_threshold=0.25,  # Select all tools with probability >= 0.25
    max_outputs=2,
)

tools = [docs_tool.metadata, sql_tool.metadata, summary_tool.metadata]
result = multi_selector.select(
    tools,
    "How does the database security policy compare with our published compliance guide?"
)

for sel in result.selections:
    print(f"Tool: {tools[sel.index].name} | {sel.reason}")
```

<a name="3-direct-query-dispatch-with-layaqueryrouter"></a>

## 直接分派查询

用 `LayaQueryRouter` 从字典注册的引擎中分派。`query()` 同步执行，`aquery()` 异步执行。

```python
from laya_multimodal.integrations.llamaindex import LayaQueryRouter

router = LayaQueryRouter(
    query_engines={
        "vector": vector_query_engine,
        "sql": sql_query_engine,
        "summary": summary_query_engine,
    },
    descriptions={
        "vector": "Semantic search over product documentation and guides",
        "sql": "Structured SQL queries for user accounts and transactions",
        "summary": "Quarterly reports and high-level business summaries",
    },
    confidence_threshold=0.75,
    fallback_key="vector",
)

# Route and execute in one call:
response = router.query("How many active subscriptions were renewed in Q3?")
print(response)
```

<a name="4-confidence-threshold-gating"></a>

## 置信度与回退

`LayaSingleSelector` 和 `LayaQueryRouter` 使用 `answer_confidence`。通过
`fallback_index` 或 `fallback_key` 选择回退引擎。`LayaSingleSelector` 可配置
`raise_on_low_confidence=True`。阈值按实际问题和候选数校准。
[原生校准](multimodal.md#calibration-and-evaluation)

<a name="5-remote-http-deployments"></a>

## HTTP 部署

```python
from laya_multimodal.integrations.llamaindex import LayaSingleSelector

selector = LayaSingleSelector(
    base_url="http://laya-multimodal-serve.internal:8080",
    confidence_threshold=0.85,
    fallback_index=0,
)
```

`base_url` 指向本机或内网的 System One 服务。 [服务设置](http-api.md)

<a name="6-per-call-decision-controls"></a>

## 预测控制

每个组件可配置 `max_len`、`lang`、`min_confidence`、`task`、`lang_guess`。
装在执行进程的钩子通过 `hooks`、`on_predict_start`、`on_predict_end`、
`hooks_raise` 和 `hooks_timeout` 设置。`max_len` 包含状态、指令和候选。

<a name="language-and-abstention"></a>

### 语言和暂缓标记

`lang` 提供语言控制。`min_confidence` 在 `last_decision` 中附上 `low_confidence`、
`abstention` 与 `abstention_threshold`，原选择仍可读取。`confidence_threshold`
连同回退选项决定实际选择。未提供的控制沿用后端配置。

<a name="which-checkpoint-answers"></a>

### 选择检查点

用 `model` 或 `task` 从 Router 的已注册检查点中选择。直接使用 `Agent` 时固定为该 Agent
加载的检查点。远程调用使用字符串语言提示。 [检查点路由](routing.md)
