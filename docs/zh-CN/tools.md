# 工具与函数选择

[English](../tools.md) · **简体中文**

`LayaToolSelector` 用原生检查点评估候选工具。`ToolRouteDecision` 包含选中的工具、候选概率
及是否回退。应用先登记允许调用的工具，再使用结果。

<a name="installation"></a>

## 安装

```bash
python -m pip install -e '.[multimodal]'
```

<a name="1-single-pass-tool-selection"></a>

## 选择一个工具

使用 `select()` 或异步 `aselect()` 选择。候选的名称和说明进入决策提示。

```python
from laya_multimodal.integrations.tools import LayaToolSelector

def search_web(query: str):
    """Search Google or Bing for real-time news and websites."""
    return f"Search results for: {query}"

def calculate_mortgage(principal: float, rate: float, years: int):
    """Calculate monthly mortgage amortization payments and interest."""
    return "Monthly payment: $2,450"

def query_database(sql: str):
    """Execute read-only SQL queries against Postgres customer database."""
    return "Rows returned: 12"

tools = [search_web, calculate_mortgage, query_database]

# Initialize tool selector
selector = LayaToolSelector(tools=tools)

# Select the matching tool:
decision = selector.select("What is the current mortgage rate on a 30-year fixed loan?")
print(f"Selected Tool: {decision.tool_name}")
print(f"Confidence:    {decision.confidence:.3f}")
print(f"Probabilities: {decision.probabilities}")
```

<a name="2-direct-execution-call"></a>

## 调用所选工具

`call()` 会选择候选并用传入参数执行。对 LangChain 工具调用其 `.run()`。
把参数校验和业务授权放在调用边界。

```python
result = selector.call("Calculate payments for 500k at 6.5% for 30 years",
                       principal=500000, rate=0.065, years=30)
print(result)  # "Monthly payment: $2,450"
```

<a name="3-direct-answer-detection-no-tool-needed"></a>

## 允许直接回答

`allow_direct_answer=True` 把直接回复作为一个分支，应用可按选出的分支继续。

```python
selector = LayaToolSelector(
    tools=[search_web, query_database],
    allow_direct_answer=True,
)

decision = selector.select("Hello! How can you help me today?")
if decision.is_direct_answer:
    print("No tool needed: respond conversationally.")
else:
    print(f"Call tool: {decision.tool_name}")
```

<a name="4-calibrated-confidence-gating--fallbacks"></a>

## 低置信度与回退

用与部署相同的工具集合和校准数据确定 `confidence_threshold`。用 `fallback_tool`
指定回退工具，或在低置信度时处理 `LayaLowConfidenceError`。

```python
from laya_multimodal.integrations._errors import LayaLowConfidenceError

def general_fallback_llm(prompt: str):
    """Fallback handler for ambiguous or novel requests."""
    return "Dispatching to general reasoning LLM..."

selector = LayaToolSelector(
    tools=[search_web, calculate_mortgage],
    confidence_threshold=0.85,
    fallback_tool=general_fallback_llm,
)

decision = selector.select("Compose a haiku about autumn leaves.")
if decision.is_fallback:
    print(f"Ambiguous request ({decision.confidence:.2f} < 0.85). Routed to fallback handler.")
```

```python
try:
    decision = selector.select("Unclear request")
except LayaLowConfidenceError as e:
    print(f"Abstained: confidence {e.confidence:.3f} below threshold {e.threshold}")
```

<a name="5-langchain-lcel-integration"></a>

## LangChain LCEL

`as_runnable()` 将选择器接入 LangGraph 或 LCEL。

```python
from laya_multimodal.integrations.tools import LayaToolSelector

selector = LayaToolSelector(tools=[search_web, query_database])

# Compose with pipe operator |
chain = selector.as_runnable() | (lambda decision: decision.tool_name)
tool_to_call = chain.invoke("Look up order #45920 in the database")
print(f"Dispatching to: {tool_to_call}")
```

<a name="6-remote-http-mode-laya-multimodal-serve"></a>

## 自托管 HTTP

在 `base_url` 中提供 System One 服务地址。

```python
selector = LayaToolSelector(
    tools=[search_web, query_database],
    base_url="http://laya-cluster.internal:8000",
    api_key="your-cluster-secret",
    model="multimodal",
)

decision = selector.select("Search for latest updates")
```

<a name="7-supported-tool-formats"></a>

## 工具定义格式

| 对象 | 使用的字段 |
| --- | --- |
| Python 函数 | `__name__` 和文档字符串首行 |
| LangChain、CrewAI | `name` 与 `description` |
| LlamaIndex | `metadata.name` 与 `metadata.description` |
| 函数 schema | `{"type":"function","function":{"name":"...","description":"..."}}` |
| 字典 | `name` 与 `description` |
| 字符串 | `工具名: 说明` |

[工作流示例](use-cases.md) · [HTTP 契约](http-api.md)
