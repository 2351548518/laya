# Tool & Function Selection

**English** · [简体中文](../docs_zh/tools.md)

`LayaToolSelector` scores candidate tools with a native decision checkpoint. The returned
`ToolRouteDecision` carries the selected tool, its option probabilities and fallback status.

## Installation

```bash
python -m pip install -e '.[multimodal]'
```

---

## 1. Single-Pass Tool Selection

Define the candidate tools and select one for the input.

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

Both synchronous `select()` and non-blocking asynchronous `aselect()` are supported.

---

## 2. Direct Execution (`call`)

`call()` selects the candidate tool and immediately executes it with the supplied arguments:

```python
result = selector.call("Calculate payments for 500k at 6.5% for 30 years",
                       principal=500000, rate=0.065, years=30)
print(result)  # "Monthly payment: $2,450"
```

If the tool is a LangChain tool with `.run()`, `call()` automatically invokes its execution method.

---

## 3. Direct Answer Detection (No Tool Needed)

In conversational agent pipelines, many user queries do not require external tool invocation (e.g. greetings, general conversational questions). Setting `allow_direct_answer=True` enables Laya to recognize when to answer conversationally without calling any tool:

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

---

## 4. Calibrated Confidence Gating & Fallbacks

`LayaToolSelector` uses Laya's calibrated confidence scores to gate routing decisions. When an ambiguous or out-of-domain query is presented, the selector avoids bad tool invocations by falling back to a general agent or raising `LayaLowConfidenceError`:

```python
from laya_multimodal.integrations import LayaLowConfidenceError

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

If `fallback_tool` is omitted, requests below threshold raise `LayaLowConfidenceError`:

```python
try:
    decision = selector.select("Unclear request")
except LayaLowConfidenceError as e:
    print(f"Abstained: confidence {e.confidence:.3f} below threshold {e.threshold}")
```

---

## 5. LangChain LCEL Integration

`LayaToolSelector` exports an LCEL-compatible Runnable via `.as_runnable()`, allowing it to be composed seamlessly into LangGraph nodes or LangChain pipelines:

```python
from laya_multimodal.integrations.tools import LayaToolSelector

selector = LayaToolSelector(tools=[search_web, query_database])

# Compose with pipe operator |
chain = selector.as_runnable() | (lambda decision: decision.tool_name)
tool_to_call = chain.invoke("Look up order #45920 in the database")
print(f"Dispatching to: {tool_to_call}")
```

---

## 6. Remote HTTP Mode (`laya-multimodal-serve`)

In serverless or microservice architectures, `LayaToolSelector` connects directly to a self-hosted `laya-multimodal-serve` cluster using standard library `urllib` (no external HTTP client required):

```python
selector = LayaToolSelector(
    tools=[search_web, query_database],
    base_url="http://laya-cluster.internal:8000",
    api_key="your-cluster-secret",
    model="multimodal",
)

decision = selector.select("Search for latest updates")
```

---

## 7. Supported Tool Formats

`LayaToolSelector` automatically parses candidate tools from all standard formats:
* **Python functions**: extracts `__name__` and first line of docstring `__doc__`.
* **LangChain & CrewAI tools**: reads `.name` and `.description`.
* **LlamaIndex tools**: reads `tool.metadata.name` and `tool.metadata.description`.
* **JSON Schema / OpenAI specs**: parses `{"type": "function", "function": {"name": ..., "description": ...}}`.
* **Dictionaries**: reads `{"name": ..., "description": ...}`.
* **Plain strings**: parses `"tool_name: tool description"`.
