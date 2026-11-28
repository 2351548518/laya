# 钩子使用模式

[English](../../docs/hooks/patterns.md) · **简体中文**

输入改写在开始事件完成，审计和指标在结束事件读取结果，路由改写在 `on_route` 中进行。
实际动作由应用执行。对于共享的钩子对象，明确其数据范围、并发方式和错误策略。
[生命周期](lifecycle.md) · [错误处理](errors.md)

<a name="patterns"></a>

## 推荐模式

<a name="audit-log"></a>

### 审计日志

按 `run_id` 记录所选检查点、结果、耗时和错误。批量调用遍历所有结果。

```python
import json, sys

def audit(ctx):
    for state, result in zip(ctx.states, ctx.results or []):
        json.dump({
            "run_id": ctx.run_id,
            "model": ctx.model,
            "state": state,
            "routing": result.get("routing"),
            "answers": result["answers"],
            "usage": result.get("usage"),
            "call_usage": ctx.usage,
            "call_elapsed_ms": round(ctx.elapsed_ms or 0.0, 3),
        }, sys.stdout)
        sys.stdout.write("\n")

laya_multimodal.load( on_predict_end=audit)
```

<a name="pii-redaction"></a>

### 脱敏

开始事件替换含敏感信息的状态，保护推理上下文以及后续钩子读取的数据。

```python
import re
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

def redact(ctx):
    ctx.states = [
        EMAIL.sub("[email]", s) if isinstance(s, str) else s
        for s in ctx.states
    ]

laya_multimodal.load( on_predict_start=redact)
```

<a name="caching"></a>

### 缓存

缓存键包含顺序敏感的问题、状态、模型与预算。整批都有结果时再调用 `skip`。
缓存长期保存时，还应纳入检查点版本及媒体身份。

```python
import hashlib, json

CACHE = {}

def key(ctx, index):
    # Not sort_keys=True: criteria order is positional, so two orders are two questions,
    # and the checkpoint and token budget change the answer too.
    payload = json.dumps([ctx.states[index], ctx.questions, ctx.model,
                          ctx.max_len, ctx.head_max_len], default=str)
    return hashlib.sha256(payload.encode()).hexdigest()

def read(ctx):
    hits = [CACHE.get(key(ctx, i)) for i in range(len(ctx.states))]
    if all(hit is not None for hit in hits):
        ctx.skip(hits)   # one per state: skip replaces the whole call

def write(ctx):
    for i, result in enumerate(ctx.results or []):
        CACHE[key(ctx, i)] = result

laya_multimodal.load( on_predict_start=read, on_predict_end=write)
```

<a name="metrics"></a>

### 指标

在结束事件读取实际耗时和用量。独立统计调用量和失败数。

```python
COUNTS, LATENCIES = {}, []

def metrics(ctx):
    COUNTS[ctx.model] = COUNTS.get(ctx.model, 0) + 1
    if ctx.elapsed_ms is not None:
        LATENCIES.append(ctx.elapsed_ms)

laya_multimodal.load( on_predict_end=metrics, hooks_raise=False)
```

<a name="guardrails"></a>

### 执行条件

在开始事件检查业务前置条件。严格模式下异常会传给调用方。

```python
class Blocked(Exception):
    pass

def guard(ctx):
    if any("ssn" in str(state).lower() for state in ctx.states):
        raise Blocked("possible PII in state")

laya_multimodal.load( on_predict_start=guard)
```

<a name="confidence-gating"></a>

### 置信度标记

读取按部署分布校准的答案概率，标注需要复核的结果。

```python
def gate(ctx):
    for result in ctx.results or []:
        answer = result["answers"].get("dept")
        if answer and answer["confidence"] < 0.6:
            answer["choice"] = "human-review"
            answer["gated"] = True

laya_multimodal.load( on_predict_end=gate)
```

<a name="routing-override"></a>

### 自定义路由

在 `on_route` 设置已注册检查点，随后 Router 解析名称及加载权重。

```python
from laya_multimodal.decisions.routing import RouteDecision

class Pin:
    def on_route(self, ctx):
        if "refund" in str(ctx.states[0]).lower():
            ctx.decision = RouteDecision(
                model="support",
                repo="models/support",
                reason="refund workflow",
                detection=None,
                workflow=None,
            )

Router(models={"support": "models/support"}, hooks=[Pin()])
```

<a name="model-lifecycle"></a>

### 模型状态

记录实际发生的模型加载及卸载。

```python
class Lifecycle:
    def on_load(self, ctx):
        print("loaded", ctx.model)

    def on_evict(self, ctx):
        print("evicted", ctx.model)

Router(hooks=[Lifecycle()])
```

<a name="multi-tenant-context"></a>

### 多租户上下文

用闭包或按请求安装的对象保存租户信息，避免跨租户复用可变状态。

```python
def make_audit(tenant):
    def audit(ctx):
        ship(tenant, ctx.run_id, ctx.results)
    return audit

agent = laya_multimodal.load( on_predict_end=make_audit("acme"))
```

<a name="composition"></a>

### 组合

按依赖顺序注册回调。进程默认先于实例钩子，实例先于单次调用。例如先脱敏，再记录审计日志。

```python
agent = laya_multimodal.load(
    hooks=[Metrics(), Guardrail()],     # metrics first, then policy
    on_predict_start=redact,            # convenience callables appended after hooks
    hooks_raise=True,                   # policy failures are fatal
)
```

<a name="scoped-instrumentation"></a>

### 临时观察

用作用域限定调试钩子，退出块时撤下自己安装的回调。

```python
with agent.hooks_installed(DebugDump()):
    agent.system_one(state, questions)   # DebugDump only here
```

<a name="process-wide-instrumentation"></a>

### 进程级观察

需要统一覆盖多个 Agent 或 Router 时，通过进程默认钩子注册。测试结束后清理注册表。

```python
from laya_multimodal import BaseHook
from laya_multimodal.decisions import hooks

class Metrics(BaseHook):
    def on_predict_end(self, ctx):
        record(ctx.model, ctx.elapsed_ms)

hooks.set_default_hooks(hooks=[Metrics()])
```

<a name="token-budget-shaping"></a>

### 调整输入预算

开始阶段可设置 `ctx.max_len`。测试不同预算下处理后的输入长度、资源占用和质量。

```python
def widen_for_many_options(ctx):
    option_count = max(
        (len(q.get("criteria") or ()) for q in ctx.questions.values() if q.get("type") == "choice"),
        default=0,
    )
    if option_count >= 50:
        ctx.max_len = max(ctx.agent.max_len, ctx.max_len or 0, 8192)

agent = laya_multimodal.load(on_predict_start=widen_for_many_options)
```

<a name="anti-patterns"></a>

## 常见错误

<a name="blocking-work"></a>

### 在回调中阻塞

耗时的 I/O 占用当前预测调用。用有界队列把日志或导出交给独立工作者。

```python
# bad: blocks the whole server
def audit(ctx):
    requests.post("https://slow.example/decisions", json=..., timeout=30)

# better: enqueue, let a background worker ship it
def audit(ctx):
    QUEUE.put_nowait(record(ctx))
```

<a name="raising-from-end-hooks-for-control-flow"></a>

### 把结束异常当分支

希望调用者检查结果时，标注意图并交给应用逻辑处理。严格模式的结束异常会使成功的前向返回错误。

<a name="shared-mutable-state-without-a-lock"></a>

### 无锁修改共享状态

跨请求可见的缓存或计数由锁或线程安全容器保护。

```python
# bad
class Count:
    def __init__(self): self.n = 0
    def on_predict_end(self, ctx): self.n += 1

# good
import threading
class Count:
    def __init__(self):
        self.n = 0
        self._lock = threading.Lock()
    def on_predict_end(self, ctx):
        with self._lock:
            self.n += 1
```

<a name="silent-failure"></a>

### 吞掉失败

记录触发异常及其具体回调。关键业务策略由调用方明确决定失败处理。

```python
# bad: no one will ever know the audit trail stopped
def audit(ctx):
    try:
        ship(record(ctx))
    except Exception:
        pass
```

<a name="retaining-contexts"></a>

### 长期保留整个语境

只存需要的标量或摘要，避免让请求状态和结果持续占用内存。

```python
# bad: unbounded memory growth
SEEN = []
def audit(ctx):
    SEEN.append(ctx)

# good: keep only what you need
SEEN = []
def audit(ctx):
    SEEN.append((ctx.run_id, ctx.model, ctx.elapsed_ms))
```

<a name="redacting-too-late"></a>

### 太晚脱敏

在模型处理前修改输入。仅在结束修改响应时，模型和之前的钩子已经看到原输入。

<a name="per-question-logic-in-a-per-call-hook"></a>

### 把调用范围当作题目范围

一个调用可有多个状态和多个题目，按所需层次迭代。

```python
def flag(ctx):
    for result in ctx.results or []:
        for qid, answer in result["answers"].items():
            if answer.get("confidence", 1.0) < 0.5:
                alert(qid, ctx.run_id)
```

<a name="recursive-predict"></a>

### 递归预测

嵌套调用可再次触发同一回调。使用专用 Agent 或显式保护，批量时覆盖所有状态。

```python
# bad
def enrich(ctx):
    ctx.results = [agent.predict(ctx.states[0], EXTRA_QUESTIONS)]

# good: guard, or use a separate agent with no hooks
def enrich(ctx):
    if getattr(ctx, "_enriched", False):
        return
    ctx._enriched = True
    ctx.results = [enricher.predict(state, EXTRA_QUESTIONS) for state in ctx.states]
```

<a name="plain-callables-in-hooks"></a>

### 传错函数参数

单事件函数传入 `on_predict_start` 或 `on_predict_end`，`hooks` 接收事件对象。

```python
# bad: TypeError
laya_multimodal.load( hooks=[lambda ctx: None])

# good
laya_multimodal.load( on_predict_end=lambda ctx: None)
```

<a name="assuming-results-exist-in-end-hooks"></a>

### 假定结束事件一定有结果

读取前检查 `ctx.error` 与 `ctx.results`。成功的空批次也要独立处理。

```python
def audit(ctx):
    if ctx.results is None:
        log_failure(ctx.run_id, ctx.error)
        return
    log_success(ctx.run_id, ctx.results)
```

<a name="order-dependent-hooks"></a>

### 隐含的顺序依赖

写明修改输入、读写缓存、复核和日志的依赖，在测试中固定触发顺序。
[钩子示例](examples.md) · [请求追踪](tracing.md)
