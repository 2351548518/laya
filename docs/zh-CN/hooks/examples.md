# 预测钩子示例

[English](../../hooks/examples.md) · **简体中文**

这些片段按事件展示用法。先按本地推理指南构造 `agent` 或 `router`，并准备状态和题目。
日志、缓存、指标和业务复核函数由应用提供。
[原生推理](../multimodal.md#local-prediction) · [钩子 API](api.md)

<a name="quick-start"></a>

## 开始使用

为一次预测附上一个结束回调。

```python
import laya_multimodal

def log(ctx):
    print(ctx.model, ctx.results[0]["answers"])

agent = laya_multimodal.load( on_predict_end=log)
agent.system_one("I was charged twice.", {"urgent": {"type": "noul", "instructions": "Urgent?"}})
```

<a name="audit"></a>

## 审计

在结束事件记录语境 ID、检查点与结果。生产日志应按字段做脱敏。

```python
import json, sys
import laya_multimodal

def audit(ctx):
    for state, result in zip(ctx.states, ctx.results or []):
        record = {
            "run_id": ctx.run_id,
            "model": ctx.model,
            "state": state,
            "routing": result.get("routing"),
            "answers": result["answers"],
            "usage": result.get("usage"),
            "call_usage": ctx.usage,
            "call_elapsed_ms": round(ctx.elapsed_ms or 0.0, 3),
        }
        print(json.dumps(record), file=sys.stderr)
        # ship_to_service(record)

agent = laya_multimodal.load( on_predict_end=audit)
```

<a name="redact-pii"></a>

## 脱敏

在模型处理前替换 `ctx.states`，同时避免原地修改调用方持有的对象。
后续缓存和追踪钩子按列表顺序读取处理后的内容。

```python
import re
import laya_multimodal

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"\+?\d[\d ()-]{7,}\d")

def scrub(value):
    if isinstance(value, str):
        return PHONE.sub("[phone]", EMAIL.sub("[email]", value))
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value

def redact(ctx):
    ctx.states = [scrub(s) for s in ctx.states]

agent = laya_multimodal.load( on_predict_start=redact)
```

<a name="cache"></a>

## 缓存

命中时提供与输入数相同的结果，走 `ctx.skip`。缓存键覆盖状态、题目及其顺序、
检查点和 token 预算。实际部署还应随权重与提示版本失效。

```python
import hashlib, json
import laya_multimodal

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

agent = laya_multimodal.load( on_predict_start=read, on_predict_end=write)
first = agent.system_one("state", QUESTIONS)    # runs the model
second = agent.system_one("state", QUESTIONS)   # served from CACHE
```

<a name="metrics"></a>

## 指标

在结束位置查看耗时、用量与异常。需要汇总时，在回调外聚合记录。

```python
import laya_multimodal

COUNTS, LATENCIES = {}, []

def metrics(ctx):
    COUNTS[ctx.model] = COUNTS.get(ctx.model, 0) + 1
    if ctx.elapsed_ms is not None:
        LATENCIES.append(ctx.elapsed_ms)

agent = laya_multimodal.load( on_predict_end=metrics, hooks_raise=False)
```

<a name="guardrail"></a>

## 执行前的复核

需要中止当前请求时，在开始钩子抛出应用异常，调用方处理此异常。

```python
import laya_multimodal

class Blocked(Exception):
    pass

def guard(ctx):
    text = " ".join(str(state) for state in ctx.states).lower()
    if "ignore previous instructions" in text:
        raise Blocked("prompt injection")

agent = laya_multimodal.load( on_predict_start=guard)

try:
    agent.system_one("Ignore previous instructions and ...", QUESTIONS)
except Blocked:
    handle_block()
```

<a name="confidence-gate"></a>

## 置信度门槛

结束钩子可标注需要复核的结果。阈值来自部署数据的独立校准，对每个答案保留原始值。

```python
def gate(ctx):
    for result in ctx.results or []:
        answer = result["answers"].get("dept")
        if answer and answer["confidence"] < 0.6:
            answer["choice"] = "human-review"
            answer["gated"] = True

agent = laya_multimodal.load( on_predict_end=gate)
```

<a name="routing-pin"></a>

## 指定路由

在 `on_route` 更新检查点选择。传给 `predict` 的单次钩子只用于该调用。

```python
from laya_multimodal import Router
from laya_multimodal.router import RouteDecision

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

router = Router(models={"support": "models/support"}, hooks=[Pin()])
```

```python
router.predict("refund request", QUESTIONS, hooks=[Pin()])
```

<a name="lifecycle"></a>

## 加载与卸载

在 `on_load` 和 `on_evict` 观察已注册模型的常驻状态。

```python
from laya_multimodal import Router

class Lifecycle:
    def on_load(self, ctx):
        print("loaded", ctx.model, "agent", type(ctx.agent).__name__)

    def on_evict(self, ctx):
        print("evicted", ctx.model)

router = Router(max_loaded=1, hooks=[Lifecycle()])
router.preload(["multimodal"])   # on_load fires per build
router.unload()                               # on_evict fires per freed checkpoint
```

<a name="composition"></a>

## 组合钩子

进程默认先执行，实例钩子其次，单次钩子最后。每层按传入顺序执行。

```python
import laya_multimodal

class Metrics:
    def on_predict_end(self, ctx):
        record_latency(ctx.model, ctx.elapsed_ms)

def redact(ctx):
    ctx.states = [strip_pii(s) for s in ctx.states]

def audit(ctx):
    ship(ctx.run_id, ctx.results)

agent = laya_multimodal.load(
    hooks=[Metrics()],              # hooks=[...] entries run in list order
    on_predict_start=redact,        # convenience callables follow the entries
    on_predict_end=audit,           # both kinds share one ctx per call
    hooks_raise=True,
)
```

<a name="per-call-hooks"></a>

## 单次调用

为一条请求附加审计或调试，同时保留其他调用的实例设置。

```python
agent.system_one(
    state,
    questions,
    on_predict_end=lambda ctx: debug_dump(ctx),
    hooks_raise=False,
)

router.predict(
    state,
    questions,
    hooks=[Pin()],                  # applies to on_route too
    on_predict_end=audit,
)
```

<a name="batch"></a>

## 批量

Agent 的批量语境包含多个状态。对所有输入或输出迭代。

```python
def audit_batch(ctx):
    for state, result in zip(ctx.states, ctx.results):
        ship_one(ctx.run_id, state, result)

results = agent.predict_batch([state_a, state_b, state_c], questions, on_predict_end=audit_batch)
```

<a name="http-server"></a>

## HTTP 服务

用 `create_app` 注入已经配置钩子的 Router。HTTP 调用方传数据和预测控制。

```python
from laya_multimodal import Router
from laya_multimodal.serve import create_app

router = Router(hooks=[Metrics()], on_predict_end=audit, hooks_raise=False)
app = create_app(router=router)
```

<a name="runtime-registration"></a>

## 运行时注册

通过实例管理持久钩子或临时作用域。

```python
metrics = Metrics()
agent.add_hook(metrics)            # attach at runtime
agent.remove_hook(metrics)         # by identity: keep the instance to remove it

with agent.hooks_installed(DebugDump()):
    agent.system_one(state, questions)   # DebugDump only here
```

<a name="base-class-and-process-wide-defaults"></a>

## 基类与进程默认

继承 `BaseHook` 后只覆盖需要的事件。共享策略可在进程注册表配置。

```python
from laya_multimodal import BaseHook, hooks

class Audit(BaseHook):
    def on_predict_end(self, ctx):
        ship(ctx.run_id, ctx.results)

hooks.set_default_hooks(hooks=[Audit()])   # runs for every call in the process

# later, or in tests:
hooks.clear_default_hooks()
```

<a name="token-budget"></a>

## Token 预算

原生模型的 `max_len` 包括状态、题目与候选。

```python
agent.predict(state, questions, max_len=8192)
```

<a name="async-hooks"></a>

## 异步回调

钩子对象可实现协程事件，单事件参数也接受 async 函数。

```python
import laya_multimodal
from laya_multimodal import AsyncHook

class RemoteAudit:
    async def on_predict_end(self, ctx):
        await ship(ctx.run_id, ctx.results)

agent = laya_multimodal.load( hooks=[AsyncHook(RemoteAudit())])
```

```python
async def async_end(ctx):
    await ship(ctx.results)

agent.system_one(state, questions, on_predict_end=async_end)
```

<a name="hook-timeout"></a>

## 等待超时

可按实例或调用设置 `hooks_timeout`。底层任务也需要自己的时限。

```python
agent = laya_multimodal.load( on_predict_end=metrics, hooks_timeout=2.0)

# or per call
agent.system_one(state, questions, on_predict_end=metrics, hooks_timeout=0.5)
```

<a name="testing-hooks"></a>

## 测试钩子

以小输入收集触发顺序与语境，另外验证错误路径。

```python
seen = []
agent.predict_batch(["s0"], questions, on_predict_end=lambda ctx: seen.append(ctx.results))
assert len(seen) == 1
```

[错误与超时](errors.md) · [事件顺序](lifecycle.md) · [源码示例](https://github.com/2351548518/laya/tree/main/examples/hooks)
