# Examples

**English** · [简体中文](../../docs_zh/hooks/examples.md)

Copy-paste recipes. Every snippet is self-contained apart from the helpers it names
(`ship`, `CACHE`, and so on), which you supply.

- [Quick start](#quick-start)
- [Audit](#audit)
- [Redact PII](#redact-pii)
- [Cache](#cache)
- [Metrics](#metrics)
- [Guardrail](#guardrail)
- [Confidence gate](#confidence-gate)
- [Routing pin](#routing-pin)
- [Lifecycle](#lifecycle)
- [Composition](#composition)
- [Per-call hooks](#per-call-hooks)
- [Batch](#batch)
- [HTTP server](#http-server)
- [Runtime registration](#runtime-registration)
- [Base class and process-wide defaults](#base-class-and-process-wide-defaults)
- [Async hooks](#async-hooks)
- [Hook timeout](#hook-timeout)
- [Token budget](#token-budget)
- [Testing hooks](#testing-hooks)

## Quick start

```python
import laya_multimodal

def log(ctx):
    print(ctx.model, ctx.results[0]["answers"])

agent = laya_multimodal.load( on_predict_end=log)
agent.system_one("I was charged twice.", {"urgent": {"type": "noul", "instructions": "Urgent?"}})
```

## Audit

The browser-use use case: capture every decision and ship it to an external service.

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

One hook call covers the whole call, so the loop writes one record per decision; see
[Batch](#batch) for the same shape on `predict_batch`.

A full runnable version is in [`examples/hooks/audit.py`](https://github.com/2351548518/laya/blob/main/examples/hooks/audit.py).

## Redact PII

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

See [`examples/hooks/redact.py`](https://github.com/2351548518/laya/blob/main/examples/hooks/redact.py).

## Cache

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

See [`examples/hooks/cache.py`](https://github.com/2351548518/laya/blob/main/examples/hooks/cache.py).

## Metrics

```python
import laya_multimodal

COUNTS, LATENCIES = {}, []

def metrics(ctx):
    COUNTS[ctx.model] = COUNTS.get(ctx.model, 0) + 1
    if ctx.elapsed_ms is not None:
        LATENCIES.append(ctx.elapsed_ms)

agent = laya_multimodal.load( on_predict_end=metrics, hooks_raise=False)
```

See [`examples/hooks/otel.py`](https://github.com/2351548518/laya/blob/main/examples/hooks/otel.py).

## Guardrail

Block a request by raising from a start hook.

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

A start hook sees every state of the call, so test them all: reading only `ctx.states[0]` lets the
rest of a `predict_batch` call through.

## Confidence gate

Rewrite a low-confidence answer, or annotate it.

```python
def gate(ctx):
    for result in ctx.results or []:
        answer = result["answers"].get("dept")
        if answer and answer["confidence"] < 0.6:
            answer["choice"] = "human-review"
            answer["gated"] = True

agent = laya_multimodal.load( on_predict_end=gate)
```

`ctx.results` holds one dict per state of the call, so the loop annotates every answer that
misses the threshold, not only the first state's.

## Routing pin

Force a checkpoint for a class of traffic. A routing hook is an object, not a plain callable:
`on_predict_start=` / `on_predict_end=` are the only single-event parameters, so `hooks=` is the
only way in and it reads for lifecycle method names.

```python
from laya_multimodal import Router
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

router = Router(models={"support": "models/support"}, hooks=[Pin()])
```

Per-call, without installing:

```python
router.predict("refund request", QUESTIONS, hooks=[Pin()])
```

## Lifecycle

Observe checkpoint build and eviction.

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

## Composition

Several hooks of different kinds compose naturally. Within one scope, `hooks=[...]` entries run in
list order and the `on_predict_start=` / `on_predict_end=` convenience callables follow. Across
scopes, process-wide default hooks come before an instance's hooks, and an instance's hooks before
per-call hooks. Every hook of one call shares that call's context.

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

## Per-call hooks

Override or extend hooks for a single call.

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

## Batch

Hooks fire once per `Agent.predict_batch` call, with `ctx.states` holding every state.
`Router.predict_batch` runs its Router-level hooks once per request instead, each with one state
and its own `run_id`, so the same hook there writes one record per call of the hook.

```python
def audit_batch(ctx):
    for state, result in zip(ctx.states, ctx.results):
        ship_one(ctx.run_id, state, result)

results = agent.predict_batch([state_a, state_b, state_c], questions, on_predict_end=audit_batch)
```

## HTTP server

Router hooks fire for `laya_multimodal.serve` automatically, because the server calls `Router.predict`.

```python
from laya_multimodal import Router
from laya_multimodal.serve import create_app

router = Router(hooks=[Metrics()], on_predict_end=audit, hooks_raise=False)
app = create_app(router=router)
```

## Runtime registration

Attach, detach or scope hooks after construction.

```python
metrics = Metrics()
agent.add_hook(metrics)            # attach at runtime
agent.remove_hook(metrics)         # by identity: keep the instance to remove it

with agent.hooks_installed(DebugDump()):
    agent.system_one(state, questions)   # DebugDump only here
```

`remove_hook` compares with `is`, so a fresh `Metrics()` matches nothing: it returns `False` and
the installed hook keeps running.

## Base class and process-wide defaults

Subclass `BaseHook` to override only what you need, and register something once for the whole
process instead of passing it to every `Agent` and `Router`.

```python
from laya_multimodal import BaseHook
from laya_multimodal.decisions import hooks

class Audit(BaseHook):
    def on_predict_end(self, ctx):
        ship(ctx.run_id, ctx.results)

hooks.set_default_hooks(hooks=[Audit()])   # runs for every call in the process

# later, or in tests:
hooks.clear_default_hooks()
```

## Token budget

Pass `max_len` for an individual native prediction.

```python
agent.predict(state, questions, max_len=8192)
```

Use a [start hook](patterns.md#token-budget-shaping) for budgets that depend on the question set.

## Async hooks

Wrap an async hook in `AsyncHook`; each coroutine runs to completion in the sync core, whether the
caller is synchronous or already inside an event loop.

```python
import laya_multimodal
from laya_multimodal import AsyncHook

class RemoteAudit:
    async def on_predict_end(self, ctx):
        await ship(ctx.run_id, ctx.results)

agent = laya_multimodal.load( hooks=[AsyncHook(RemoteAudit())])
```

A plain async callable works too:

```python
async def async_end(ctx):
    await ship(ctx.results)

agent.system_one(state, questions, on_predict_end=async_end)
```

## Hook timeout

Bound each hook call, so a stuck hook cannot hang a served request:

```python
agent = laya_multimodal.load( on_predict_end=metrics, hooks_timeout=2.0)

# or per call
agent.system_one(state, questions, on_predict_end=metrics, hooks_timeout=0.5)
```

A timed-out hook raises `TimeoutError` (or warns when `hooks_raise=False`). The hook keeps running
in the background, so also give network calls their own timeout. See
[errors](errors.md#timeouts).

## Testing hooks

Assert what a hook saw without a model: drive `predict_batch` with the encode/forward/decode
helpers stubbed, as [`tests/test_hooks.py`](https://github.com/2351548518/laya/blob/main/tests/test_hooks.py) does.

```python
seen = []
agent.predict_batch(["s0"], questions, on_predict_end=lambda ctx: seen.append(ctx.results))
assert len(seen) == 1
```

The API surface is pinned by [`tests/test_hooks_api.py`](https://github.com/2351548518/laya/blob/main/tests/test_hooks_api.py).

## See also

- [Tracing](tracing.md): `run_id`, spans, OpenTelemetry.
- [Patterns and anti-patterns](patterns.md): the reasoning behind these recipes.
