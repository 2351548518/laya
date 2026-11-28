# 钩子生命周期

[English](../../docs/hooks/lifecycle.md) · **简体中文**

钩子顺序取决于调用入口。Router 还处理模型选择和常驻状态。

## Agent.predict_batch

`predict_batch` 是原生预测的实现。单条 `predict` 和 `system_one` 以一个状态调用它。

```
predict_batch(states, questions, batch_size=..., hooks=..., ...)
  │
  ├─ active  = default hooks + installed hooks + per-call hooks   (defaults first)
  ├─ ctx     = PredictContext(states, questions, model=self.model_id, agent=self)
  │
  ├─ try:
  │    │
  │    ├─ on_predict_start ─────────────────────────────┐
  │    │      a hook may:                                │
  │    │        • rewrite ctx.states / ctx.questions     │
  │    │        • set ctx.max_len                        │
  │    │        • ctx.skip(results) ─────────────┐       │
  │    │        • raise (aborts; see errors)     │       │
  │    │                                         │      │
  │    ├─ if ctx.results is not None:  ◄─────────┘      │  cache hit
  │    │      skip tokenization and forward             │
  │    ├─ else:                                         │
  │    │      validate states is a list                 │
  │    │      for each batch chunk:                     │
  │    │        _encode_state ─► collate ─► _forward    │
  │    │      _decode_answers                           │
  │    │      ctx.results = [...]                       │
  │    │                                                │
  │    └─ (any failure here) ──► except BaseException:  │
  │              ctx.error = exc                        │
  │              on_error                               │
  │              re-raise                               │
  │                                                     │
  │   finally:                                          │
  │     ctx.elapsed_ms = now - ctx.started_at           │
  │     if ctx.results is not None:                     │
  │         ctx.usage = aggregate_usage(...)            │
  │         apply_confidence_gate(..., min_confidence)  │
  │     on_predict_end ─────────────────────────────────┘
  │
  └─ return ctx.results
```

开始、错误和结束使用同一个 `ctx` 与 `run_id`。没有结果的失败使 `ctx.usage` 为 `None`。
成功的空批次返回空列表及归零的用量。

## Agent.system_one / predict

```
system_one(state, questions, hooks=..., ...)
  └─ predict_batch([state], questions, hooks=..., ...)[0]
```

单条语境的 `ctx.states` 为 `[state]`。

## Router.predict

```
Router.predict(state, questions, model=..., hooks=..., on_predict_start=..., on_predict_end=...)
  │
  ├─ active = default hooks + installed hooks + per-call hooks
  │
  ├─ route(state, questions, ..., hooks=per-call, hooks_raise=...)
  │    │
  │    ├─ _route(...)                      detect script / language / workflow
  │    ├─ on_route  ──► ctx.decision       a hook may replace the decision
  │    └─ return ctx.decision
  │
  ├─ load(decision["model"])
  │    │
  │    ├─ already resident? ──► return it
  │    ├─ else build Agent(...) ──► on_load   (after the Router lock is released)
  │    └─ evict LRU checkpoints ──► on_evict  (after the Router lock is released)
  │
  ├─ ctx = PredictContext(states=[state], questions, decision, model=decision.model,
  │                       agent=agent, router=self)
  ├─ try:
  │    ├─ on_predict_start
  │    ├─ if ctx.results is None:
  │    │      result = agent.system_one(ctx.states[0], ctx.questions)
  │    │        └─ the Agent's own hooks run here (start / forward / end)
  │    │      result["routing"] = decision
  │    │      ctx.results = [result]
  │    └─ else:
  │           for each cached result: result.setdefault("routing", decision)
  │    └─ (any failure) ──► except: on_error, re-raise
  │    └─ finally: elapsed_ms, usage, on_predict_end
  │
  └─ return ctx.results[0]
```

`on_route` 在加载前可修改模型。`ctx.elapsed_ms` 从选路和加载完成后计时。端到端耗时由调用方
测量。Agent 上的钩子仍按自身配置运行。Router 在短路结果中补齐 `routing`。

## Router.predict_batch

每条请求用自己的 `PredictContext`、`run_id` 和耗时。

```
Router.predict_batch(requests, batch_size=..., hooks=...)
  │
  ├─ active = default hooks + installed hooks + per-call hooks   (defaults first; None and [] add nothing)
  ├─ route_batch(requests, hooks=per-call) ──► on_route, once per request   (no checkpoint loaded yet)
  │
  └─ for each checkpoint, in order of first appearance:
       │
       ├─ load(checkpoint) ──► on_load / on_evict
       ├─ for each request of this checkpoint, in input order:
       │      ctx = PredictContext(states=[state], questions, decision, model, agent, router,
       │                           max_len=request.get("max_len"))
       │      on_predict_start       a hook may redact, rewrite, set a token budget or skip
       ├─ group the requests left to infer by questions and prompt budget
       │      agent.predict_batch(states, questions, ...)  ──► shared inference batch
       │      result["routing"] = decision;  ctx.results = [result]
       ├─ ctx.usage, once per request
       ├─ (any failure) ──► for every started request, in reverse input order:
       │                    ctx.error = exc, on_error, on_predict_end;  then re-raise
       └─ on_predict_end, once per request of this checkpoint, in reverse input order
```

- 同一检查点组先完成开始事件，再做分组推理。结束钩子填充的缓存可供后续调用读取。
- 结束事件按开始顺序的逆序退出，便于恢复 `contextvars` 或 OTel 上下文。
- 替换 `ctx.states`、`ctx.questions` 或 `max_len` 影响当前请求。共享字典应通过复制后替换，
  避免原地更改影响其他请求。
- 按解析后的检查点名称和预算组织请求。由开始钩子设置的 `ctx.max_len` 覆盖该请求的预算。
- 已启动的请求各执行一次结束钩子。如果一组失败，组中所有已启动请求都获得原异常，再触发结束事件。

<a name="model-lifecycle"></a>

## 模型加载与卸载

`on_load` 与 `on_evict` 在 Router 内部锁释放后触发。`attach` 注册已构造的对象。

```
load("gemma")                        (a separately registered checkpoint)
  │
  ├─ [lock]
  │    build MultimodalAgent(...)     (load weights and processor)
  │    register in _agents / _order
  │    evict LRU if over max_loaded ──► evicted = ["multimodal"]
  ├─ [unlock]
  ├─ on_evict("multimodal")
  └─ on_load("gemma")

unload("gemma")
  ├─ [lock] remove from _agents / _order
  ├─ [unlock]
  └─ on_evict("gemma")
```

<a name="caching-with-skip"></a>

## 用 skip 复用结果

```
on_predict_start
  ├─ cache hit?  ctx.skip([cached_result])
  │     └─ forward pass skipped
  │     └─ on_predict_end still runs
  │     └─ Router adds `routing` if missing
  └─ cache miss? nothing
        └─ forward pass runs
        └─ on_predict_end can store the result
```

[缓存示例](https://github.com/2351548518/laya/blob/main/examples/hooks/cache.py)

<a name="empty-inputs"></a>

## 空输入

空状态批返回 `[]`，有状态但无题目时每项返回空答案。两种成功路径都触发开始和结束事件，token
计为 0。失败且未生成结果时，用量为 `None`。

<a name="ordering-rules"></a>

## 顺序规则

1. 进程级默认钩子先于实例钩子，最后是单次调用钩子。
2. 同一事件按照列表顺序调用，实现该事件的钩子均完成后进入下一个事件。
3. 失败时先调用 `on_error` 再调用 `on_predict_end`。
4. 一次加载同时发生卸载和加载时，先执行 `on_evict` 再执行 `on_load`。

```
defaults: [D]      installed: [A, B]      per-call: [C]
on_predict_start: D, A, B, C
on_predict_end:   D, A, B, C
```

<a name="concurrency"></a>

## 并发

每个调用有自己的语境。共享钩子对象的可变状态需自行加锁，也可使用
`hooks_concurrent=False`。锁保护单次回调，两个请求仍可能在事件之间交错。

```
hooks_concurrent=True (default)      hooks_concurrent=False
  thread 1 ─┐                          thread 1 ─┐
  thread 2 ─┼─ hooks run in parallel   thread 2 ─┼─ one hook at a time
  thread 3 ─┘                          thread 3 ─┘   (RLock)
```

[错误处理](errors.md) · [钩子模式](patterns.md)
