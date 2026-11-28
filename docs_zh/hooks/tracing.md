# 请求追踪

[English](../../docs/hooks/tracing.md) · **简体中文**

同一次预测的开始、结束与错误事件通过 `PredictContext.run_id` 关联。

## run_id

预测语境使用 `uuid4().hex`。Agent 批量共享一次调用的 ID，Router 批量每条请求一个 ID。
`on_route`、`on_load`、`on_evict` 创建自己的语境。用预测时的 `ctx.decision` 关联
选路结果，将 `run_id` 写入日志或外发元数据来跨进程追踪。

```
call A:  run_id=1a2b...  start ──┐
                                 ├─ end
                          error ─┘
call B:  run_id=9f8e...  start ─── end
```

<a name="a-minimal-tracer"></a>

## 最小追踪器

开始时记录 span，结束时根据 `run_id` 取出并完成它。

```python
import time

class Tracer:
    def __init__(self):
        self.spans = {}

    def on_predict_start(self, ctx):
        self.spans[ctx.run_id] = {
            "model": ctx.model,
            "started_at": ctx.started_at,
        }

    def on_predict_end(self, ctx):
        span = self.spans.pop(ctx.run_id, None)
        if span is None:
            return
        emit_span(
            name="laya.predict",
            run_id=ctx.run_id,
            model=ctx.model,
            duration_ms=ctx.elapsed_ms,
            usage=ctx.usage,
            ok=ctx.error is None,
        )

    def on_error(self, ctx):
        # on_error runs before on_predict_end; leaving the span for on_predict_end is fine,
        # or close it here if you prefer.
        pass

agent = laya_multimodal.load( hooks=[Tracer()])
```

<a name="span-lifetime"></a>

## Span 生命周期

```
on_predict_start ──► open span (run_id, model, started_at)
      │
      ├─ inference
      │
on_error ──► record ctx.error on the span
      │
on_predict_end ──► close span (elapsed_ms, usage, ok)
```

<a name="error-spans"></a>

## 失败路径

结束事件携带 `ctx.error`，可以统一关闭 span 并记录失败。处理两种事件时，选择一个关闭位置。

```python
def on_predict_end(self, ctx):
    span = self.spans.pop(ctx.run_id, None)
    if span is None:
        return
    if ctx.error is not None:
        span["status"] = "error"
        span["error_type"] = type(ctx.error).__name__
        span["error_message"] = str(ctx.error)
    span["duration_ms"] = ctx.elapsed_ms
    emit(span)
```

## OpenTelemetry

通过 OTel API 记录 span。阻塞导出可放到后台队列，纯追踪故障可用 `hooks_raise=False` 记录。

```python
from opentelemetry import trace

tracer = trace.get_tracer("laya")

class OTelHooks:
    def __init__(self):
        self.spans = {}

    def on_predict_start(self, ctx):
        span = tracer.start_span("laya.predict", attributes={"laya.run_id": ctx.run_id, "laya.model": ctx.model})
        self.spans[ctx.run_id] = span

    def on_predict_end(self, ctx):
        span = self.spans.pop(ctx.run_id, None)
        if span is None:
            return
        if ctx.usage:
            span.set_attribute("laya.input_tokens", ctx.usage["input_tokens"])
        if ctx.error is not None:
            span.record_exception(ctx.error)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
        span.end()

laya_multimodal.load( hooks=[OTelHooks()], hooks_raise=False)
```

[OTel 示例](https://github.com/2351548518/laya/blob/main/examples/hooks/otel.py)

<a name="distributed-propagation"></a>

## 跨进程传播

在日志、审计载荷或下游请求头传递字符串 ID。高并发场景可队列化写入。

```python
def audit(ctx):
    requests.post(
        "https://audit.example/decisions",
        json=record(ctx),
        headers={"X-Laya-Run-Id": ctx.run_id},
        timeout=2,
    )
```

[阻塞工作](patterns.md#blocking-work)

<a name="nested-calls"></a>

## 嵌套调用

每个子预测有自己的 ID，由子调用的开始钩子读取。批量时遍历父调用的全部状态，并逐个关联。

```python
def enrich(ctx):
    for state in ctx.states:                     # a hook sees every state of the call
        def link(child_ctx, parent_run_id=ctx.run_id):
            record_child_span(parent_run_id=parent_run_id, child_run_id=child_ctx.run_id)

        enricher.predict(state, EXTRA_QUESTIONS, on_predict_start=link)
```

用独立的无钩子 Agent 执行辅助预测，或设置递归保护。
[递归调用](patterns.md#recursive-predict) · [语境 API](api.md#predictcontext)
