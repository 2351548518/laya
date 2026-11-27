# 钩子错误处理

[English](../../hooks/errors.md) · **简体中文**

<a name="the-two-policies"></a>

## 错误策略

`hooks_raise=True` 是默认值，钩子异常会传播。设置 `False` 时，普通钩子异常转换成
`RuntimeWarning`，后续工作继续。可以按实例设置，也可以在一次调用中覆盖。

```python
# strict: a broken audit hook fails the request
laya_multimodal.load( on_predict_end=audit, hooks_raise=True)

# lenient: telemetry must never take down a served request
laya_multimodal.load( on_predict_end=metrics, hooks_raise=False)
```

<a name="what-runs-when-something-fails"></a>

## 失败时的事件

```
try:
    on_predict_start
    inference
except BaseException as exc:
    ctx.error = exc
    on_error            (best effort; cannot mask exc)
    raise
finally:
    elapsed_ms, usage
    on_predict_end      (best effort; cannot mask exc on the failure path)
```

| 事件 | 抛错后的行为 |
| --- | --- |
| `on_predict_start` | 严格模式下触发 `on_error` 与 `on_predict_end` 后抛出。宽松模式发警告并保留已经做出的修改。 |
| 推理 | 把异常放入 `ctx.error`，运行错误和结束事件，再抛出。 |
| `on_error` | 保留原异常，钩子异常成为上下文。 |
| `on_predict_end` | 成功路径上的异常遵循 `hooks_raise`。失败路径保留原异常。 |
| `on_route` | 作为预测的一部分失败时进入预测错误链。宽松模式保留当时的决策。 |
| `on_load` | 检查点可能已加载并缓存，此后命中缓存不会再次触发加载。 |
| `on_evict` | 检查点已卸载，调度策略决定警告或抛错。 |

<a name="exception-chaining"></a>

## 异常链

处理错误或执行结束钩子时再次失败，原始异常继续抛出，钩子异常可从 `__context__` 读取。

```python
class BadTelemetry:
    def on_error(self, ctx):
        raise RuntimeError("telemetry down")

try:
    agent.system_one(state, questions, hooks=[BadTelemetry()])
except RuntimeError as exc:
    assert exc.__context__ is not None   # the telemetry failure
```

## BaseException

`KeyboardInterrupt` 和 `SystemExit` 等通过普通钩子异常处理直接传播。预测生命周期仍尝试运行
错误和结束事件。回调应尽快完成，以便调用方及时退出。

<a name="configuration-errors"></a>

## 配置错误

钩子列表传实例，至少实现一个受支持的事件，并确保对应属性可调用。普通函数传给单事件参数。
实例配置在构造时校验，单次钩子在调用时校验，参数类型错误会抛出 `TypeError`。

<a name="warnings"></a>

## 警告

宽松模式警告示例如下。Python 警告过滤器可能合并同一位置的相同事件。
如需计数，应在日志或监控中单独累计。

```
laya: hook Metrics.on_predict_end failed: connection reset
```

<a name="timeouts"></a>

## 超时

`hooks_timeout` 为每个回调的秒数上限，接受正有限值。`None` 沿用实例设置或无时间限制。
严格模式为 `TimeoutError`，宽松模式发出运行时警告。

```python
laya_multimodal.load( on_predict_end=metrics, hooks_timeout=2.0)
```

超时限制请求等待时间。线程中的函数或事件循环中的协程可能继续运行，调度锁会释放。
如回调涉及文件或网络，给底层操作设置独立的超时。回调中的 `contextvars` 来自调用者的上下文副本。

<a name="choosing-a-policy"></a>

## 选择策略

影响数据或动作的钩子应显式处理失败。纯监控可配置继续执行。需要混用时，让日志钩子捕获并记录其
自身错误。 [生命周期](lifecycle.md) · [钩子模式](patterns.md)
