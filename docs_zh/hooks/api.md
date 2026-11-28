# 预测钩子 API

[English](../../docs/hooks/api.md) · **简体中文**

常用类型从 `laya_multimodal` 导入，完整的辅助 API 位于 `laya_multimodal.decisions.hooks`。

```python
from laya_multimodal import PredictContext, PredictHook, Hook, OptionStabilityHook
from laya_multimodal.decisions.hooks import HOOK_EVENTS, normalise_hooks, dispatch, aggregate_usage
```

## PredictContext

每次公开预测创建一个可变语境。`Router.predict_batch` 为每条请求创建一个。
语境按对象身份比较。钩子可以替换输入与结果，`on_route` 可以修改路由。

```text
@dataclass(eq=False)
class PredictContext:
    states: List[Any]
    questions: Dict[str, Any]
    run_id: str = <uuid4 hex>
    results: Optional[List[Dict[str, Any]]] = None
    decision: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    agent: Any = None
    router: Any = None
    max_len: Optional[int] = None
    head_max_len: Optional[int] = None
    usage: Optional[Dict[str, int]] = None
    started_at: float = <perf_counter()>
    elapsed_ms: Optional[float] = None
    error: Optional[BaseException] = None
```

| 字段 | 含义 |
| --- | --- |
| `states`、`questions` | 当前状态列表与题目映射，可在开始事件替换 |
| `run_id` | 开始、出错与结束事件共用的预测 ID |
| `results` | 每个状态的结果，推理完成或调用 `skip` 后获得 |
| `decision` | Router 的模型选择，选路事件可修改 |
| `model`、`agent`、`router` | 当前模型、运行时与 Router |
| `max_len`、`head_max_len` | 输入预算和兼容后端的候选预算，原生模型使用前者 |
| `usage` | 结束时汇总输入和输出 token |
| `started_at`、`elapsed_ms` | 性能计时与结束时的毫秒数，Router 的计时从加载后开始 |
| `error` | 失败路径上的原异常 |

### `PredictContext.skip(results)`

开始事件可提供完整结果以跳过推理。结果数与状态数相同，且仍运行后续结束钩子。
缓存键覆盖状态、题目顺序、检查点和 token 预算。Router 会补齐缺失的 `routing`。

```python
import hashlib, json

CACHE = {}

def key(ctx, index):
    # Not sort_keys=True: criteria order is positional, so two orders are two questions,
    # and the checkpoint and token budget change the answer too.
    payload = json.dumps([ctx.states[index], ctx.questions, ctx.model,
                          ctx.max_len, ctx.head_max_len], default=str)
    return hashlib.sha256(payload.encode()).hexdigest()

def cache_read(ctx):
    hits = [CACHE.get(key(ctx, i)) for i in range(len(ctx.states))]
    if all(hit is not None for hit in hits):
        ctx.skip(hits)   # one entry per state, same shape as predict_batch's return

def cache_write(ctx):
    for i, result in enumerate(ctx.results or []):
        CACHE[key(ctx, i)] = result

laya_multimodal.load( on_predict_start=cache_read, on_predict_end=cache_write)
```

<a name="hook-protocol"></a>

## 事件协议

实现一个或多个事件。事件属性需为可调用对象。

```python
class Hook(Protocol):
    def on_predict_start(self, ctx: PredictContext) -> None: ...
    def on_predict_end(self, ctx: PredictContext) -> None: ...
    def on_route(self, ctx: PredictContext) -> None: ...
    def on_load(self, ctx: PredictContext) -> None: ...
    def on_evict(self, ctx: PredictContext) -> None: ...
    def on_error(self, ctx: PredictContext) -> None: ...
```

| 事件 | 触发时机 | 可改写 |
| --- | --- | --- |
| `on_predict_start` | 推理前 | 状态、题目、预算、缓存结果 |
| `on_predict_end` | 成功或失败路径的收尾 | 结果 |
| `on_route` | Router 选路后、加载前 | 路由决策 |
| `on_load` | Router 加载完成后 | 观察 |
| `on_evict` | Router 卸载后 | 观察 |
| `on_error` | 预测失败时 | 观察原异常 |

### BaseHook

继承空实现，并覆盖需要的事件。

```python
from laya_multimodal import BaseHook

class Audit(BaseHook):
    def on_predict_end(self, ctx):
        ship(ctx.run_id, ctx.results)
```

## OptionStabilityHook

以候选重排或重命名探测选择稳定性，具体探测由后端支持和候选布局决定。
`accept=None, escalate=None` 可先采集连续值，再在独立数据上选阈值。

```python
from laya_multimodal import OptionStabilityHook

# Required accept and escalate thresholds; pass None to disable categorical triage:
stability_hook = OptionStabilityHook(accept=0.85, escalate=0.60, variants="fast")
agent = laya_multimodal.load( hooks=[stability_hook])
```

`reliability.probes` 记录实际运行的探测。`soft_stability` 为主选项在变体上的平均概率，
`stability` 为保持同一主选项的比例。`rename_stability`、`reorder_stability` 分别给出
子集统计。通过 `variant_choices` 和 `variant_support` 查看各变体。没有可用探测时，
连续指标可为 `None`。`decision` 为 `ACCEPT`、`VERIFY`、`ESCALATE` 或 `None`。

<a name="calibrating-and-replaying-decisions"></a>

### 校准与回放

收集有效探测上的 `soft_stability` 和相应正确性，用 `calibrate_reliability` 拟合
复核策略。回放 `decide_reliability` 时使用同一指标定义，并保留没有探测的样本。
[评测](../evals.md)

<a name="what-it-costs"></a>

### 资源开销

变体增加参与计算的记录。用目标后端、实际候选数和问题分布测量吞吐与延迟。
[延迟评测](../benchmarks.md#latency)

<a name="convenience-types"></a>

## 单事件函数

```python
PredictHook = Callable[[PredictContext], None]
```

`on_predict_start` 和 `on_predict_end` 分别接受一个函数或一组函数。

<a name="runtime-registration"></a>

## 运行时注册

注册表在调用开始时取快照。临时作用域退出时只移除该作用域新增的实例。

```python
agent.add_hook(tracer)              # one hook or a sequence; returns self for chaining
agent.remove_hook(tracer)           # by identity; True if it was installed

with agent.hooks_installed(debug):  # installed for the block, removed on exit
    agent.system_one(state, questions)
```

<a name="process-wide-defaults"></a>

## 进程默认钩子

进程默认、实例和单次钩子依次运行，包括 Router 的加载与卸载事件。更改会在下次调用生效。

```python
from laya_multimodal.decisions import hooks

hooks.set_default_hooks(hooks=[Tracer()])          # replaces the set, accepts the hooks= arguments
hooks.add_default_hook(Metrics())                  # appends
hooks.clear_default_hooks()                        # removes everything
hooks.default_hooks()                              # a copy of the current list

hooks.compose_hooks(agent.hooks)                   # defaults + installed (advanced)
```

<a name="async-hooks"></a>

## 异步回调

```python
from laya_multimodal import AsyncHook

class Remote:
    async def on_predict_end(self, ctx):
        await ship(ctx.results)

agent = laya_multimodal.load( hooks=[AsyncHook(Remote())])
```

可在 `AsyncHook` 中包装协程事件，普通单事件参数也可返回 awaitable。
调用线程没有运行中的循环时用 `asyncio.run`，否则由后台循环执行。显式传入的循环需已运行，
并位于另一线程。

<a name="configuration-surface"></a>

## 配置

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `hooks` | `None` | 一个或多个事件对象 |
| `on_predict_start`、`on_predict_end` | `None` | 对应事件的函数 |
| `hooks_raise` | `True` | 抛出钩子异常，关闭后告警并继续 |
| `hooks_concurrent` | `True` | 关闭后对单个回调加调度锁 |
| `hooks_timeout` | `None` | 每个回调的等待秒数 |

### Agent

```python
agent = load(model_id_or_path=None, device=None, hooks=None,
             on_predict_start=None, on_predict_end=None,
             hooks_raise=True, hooks_concurrent=True, hooks_timeout=None)

agent.predict(state, questions, image=None, audio=None, video=None,
              hooks=None, on_predict_start=None, on_predict_end=None,
              hooks_raise=None, hooks_timeout=None, max_len=None)

agent.predict_batch(states, questions, batch_size=None,
                    hooks=None, on_predict_start=None, on_predict_end=None,
                    hooks_raise=None, hooks_timeout=None, max_len=None)

agent.system_one(...)       # alias of predict
```

### Router

```python
Router(
    models=None, device=None, token=None, max_loaded=2, default="multimodal",
    preload=False, lang_guess=None, multimodal=None,
    hooks=None, on_predict_start=None, on_predict_end=None,
    hooks_raise=True, hooks_concurrent=True, hooks_timeout=None,
)

router.route(state, questions=None, model=None, task=None, lang=None, lang_guess=None,
             hooks=None, hooks_raise=None, hooks_timeout=None)

router.predict(state, questions, model=None, task=None, lang=None, lang_guess=None,
               hooks=None, on_predict_start=None, on_predict_end=None, hooks_raise=None,
               hooks_timeout=None, max_len=None)

router.predict_batch(requests, batch_size=None, hooks_timeout=None, min_confidence=None,
                     sort_by_length=False, hooks=None, on_predict_start=None, on_predict_end=None,
                     hooks_raise=None, diagnostics=None)

router.route_batch(requests, hooks_timeout=None, hooks=None, hooks_raise=None)

router.system_one(...)      # alias of predict
router.load(name)           # builds on first use; fires on_load
router.preload(names=None)  # builds several; fires on_load per build
router.unload(name=None)    # frees one or all; fires on_evict
router.attach(name, agent)  # registers an existing agent; does not fire on_load
router.loaded               # list of resident checkpoint names
```

<a name="event-payloads"></a>

## 事件载荷

预测开始有状态和题目，结束有耗时以及成功结果或原异常。`on_error` 早于结束用量与耗时的计算。
选路有 `decision`，加载与卸载有目标模型，并各有自己的语境。空输入成功和预测失败可以通过
`results` 与 `usage` 区分。 [生命周期](lifecycle.md)

<a name="validation"></a>

## 校验

构造时验证实例钩子，调用时验证单次钩子。传实例而非类，实现至少一种事件，并确保事件属性可调用。
单事件函数用对应的关键字参数。类型错误为 `TypeError`。

<a name="advanced-helpers"></a>

## 辅助 API

```text
HOOK_EVENTS
normalise_hooks(hooks=None, on_predict_start=None, on_predict_end=None) -> list
dispatch(hooks, event, ctx, *, raise_errors=True, lock=None, timeout=None) -> None
aggregate_usage(results) -> {"input_tokens": int, "output_tokens": int}
run_coroutine_sync(coro, loop=None)
```

`normalise_hooks` 按顺序组装回调。`dispatch` 使用异常、锁和超时策略执行事件。
`aggregate_usage` 汇总用量，`run_coroutine_sync` 从同步调用中执行 awaitable。

```python
from laya_multimodal.decisions.hooks import normalise_hooks, dispatch, PredictContext

hooks = normalise_hooks(on_predict_start=[log, redact])
ctx = PredictContext(states=["..."], questions={...})
dispatch(hooks, "on_predict_start", ctx)
```

[生命周期](lifecycle.md) · [错误处理](errors.md) · [使用模式](patterns.md)
