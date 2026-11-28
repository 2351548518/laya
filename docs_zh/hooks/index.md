# 预测钩子

[English](../../docs/hooks/index.md) · **简体中文**

在推理进程安装钩子，可记录审计日志、脱敏、查缓存、采集指标或修改路由。LangChain
组件也可在支持的本地节点上安装每次调用的钩子。
[LangChain](../langchain.md#8-prediction-hooks-on-a-single-node)

| 页面 | 内容 |
| --- | --- |
| [API](api.md) | 字段、事件、参数与默认值 |
| [生命周期](lifecycle.md) | 单条、批量和检查点加载的触发顺序 |
| [错误处理](errors.md) | `hooks_raise`、异常链及超时 |
| [使用模式](patterns.md) | 缓存、脱敏、复核和共享状态 |
| [示例](examples.md) | 对应事件的代码 |
| [追踪](tracing.md) | 运行 ID 和 span |

<a name="quick-start"></a>

## 开始使用

单个查询的结束回调可通过 `on_predict_end` 传入。

```python
import laya_multimodal

def log(ctx):
    print(ctx.model, ctx.results[0]["answers"], ctx.elapsed_ms)

agent = laya_multimodal.load( on_predict_end=log)
agent.system_one("I was charged twice.", {"urgent": {"type": "noul", "instructions": "Urgent?"}})
```

同时监听多种事件时，定义对象并放入 `hooks`。

```python
class Audit:
    def on_predict_start(self, ctx):
        print("start", ctx.run_id)

    def on_predict_end(self, ctx):
        print("end", ctx.run_id, ctx.usage, ctx.elapsed_ms)

    def on_error(self, ctx):
        print("failed", ctx.run_id, ctx.error)

laya_multimodal.load( hooks=[Audit()])
```

实例可以在运行时注册钩子，也可以只在一个作用域内安装。

```python
agent.add_hook(tracer)               # attach at runtime
with agent.hooks_installed(debug):   # installed for the block, removed on exit
    agent.system_one(state, questions)
```

在进程范围内共享的钩子通过默认注册表设置。

```python
from laya_multimodal.decisions import hooks

hooks.set_default_hooks(hooks=[Tracer()])
```

[运行时注册](api.md#runtime-registration) · [进程默认值](api.md#process-wide-defaults)

<a name="the-mental-model"></a>

## 触发方式

- `hooks=` 接收实现生命周期方法的对象。单事件函数放到 `on_predict_start` 或 `on_predict_end`。
- 同一次预测的开始、出错和结束事件共享可变 `PredictContext`。批量钩子遍历 `ctx.states`
  与 `ctx.results`，并可在开始时替换输入或用 `skip()` 提供缓存结果。
- Agent 钩子围绕推理。Router 还提供选择、加载与卸载事件。Router 批量调用为每条请求生成语境。

```
                             Router.predict(state, questions)
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  route()                                                                 │
   │    ├─ detect language / workflow                                         │
   │    └─ on_route          ctx.decision  (a hook may replace it)            │
   │                                                                          │
   │  load(decision.model)                                                    │
   │    ├─ build checkpoint on first use ──► on_load    ctx.model, ctx.agent  │
   │    └─ evict LRU checkpoint ───────────► on_evict   ctx.model             │
   │                                                                          │
   │  on_predict_start       ctx.states, ctx.questions, ctx.decision          │
   │    │                                                                     │
   │    ├── ctx.skip(results)? ──► skip the forward pass                      │
   │    │                                                                     │
   │    └── Agent.system_one(...)  ──►  Agent-level hooks run here            │
   │           on_predict_start  ─►  forward  ─►  on_predict_end              │
   │                                                                          │
   │  result["routing"] = decision                                            │
   │  on_predict_end         ctx.results, ctx.usage, ctx.elapsed_ms           │
   └──────────────────────────────────────────────────────────────────────────┘
                 any failure on the way ──► on_error, then on_predict_end
```

<a name="scope-at-a-glance"></a>

## 事件范围

| 事件 | `Agent` | `Router` |
| --- | --- | --- |
| `on_predict_start`、`on_predict_end`、`on_error` | 支持 | 支持 |
| `on_route`、`on_load`、`on_evict` | — | 支持 |

HTTP 和 MCP 在本地预测时使用路由器。将钩子安装到运行模型的进程。

<a name="compatibility"></a>

## 参数和默认行为

默认由进程级、实例级和调用级钩子组成执行列表。事件按顺序触发，异常行为由 `hooks_raise` 控制。
空请求也可产生生命周期记录。 [触发规则](lifecycle.md#ordering-rules) · [错误策略](errors.md)
