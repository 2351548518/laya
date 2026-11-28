# 分阶段上线

[English](../docs/staged-adoption.md) · **简体中文**

逐步引入模型决策。模型提供答案、概率和钩子事件。应用负责执行和复核，并比较新旧流程。

<a name="1-shadow-observe-without-side-effects"></a>

## 1. 运行影子决策

在代表性流量上推理，由原有流程继续执行。记录请求类型、题目 schema、模型答案及候选概率、
检查点和 `run_id`，同时保留原流程动作、人工复核或真值、延迟、错误和回退情况。
用预测钩子采集模型侧信息。示例的日志及原流程处理函数由应用实现。

```python
from laya_multimodal import Router

class ShadowLog:
    def on_predict_end(self, ctx):
        result = ctx.results[0] if ctx.results else None
        write_shadow_record({
            "run_id": ctx.run_id,
            "model": ctx.model,
            "decision": ctx.decision,
            "result": result,
            "elapsed_ms": ctx.elapsed_ms,
            "error": None if ctx.error is None else repr(ctx.error),
        })

router = Router(hooks=[ShadowLog()])

def handle(request):
    try:
        router.predict(request.state, request.questions)
    except Exception as exc:
        record_laya_failure(request, exc)
        return run_incumbent_action(request)
    # The shadow result is recorded by the hook; do not execute it here.
    return run_incumbent_action(request)
```

按照应用策略脱敏。在应用入口捕获 `router.predict(...)` 的异常，以覆盖发生在预测生命周期开始
之前的失败。 [预测钩子](hooks/index.md) · [生命周期](hooks/lifecycle.md) · [追踪](hooks/tracing.md)

<a name="2-compare-disagreement-is-a-signal-not-a-verdict"></a>

## 2. 对比同一请求

确保新旧流程面对相同状态和问题含义。用人工复核或真实结果标注分歧。含糊样本进入待复核类别。
按检查点、语言、题目结构、动作和风险级别报告准确率、覆盖率及分歧比例。

<a name="3-choose-a-policy-from-held-out-evidence"></a>

## 3. 用独立数据确定上线策略

在有代表性的独立数据上拟合概率，再按应用可接受的覆盖率和错误代价选阈值。策略记录检查点、
题目版本、校准方法、阈值、评测集和负责人。高影响动作在执行前单独复核。

`laya-multimodal eval run --json` 报告记录 `config.revisions`、`config.dataset_sha256`、
`config.questions_sha256` 和 `config.thresholds`。本地目录在修订号字段中为 `null`。
校准方法和负责人由应用记录。 [置信度字段](questions-and-answers.md#reading-confidence) · [报告比较](evals.md)

<a name="attacker-controlled-context"></a>

### 对外部输入做成对检查

对可能由外部用户控制的文本，保留原样本及加入无关指令的对照。两者使用相同的任务与候选项，
比较所选动作和置信度。早期的少量示例说明不同检查点和问题措辞可能遇到不同失败。
在自己的场景验证。 [历史试验](https://github.com/NandhaKishorM/laya/issues/1229) · [守卫评测](guardrail-limitations.md)

<a name="4-promote-a-bounded-slice"></a>

## 4. 逐步放量

从已验证且有明确回退方式的一小类请求开始。限定检查点、题目结构、动作和流量上限。
保留复核与回退，持续采样并监控分歧、覆盖率、错误和延迟。达到约定条件后执行回滚。

<a name="applicationlaya-boundary"></a>

## 应用边界

模型通过 API 和钩子提供决策信息。应用持有执行、授权、复核、回退和回滚逻辑。

```text
real request
    ├─ incumbent action (authoritative)
    └─ Laya shadow decision ──> log, compare, evaluate
                                  └─ bounded eligible slice
                                      └─ review / fallback / rollback
```

<a name="rollout-checklist"></a>

## 上线核对

- 影子日志按 `run_id` 关联，实际动作仍由原流程执行。
- 对比记录包含原流程结果和可获得的复核标签。
- 阈值经过代表性独立数据的验证。
- 高成本或不可逆动作有明确复核入口。
- 已记录放量范围、回退条件和策略负责人。

[钩子 API](hooks/api.md) · [评测](evals.md)
