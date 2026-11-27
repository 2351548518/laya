# 问题与答案

[English](../questions-and-answers.md) · **简体中文**

一个请求包括输入状态和按 ID 索引的问题。每道题定义自己的输出结构，模型按同名键返回答案。
未启用题目批处理时，原生 Agent 每题执行一次骨干前向。

```python
from laya_multimodal import MultimodalAgent

agent = MultimodalAgent("models/jev-multimodal-qwen3.5-4b", device="cpu")
questions = {
    "queue": {
        "type": "choice",
        "instructions": "Which team should handle the request?",
        "criteria": {"billing": "invoices or refunds", "support": "product help"},
    },
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "urgency": {"type": "score", "instructions": "How urgent is the request?",
                "criteria": ["routine", "soon", "immediate"]},
}
result = agent.predict("I was billed twice. Please refund the duplicate charge.", questions)
print(result["answers"], result["usage"])
```

<a name="the-three-candidate-based-types"></a>

## 三种候选型决策

| 题型 | 怎样定义 | 怎样读取 |
| --- | --- | --- |
| `choice` | 按顺序给出候选键及说明 | `choice` 是最高概率候选，`probabilities` 使用原候选键。 |
| `noul` | 给出要判断是否成立的条件 | `noul` 是条件成立的概率。 |
| `score` | 用 `criteria` 从低到高列出档位 | `score` 是档位索引的期望值，`probabilities` 和 `legend` 的索引从 0 开始。 |

为候选写简短、可区分的说明。`choice` 的说明为 `null` 时只渲染标签。原生 Python 接口可用
包含 `false` 和 `true` 的 `labels` 重命名是非选项，返回值仍为 `true` 的概率。词义变动后用
复核过的数据对比效果。

如果 `score` 的概率为 `[0.1, 0.2, 0.7]`，期望值就是 `1.6`。要读最可能的档位，取
概率最大的索引后在 `legend` 中查其含义。

## 用 Box 定位一个物体

`box` 使用一张图片，并在 `instructions` 中写出指代描述，例如 `the red cup`。返回值包括
归一化的 `left`、`top`、`right`、`bottom` 和定位质量分。 [单框请求与响应](grounding.md#prediction-contract)

<a name="probability-precision"></a>

## 概率精度

原生 `choice` 和 `score` 逐项保留四位小数。三个等概率候选可能各为 `0.3333`，合计
`0.9999`。下游需要单位概率质量时，可以使用随候选数变化的容差，或对分布的副本重新归一化。

<a name="reading-confidence"></a>

## 读取置信度

| 字段 | 原生本地接口的含义 |
| --- | --- |
| `answer_confidence` | 最大候选概率。`noul` 为 `max(p_true, 1 - p_true)`。 |
| `choice` 和 `score` 的 `confidence` | 反映整个分布的集中度，`k ≥ 2` 时为 `1 - H(p) / log(k)`。 |
| `noul` 的 `confidence` | 与 `answer_confidence` 相同，均为概率较大的那一侧。 |
| `box` 的 `confidence` | 预测框的 IoU 估计。 |

用相同检查点、模态和候选规模的独立数据拟合阈值，再用 `min_confidence` 标记低于阈值的决策。
在应用计划接收的覆盖率下测量校准和准确率。严格 HTTP 响应使用 System One 字段与置信度公式。
[校准](multimodal.md#calibration-and-evaluation) · [HTTP 响应](http-api.md#response)

<a name="presets"></a>

## 预设题目

`triage_questions`、`email_questions`、`guard_questions`、`moderation_questions` 和
`router_questions` 返回题目映射。查阅其中的指令和选项，再用业务样本验证。

<a name="reading-the-options-back"></a>

## 查看渲染后的选项

`render_options` 接收短字段名的内部结构。`crit` 的顺序决定提示中的选项顺序，调整顺序或说明
都会改变模型输入。

```python
from laya_multimodal import render_options

render_options({"t": "choice", "crit": {"billing": None, "support": "product help"}})
render_options({"t": "score", "crit": ["routine", "soon", "immediate"]})
```

候选项很多时，可使用候选缩减或分组淘汰。评估最后的决策准确率时也要统计正确选项的保留率。
[大候选集](reference/helpers.md#large-candidate-sets) · [原生评测](benchmarks.md)
