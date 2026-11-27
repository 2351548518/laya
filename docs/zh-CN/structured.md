# 按 Schema 决策

[English](../structured.md) · **简体中文**

把 JSON Schema 或 Pydantic 模型转换为问题集。读出头分别给每道问题的候选评分，随后按 schema
的类型投影答案。

```python
import laya_multimodal

schema = {
    "type": "object",
    "properties": {
        "department": {"type": "string", "enum": ["billing", "support", "sales"],
                       "description": "Which team should handle this?"},
        "urgency": {"type": "integer", "minimum": 0, "maximum": 2},
        "needs_human": {"type": "boolean"},
    },
}

agent = laya_multimodal.load()
values = agent.decide("I was charged twice, refund me.", schema=schema)
# {"department": "billing", "urgency": 2, "needs_human": True}
```

使用 Pydantic 时安装 `laya-multimodal[structured]`。

```python
from typing import Literal
from pydantic import BaseModel

class Ticket(BaseModel):
    department: Literal["billing", "support", "sales"]
    urgency: Literal[0, 1, 2]
    needs_human: bool

ticket = agent.decide("I was charged twice, refund me.", schema=Ticket)
```

<a name="the-supported-subset"></a>

## 支持的结构

顶层为含 `properties` 的对象，每个属性对应一道决策题。

| 属性定义 | 问题 | 返回 |
| --- | --- | --- |
| `{"enum":["billing","support"]}` | `choice` | 原值及其类型 |
| `{"const":"billing"}` | `choice` | 唯一候选 |
| `{"type":"boolean"}` | `noul` | `true` 或 `false` |
| `{"type":"integer","minimum":0,"maximum":5}` | `score` | `minimum + argmax` |
| `{"type":"number","minimum":0,"maximum":5}` | `score` | 整数档位 |
| 含 `description` 的属性 | 对应类型 | `description` 用作题目指令 |
| 仅一个非空分支的 `anyOf`、`oneOf` 或类型列表 | 非空分支对应的类型 | 无答案时省略可选字段 |
| 引用本地 `$defs`／`definitions` 的 `$ref` | 按引用的属性定义 | 对应的原值 |
| 单项 `allOf` | 按内部属性定义 | 对应的原值 |

`Literal` 与 `Optional` 对应枚举与单非空分支。Pydantic 的枚举通常经本地引用解析。
字段的 `description` 形成指令，避免用类型标题代替问题语义。选择值保留原类型，
例如整数 2 返回整数。布尔值按 `noul >= 0.5` 投影。

<a name="rejections"></a>

## 校验错误

不支持的属性报 `laya_multimodal.structured.SchemaError`，错误消息包含属性路径。
常见问题包括自由字符串、数组、嵌套对象、无法解析或循环的引用、无有效候选的枚举、
转换后标签重复、无整数界限或反序的数字区间，以及含多个非空类型的联合。

最多 `MAX_PROPERTIES = 32` 个属性，每题 `MAX_OPTIONS = 32` 个选项，数字范围
`MAX_SCORE_LEVELS = 10` 个档位。将复杂结构扁平化，并为每个字段定义有限候选。

<a name="the-api"></a>

## Python 接口

| 函数 | 用途 |
| --- | --- |
| `laya_multimodal.decide`、`Agent.decide`、`Router.decide` | 对一个状态决策 |
| `decide_batch` | 多状态批量决策 |
| `questions_from_json_schema`、`questions_from_pydantic` | 创建问题集 |
| `answers_to_json`、`answer_to_pydantic` | 转回原 schema 的值 |
| `plan_from_json_schema` | 检查属性映射方案 |

`schema` 与 `questions` 选其一。传 `questions` 时返回原始答案。其他预测关键词转给
运行器。

```python
router.decide(state, schema=Ticket, model="multimodal", hooks=[Metrics()])
```

<a name="scoring-many-states"></a>

## 批量决策

先规划一次 schema，再交给 `predict_batch`，返回值顺序对应输入。`return_details=True`
为每条状态给出 `DecisionResult`。

```python
values = agent.decide_batch(ticket_texts, schema=Ticket)          # values[i] matches ticket_texts[i]
results = router.decide_batch(states, schema=Ticket, return_details=True, batch_size=64)
```

`Router` 为各状态分别路由。批次控制如 `batch_size`、`sort_by_length` 作为调用参数，
模型、语言与 token 预算按条传给路由器。直接 Agent 的模型固定为加载的检查点。
若运行器没有 `predict_batch`，用单条 `decide`。实际前向次数取决于后端。

<a name="confidence-and-probabilities"></a>

## 置信度与概率

```python
result = agent.decide(state, schema=Ticket, return_details=True)
result.values["department"]            # "billing"
result.answer_confidence["department"] # 0.94  max(p): the quantity min_confidence gates on
result.confidence["department"]        # 0.79  1 - H(p)/log(k) on this 3-option `choice` field
result.probabilities["department"]     # {"billing": 0.94, "support": 0.06, "sales": 0.0}
result.usage                           # {"input_tokens": ..., "output_tokens": 0,
                                       #  "state_tokens": ..., "state_tokens_dropped": ...,
                                       #  "truncated": ..., "truncated_questions": [...]}
result.routing                         # the Router decision, when a Router answered
```

`DecisionResult.usage` 保留所用运行器的用量字段。原生推理包括 `input_tokens`、
`output_tokens`、`forward_passes` 和 `truncated`。候选最大概率见
`answer_confidence`，缺少此字段时可返回 `None`。`confidence` 的本地计算方式随题型变化。
使用在同类问题上拟合的门槛。

```python
if result.answer_confidence["department"] < 0.6:
    result.values["department"] = "human-review"
```

[校准](multimodal.md#calibration-and-evaluation) · [置信度语义](questions-and-answers.md#reading-confidence)

<a name="how-it-maps-internally"></a>

## 投影规则

枚举以字符串标签做候选，返回时映射为原始值。有限整数以各个整数为有序档位。布尔字段使用
肯定概率。题目指令来自属性说明。本地引用先解析并应用属性键，引用定义的描述由属性说明覆盖。
可选字段只使用非空 schema 建题，无答案时省略该键。
[LangChain 接入](langchain.md) · [预测钩子](hooks/index.md)
