# 决策应用

[English](../use-cases.md) · **简体中文**

让模型从应用已知的动作集合中选择。动作调用和参数校验由应用负责。

## 路由请求

用文本决策选择队列。候选键对应应用中的标识。定义各候选的含义，使评测数据的标注人员能够
按同一口径选择标签。

```python
from laya_multimodal import MultimodalAgent

agent = MultimodalAgent("models/jev-multimodal-qwen3.5-4b", device="cpu")
questions = {"queue": {
    "type": "choice",
    "instructions": "Which team should handle this request?",
    "criteria": {"billing": "charges or refunds", "support": "product help"},
}}
answer = agent.predict("I was billed twice.", questions)["answers"]["queue"]
print(answer["choice"], answer["probabilities"])
```

如果多条请求共用题目，可以用 `predict_batch` 批量推理，再按返回顺序处理。
[候选类型与置信度](questions-and-answers.md) · [HTTP 批量请求](http-api.md)

## 检查画面或视频

用 `noul` 判断照片是否需要人工复核，同一个请求中可以用 `box` 定位指定物体。画框前，
按图片宽高把归一化坐标换算成像素。用 Qwen 处理视频画面时显式设置 `video_audio: false`。
回答音轨或声音相关问题时，加载具备音频能力的 Gemma 检查点。
[输入格式](multimodal.md#local-prediction) · [单框示例](grounding.md#prediction-contract)

## 选择工具或结束任务

用 `choice` 枚举当前可用动作，例如 `search`、`respond` 和 `review`。应用把结果映射到明确
注册的工具或处理器。也可用 `noul` 判断是否满足停止条件，例如已有证据是否足够完成任务。
[工具选择](tools.md) · [LangGraph](langchain.md) · [CrewAI](crewai.md) · [Agno](agno.md)

## 路由检索或转人工

本地检索服务可把可调度的索引或查询引擎设为候选项。将多语言请求作为单独的评测切片，用相同的
问题结构和复核标签对比，再按计划自动处理的流量拟合置信度阈值。
[检查点路由](routing.md) · [LlamaIndex](llamaindex.md) · [文本评测](evals.md)

## 执行动作前观察工作流

试运行时记录 `run_id`、检查点、模型给出的概率及最终执行的动作。需要转人工或回退时，由应用
在执行入口作决定。 [预测钩子](hooks/index.md) · [分阶段上线](staged-adoption.md)
