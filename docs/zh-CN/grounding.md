# 本地单框定位

[English](../grounding.md) · **简体中文**

单框头与其他决策共享骨干。读取最后一个有效隐藏状态，依次执行层归一化、线性投影、GELU
和末级线性层。五个输出分别编码四个坐标和一个定位质量 logit。训练配方中的 `single-v1` 头
为每道题定位一个被指代的物体。

<a name="prediction-contract"></a>

## 推理格式

加载经过单框监督训练的检查点。

```python
from laya_multimodal import MultimodalAgent

agent = MultimodalAgent("multimodal-runs/example/final", device="cuda")
result = agent.predict(
    image="scene.jpg",
    questions={"target": {"type": "box", "instructions": "the red mug"}},
)
target = result["answers"]["target"]
print(target["box"], target["confidence"])
```

每道定位题包含 `type` 和 `instructions`。答案的 `left`、`top`、`right` 和 `bottom` 都在
`[0, 1]` 范围内，框的宽高为正。横坐标乘原图宽度，纵坐标乘原图高度，可得到像素位置。
`confidence` 估计预测框与标注框的交并比（IoU）。每道定位题执行一次骨干前向，生成 token 数为零。

<a name="training-records"></a>

## 训练记录

单框正例把一道题与一个归一化的 `xyxy` 目标配对。

```json
{"id":"mug-1","group":"image-1","state":{"image":"images/scene.jpg"},"questions":{"target":{"type":"box","instructions":"the red mug"}},"grounding":{"task":"single","boxes":[[0.15,0.2,0.65,0.8]]}}
```

媒体路径可用绝对路径。相对路径从该 JSONL 文件的目录解析。按图片及来源分组划分训练、开发、
校准和测试。坐标需有限，位于图像内，并满足左小于右、上小于下。`task: "single"` 每题选择一个
被指代的物体。空 `boxes` 以 0 监督质量头，并跳过坐标损失。

`finetune_grounding` 可将定位行与 `choice`、`noul`、`score` 混合训练。框损失包含坐标 L1、
广义 IoU（GIoU）和以实测 IoU 为监督的质量损失。开发集选模可以组合框命中率与按模态宏平均的决策准确率。
[联合训练](finetune.md) · [张量维度](project-readme/multimodal-current.md#architecture)

<a name="evaluation-and-scope"></a>

## 评测口径

对单框正例报告平均 IoU 和 IoU ≥ 0.5 的命中率。比较不同实现时，记录数据分片、输入预算、
基座与标注转换。 [原生评测](benchmarks.md)
