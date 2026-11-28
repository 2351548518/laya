# Local single-box grounding

**English** · [简体中文](../docs_zh/grounding.md)

The single-box head reads the same backbone used for typed decisions. It takes the last valid
hidden state and applies layer normalization, a linear projection, GELU and a final linear layer.
The five outputs encode four coordinates and one localization quality logit.

The starter recipes use the compact `single-v1` head for one referred object.

## Prediction contract

Load a checkpoint trained with single-box supervision.

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

A box question takes `type` and `instructions`. The answer has `left`, `top`, `right` and
`bottom` coordinates in `[0, 1]`, with positive width and height. Multiply horizontal coordinates
by the image width and vertical coordinates by the height for pixel positions. The answer's
`confidence` estimates the intersection over union (IoU) between the predicted and target boxes.

Each box question uses one backbone pass and returns zero generated tokens.

## Training records

A positive grounding row pairs one box question with one normalized `xyxy` target.

```json
{"id":"mug-1","group":"image-1","state":{"image":"images/scene.jpg"},"questions":{"target":{"type":"box","instructions":"the red mug"}},"grounding":{"task":"single","boxes":[[0.15,0.2,0.65,0.8]]}}
```

Keep images and source groups separate across training, development, calibration and test data.
Use absolute image paths or paths relative to the JSONL file.
Each box needs finite, ordered coordinates within the image. For `task: "single"`, select a
distinct referred object for each question. An empty `boxes` list gives the quality head a target
of zero and skips the coordinate losses.

`finetune_grounding` can mix these rows with `choice`, `noul` and `score` decisions. The box loss
combines L1 coordinate error, generalized IoU (GIoU) and quality supervision from the observed
IoU. During development, joint selection can combine box accuracy with macro decision accuracy
across modalities.

[Joint fine-tuning](finetune.md) · [Architecture with tensor dimensions](project-readme/multimodal-current.md#architecture)

## Evaluation and scope

For positive single-box examples, the local evaluator reports mean IoU and the fraction of
predicted boxes with IoU ≥ 0.5. Report the source split, input budget, backbone and annotation
conversion alongside these metrics when comparing with other implementations. [Native benchmarks](benchmarks.md)
