# Local single-box grounding

The multimodal runtime attaches a compact single-box head to the same backbone used for typed
decisions. It selects the last valid native state and applies layer normalization, a projection,
GELU and a five-value readout: four coordinates and a localization quality logit.

A `single-v1` checkpoint contains no object-query decoder, set matching head or detector slots.
Explicit legacy set checkpoints remain loadable for compatibility. New starter recipes use one
box and do not train multi-object detection.

## Prediction contract

Load a checkpoint trained with single-box supervision:

```python
from laya import MultimodalAgent

agent = MultimodalAgent("multimodal-runs/example/final", device="cuda")
result = agent.predict(
    image="scene.jpg",
    questions={"target": {"type": "box", "instructions": "the red mug"}},
)
box = result["answers"]["target"]["box"]
```

A box question has only `type` and `instructions`; it has no answer candidates. Coordinates are
normalized to the image and returned as `left`, `top`, `right` and `bottom`. Multiply horizontal
coordinates by image width and vertical coordinates by image height to obtain pixel positions.
The output has positive width and height. The quality score estimates IoU and is not
automatically a calibrated probability that an object is present.

A fresh localization head must be trained. A compact single-box checkpoint rejects multi-object
`detect()` calls rather than returning an untrained set prediction. Each question uses one
native backbone pass and generates zero answer tokens.

## Training records

A grounding row contains one box question and one normalized `xyxy` target:

```json
{"id":"mug-1","group":"image-1","state":{"image":"images/scene.jpg"},"questions":{"target":{"type":"box","instructions":"the red mug"}},"grounding":{"task":"single","boxes":[[0.15,0.2,0.65,0.8]]}}
```

Keep images and source groups separate across training, development, calibration and test data.
Malformed, nonfinite and out-of-frame boxes are rejected. Multiple annotations are not truncated
to one target. An empty target supervises absence/quality without inventing a zero-sized box.

`finetune_grounding` can mix these rows with native `choice`, `noul` and `score` decisions. The
single-box objective uses L1 coordinate error, GIoU and IoU-quality supervision. Development
selection can combine geometric accuracy with macro decision accuracy, with per-modality
retention checks.

See the [multimodal guide](multimodal.md) for JSONL loading, fresh-base initialization, staged
four-GPU training, calibration and evaluation commands. The core head is in `laya/grounding.py`,
training is in `laya/grounding_train.py`, and portable settings are in `configs/multimodal/`.

## Evaluation and scope

The local evaluator reports mean IoU and accuracy at IoU 0.5 on the supplied single-box cohort.
These scores are not COCO average precision. Report the exact source split, input budget,
backbone and annotation conversion when comparing with external methods.

The geometry and typed response contract are an open local implementation. They are not a claim
about proprietary Jev weights, architecture internals or an unpublished RLCD recipe.
