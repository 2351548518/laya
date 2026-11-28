# Python API

**English** · [简体中文](../../docs_zh/reference/index.md)

These pages describe the public local inference and training APIs.

- [Agent](agent.md) covers loading native checkpoints.
- [MultimodalAgent](multimodal.md) covers text, image, video, audio and Box requests.
- [Router](router.md): `Router` picks the checkpoint for each request; `RouteDecision` records
  the choice.
- [Helpers](helpers.md): language detection, email cleaning, question presets, shortlisting
  and calibration utilities.
- [Fine-tuning](train.md) covers `finetune_grounding`, `GroundingTrainConfig`, and the shared
  decision label conversion in `laya_multimodal.train`.
- [LangChain components](langchain.md): `LayaRouter`, `LayaGuardrail`, `LayaTriage` and
  `LayaEvaluator`.

Prediction hooks have a hand-written [API reference](../hooks/api.md) with the rest of the
[hooks guide](../hooks/index.md).
