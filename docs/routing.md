# Routing

**English** · [简体中文](zh-CN/routing.md)

The default router serves the configured native checkpoint for text and media.
Set `LAYA_MULTIMODAL_MODEL` or pass `multimodal=` to choose it. Request a named
checkpoint when a task needs another backbone, such as Gemma 4 for waveform audio.

```python
from laya_multimodal import Router

router = Router(
    multimodal="models/jev-multimodal-qwen3.5-4b",
    models={"gemma": "models/decisions-gemma4"},
    device="cpu",
)
questions = {"speaker": {"type": "noul", "instructions": "Is anyone speaking?"}}
result = router.predict("A short recording", questions, model="gemma", audio="clip.wav")
```

Call `route` to inspect a checkpoint choice before loading weights. `predict_batch`
groups requests by selected checkpoint and returns the results in input order.
Use `preload` for checkpoints that should be resident before a request.

Set `LAYA_EXTRA_MODELS` to a JSON object mapping names to local paths or Hub IDs for HTTP
deployments. Set `LAYA_DEFAULT_MODEL` to select one of those names. [HTTP API](http-api.md) ·
[Router reference](reference/router.md)
