# 检查点路由

[English](../routing.md) · **简体中文**

默认路由用配置的原生检查点处理文本和媒体。通过 `LAYA_MULTIMODAL_MODEL` 或 `multimodal=`
选择它。需要另一种基座时可指定已注册检查点，例如使用 Gemma 4 处理音频波形。

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

调用 `route` 可在加载权重前查看选择结果。`predict_batch` 按检查点汇总请求并恢复原顺序。
需要提前载入权重时调用 `preload`。

HTTP 部署用 `LAYA_EXTRA_MODELS` 提供「名称 → 本地路径或 Hub ID」的 JSON 映射，
用 `LAYA_DEFAULT_MODEL` 设置默认名称。 [HTTP API](http-api.md) · [Router 参考](reference/router.md)
