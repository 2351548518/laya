# 可选的 Vega 读取器

[English](../vegaml.md) · **简体中文**

安装 `.[vega]` 可使用上游 Vega 读取器。需要 Vega 拟合后的评分头或保形校准时，显式构造
`VegaAgent`。

```python
from laya_multimodal import VegaAgent

reader = VegaAgent(vegaml=True, device="cpu")
questions = {"direct": {"type": "noul", "instructions": "Is a refund requested?"}}
print(reader.predict("Please refund the duplicate charge.", questions)["answers"])
```

离线时用 `vegaml=` 传入训练好的本地目录。在 `Router(vega=reader)` 中注册并以
`model="vega"` 选择它。Qwen3.5 和 Gemma 4 决策检查点走原生运行时。
[检查点路由](routing.md) · [Vega 拟合示例](https://github.com/2351548518/laya/blob/main/examples/fit_and_calibrate_vega.py)
