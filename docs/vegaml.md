# Optional Vega reader

Install `.[vega]` to use the upstream multimodal Vega reader. Construct it explicitly when a
workflow needs Vega's fitted heads or conformal calibration.

```python
from laya import VegaAgent

reader = VegaAgent(vegaml=True, device="cpu")
questions = {"direct": {"type": "noul", "instructions": "Is a refund requested?"}}
print(reader.predict("Please refund the duplicate charge.", questions)["answers"])
```

Pass a trained local reader directory as `vegaml=` for offline use. Attach a built reader with
`Router(vega=reader)` and select `model="vega"`. Use the standard native runtime for Qwen3.5 and
Gemma 4 decision checkpoints. [Routing](routing.md) · [Vega fitting example](https://github.com/2351548518/laya/blob/main/examples/fit_and_calibrate_vega.py)
