# Local runtime checks

For the native decision model, run the [multimodal Notebook](../notebooks/laya_native_multimodal.ipynb)
with a trained local checkpoint. It exercises text, image localization, visual video and, for
Gemma 4, waveform input. The [native CLI and evaluation guide](../docs/multimodal.md#calibration-and-evaluation)
covers labelled decision and grounding metrics. For the HTTP path, use the
[SDK examples](../docs/multimodal-clients.md).

The Python scripts in this directory test the classic text checkpoints. With the three
checkpoint directories under `models/`, run from the repository root:

```sh
python verify/laya_smoke_test.py --models ./models --device cpu
```

The classic smoke test covers checkpoint loading, text routing and typed decisions.
[Classic local setup](../LOCAL_SETUP.md)
