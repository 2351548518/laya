# Laya notebooks and training examples

| Example | Model and workflow |
| --- | --- |
| [Native multimodal decisions](laya_native_multimodal.ipynb) | Local Qwen3.5 or Gemma 4 checkpoint, typed decisions and one image box, plus a small JSONL example |
| [Typed decisions on Kaggle](laya_finetune_typed_decisions_2xT4_kaggle.ipynb) | Classic text encoder, RLCD and calibration on two T4 GPUs |
| [Typed decisions on Apple silicon](laya_finetune_typed_decisions_mps.py) | Classic text encoder training with MPS |

Start the native notebook from the repository root using a Jupyter kernel with
`.[multimodal]` installed. Set `CHECKPOINT` in the first code cell to your local trained
checkpoint. The notebook creates small example files and a JSONL under
`multimodal-runs/notebook-example/`. It can produce the files offline before you download
weights. The released Qwen checkpoint runs the text, image and visual video cells. An
audio-capable Gemma 4 checkpoint also runs the waveform example.
[Native training and setup](../docs/multimodal.md)
