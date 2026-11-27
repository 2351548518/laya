# Laya notebooks and training examples

| Example | Model and workflow |
| --- | --- |
| [Native multimodal decisions](laya_native_multimodal.ipynb) | Local Qwen3.5 or Gemma 4 checkpoint, typed decisions and one image box, plus a small JSONL example |

Start the native notebook from the repository root using a Jupyter kernel with
`.[multimodal]` installed. Set `CHECKPOINT` in the first code cell to your local trained
checkpoint. The notebook creates small example files and a JSONL under
`multimodal-runs/notebook-example/`. It can produce the files offline before you download
weights. The released Qwen checkpoint runs the text, image and visual video cells. An
audio-capable Gemma 4 checkpoint also runs the waveform example.
[Native training and setup](../docs/multimodal.md)

The classic training examples are available in the previous revision.
[Kaggle notebook](https://github.com/2351548518/laya/blob/21167295a16ec64de8b68999e17c37fb9a5e8706/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) · [Apple silicon script](https://github.com/2351548518/laya/blob/21167295a16ec64de8b68999e17c37fb9a5e8706/notebooks/laya_finetune_typed_decisions_mps.py)
