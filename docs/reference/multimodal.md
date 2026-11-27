# MultimodalAgent

**English** · [简体中文](../zh-CN/reference/multimodal.md)

Load a trained native checkpoint with `laya_multimodal.MultimodalAgent(path, device="cpu")`. Use
`predict(state, questions)` for one input or `predict_batch(states, questions)` to apply
the same question definitions to each state in order. Each question has its own prompt and
forward pass. Results contain `answers`, `usage`, the input `modality` and decoded media metadata.

Pass local media paths in the state object or as keyword arguments to `predict`.
Video requests can set `video_frames` and `video_audio` in the state object. Native waveform
input needs an audio-capable backbone. [Formats and input budgets](../multimodal.md#local-prediction)

| Question | Fields used to read the result |
| --- | --- |
| `choice` | `choice` and `probabilities` keyed by the supplied candidate labels |
| `score` | `score`, `probabilities` and `legend` keyed by level index |
| `noul` | `noul` with the probability of yes |
| `box` | `box` with normalized `left`, `top`, `right`, `bottom`, and `confidence` with estimated IoU |

`usage.input_tokens` sums the processed input tokens across executed questions.
`usage.forward_passes` counts those questions. `usage.output_tokens` is zero.
[Single-box contract](../grounding.md#prediction-contract)

::: laya_multimodal.multimodal.MultimodalAgent
    options:
      members:
        - predict
        - predict_batch
        - save
