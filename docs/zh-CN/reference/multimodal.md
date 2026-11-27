# MultimodalAgent

[English](../../reference/multimodal.md) · **简体中文**

用 `laya_multimodal.MultimodalAgent(path, device="cpu")` 加载训练后的检查点。
`predict(state, questions)` 接收一个状态，`predict_batch(states, questions)` 将题目应用
到多个状态，按原顺序返回。`answers` 按题目 ID 索引，`usage` 记录 token 和前向次数。
每条结果的 `modality` 和 `media` 记录输入信息。

媒体可通过状态对象或关键词参数指定。`video_frames` 与 `video_audio` 控制视频，
原生波形需要支持音频的基座。 [输入格式](../multimodal.md#local-prediction)

| 题型 | 返回字段 |
| --- | --- |
| `choice` | `choice` 和按候选键索引的 `probabilities` |
| `score` | 期望档位 `score`，以及 `probabilities` 与 `legend` |
| `noul` | 条件为真的概率 `noul` |
| `box` | 归一化 `box` 和 IoU 估计 `confidence` |

`usage.input_tokens` 汇总实际处理的输入，`forward_passes` 统计骨干调用，`output_tokens` 为 0。
[单框接口](../grounding.md#prediction-contract)

::: laya_multimodal.multimodal.MultimodalAgent
    options:
      members:
        - predict
        - predict_batch
        - save
