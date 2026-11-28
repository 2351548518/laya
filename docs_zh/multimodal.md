# 原生多模态决策

[English](../docs/multimodal.md) · **简体中文**

`MultimodalAgent` 把文本和原生媒体转换成类型化答案。骨干处理输入，候选评分头或单框头
读取隐藏状态。未启用题目批处理时，每题进行一次骨干前向，生成的答案 token 数为零。
Qwen3.5-4B 已发布合并后的 safetensors 和适合决策的 GGUF。 [发布与部署](multimodal-release.md)

从仓库根目录安装运行和训练所需的扩展。

```bash
pip install -e '.[multimodal]'
```

<a name="backbones-and-input-types"></a>

## 基座与输入类型

| 基座 | 文本 | 图片 | 音频 | 视频 |
| --- | --- | --- | --- | --- |
| Gemma 4 E2B | 支持 | 支持 | 原生波形 | 有序采样帧，可带音轨 |
| Qwen3.5 | 支持 | 支持 | — | 原生帧与时间戳，仅画面 |

本地模型别名包括 `gemma4`、`qwen3.5-4b` 和 `qwen3.5-9b`。也可直接传基座或训练后的
检查点目录。示例配方使用已测试的 Qwen3.5-4B，资源占用随基座和原生输入长度变化。
从原始基座创建的候选评分头和定位头需要训练。LoRA 检查点还需读取其配置记录的原始权重。

<a name="local-prediction"></a>

## 本地推理

用下载的检查点回答同一图片的两道题。

```python
import laya_multimodal

agent = laya_multimodal.MultimodalAgent("models/jev-multimodal-qwen3.5-4b", device="cpu")
questions = {
    "action": {
        "type": "choice", "instructions": "Choose the next action.",
        "criteria": {"continue": "continue normally", "inspect": "inspect the situation"},
    },
    "review": {"type": "noul", "instructions": "Does this scene need human review?"},
}
result = agent.predict("Current scene", questions, image="scene.jpg")
print(result["answers"]["action"]["probabilities"])
print(result["answers"]["review"]["noul"])
```

`choice` 返回选项及候选分布，`noul` 返回条件成立的概率，`score` 返回有序等级的分布和
档位期望，`box` 在一张图片中定位一个被指代的物体。 [题型与置信度](questions-and-answers.md) · [单框定位](grounding.md)

支持音频的基座可接收音频路径，或波形和 `sampling_rate`。Qwen3.5 的视频只使用画面，
在状态参数中设 `video_audio=False`，JSON 中写 `false`。处理后超过 `max_len`
的请求会报预算错误。

同一状态的题目共享已解码图片、波形及采样帧。每道题仍有自己的提示。Qwen3.5 推理可在同一
状态的多个问题之间复用满足条件的视觉特征。运行时支持预测钩子、`predict_batch`、`Router`
和内联媒体 HTTP 接口。
[可运行示例](https://github.com/2351548518/laya/blob/main/examples/README.md) · [检查点冒烟测试](https://github.com/2351548518/laya/blob/main/verify/README.md) · [CLI](cli-mcp.md) · [Python API](reference/multimodal.md)

<a name="from-input-to-answer"></a>

## 从输入到答案

`laya_multimodal/models/multimodal.py` 把题目和输入交给基座处理器。对 `choice`、`noul`、
`score`，`candidate_positions` 找到插入媒体 token 后各候选标记所在位置。
`laya_multimodal/models/readout.py` 中的 `CandidateScorer` 将各候选状态与题目最终
状态投影、比较，softmax 输出概率。`continue` 和 `inspect` 若分别得到 0.2 和 0.8，
选择结果是 `inspect`。若是评分的 0 档和 1 档，则期望分为 0.8。

定位时，`GroundingHead` 从最后一个有效状态得到归一化单框及预计 IoU。
[维度与结构](project-readme/multimodal-current.md#architecture)

<a name="router-and-http"></a>

## 路由与 HTTP

把训练好的检查点交给 `Router`。以下代码接续上面的示例。

```python
router = laya_multimodal.Router(multimodal="models/jev-multimodal-qwen3.5-4b", device="cpu")
result = router.predict("Current scene", questions, image="scene.jpg")
```

HTTP 服务安装 `.[multimodal,serve]`，设置 `LAYA_MULTIMODAL_MODEL`，
运行 `laya-multimodal-serve`。文本和命名图片使用 `POST /v1/systemone`。`assets`
可在状态、题目指令或候选说明中绑定图片。原生文本请求可指定 `"model": "multimodal"`。
内联音频、视频字节用 `POST /v1/multimodal`，可用模态和音轨行为由加载的检查点决定。
[HTTP API](http-api.md) · [媒体客户端](multimodal-clients.md)

Qwen3.5 的 Python 文本请求可通过 `batch_size=4` 把兼容题目作为独立样本合并到一次前向，
同时受补齐后 token 预算约束。

<a name="local-jsonl-data-loading"></a>

## 本地 JSONL 加载

`load_rows` 接收使用 System One `request` 结构的本地 JSONL，监督标签写在同一行的外层。
联合训练使用这一结构，每行一道题。两种加载器均从 JSONL 文件目录解析相对路径。平铺字段的
原生 JSONL 和问题内含 `label` 的 Kev 文本也可直接读取。

`request.state` 的图片占位符指向 `request.assets`，`media_files` 将命名 part 映射到
磁盘文件。图片占位符在前、文本在后的数组形成图像优先的提示。发送 multipart HTTP 时，把
文件按同名 part 附上并加图片请求头。

```json
{"id":"scene-1","group":"episode-1","request":{"model":"multimodal","state":["scene","Current scene"],"assets":{"scene":{"type":"image","part":"picture"}},"questions":{"action":{"type":"choice","instructions":"Choose the next action.","criteria":{"continue":"continue normally","inspect":"inspect the situation"}}}},"media_files":{"picture":"images/scene.jpg"},"expected":{"action":"inspect"}}
```

`expected` 可提供候选标签、布尔值、零基档位或 Box 标签。含来源概率分布时使用 `gold`，
它优先于同题的 `expected`。Box 标注的 `confidence` 留为 `null`，交给模型学习质量分。

```json
{"gold":{"action":{"probabilities":{"continue":0.2,"inspect":0.8}}}}
```

```json
{"expected":{"target":{"type":"box","box":{"left":0.1,"top":0.2,"right":0.5,"bottom":0.6},"confidence":null}}}
```

转换脚本在请求外写入 `option_orders`。Choice 在接口中保持显示顺序。Noul 和 Score 使用
接口约定的次序，训练加载时恢复原顺序。音频、视频和超过 10 档的评分保留本地格式和媒体控制字段。
可用 `prepare_systemone.py` 准备用于 HTTP 调用的文本与图片评测数据。
[数据脚本](https://github.com/2351548518/laya/tree/main/scripts/multimodal)

保留来源与组标识，按组和媒体划分训练、验证、校准与测试。数据准备的本机记录放在 `_local/`。
公开数据另有注明来源的版本。 [公开数据卡](https://huggingface.co/datasets/ZeeoRe/open-jev-omni-public)

联合训练目录包括 `train.jsonl`、单框开发 `dev.jsonl` 和决策开发 `decision_dev.jsonl`。
预热时用 `--train-split warmup` 选择 `warmup.jsonl`，开发集共用。已准入的 Qwen 行在
`training_admission` 中记录处理器、原生 token 数和上下文预算，用与训练一致的配置。
本地文件没有准入记录时可用 `--allow-unadmitted`，仍检查模态和 `max_len`。
`max_micro_tokens` 约束多条记录拼出的微批。单条请求可独立使用至 `max_len`。

<a name="head-warmup-and-lora-training"></a>

## 头部预热与 LoRA 训练

`laya-multimodal train` 运行联合训练。预热冻结原始骨干，训练决策和单框读出头。
LoRA 从预热选中的检查点开始，建立新优化器和适配器。选中的路径见 `training_report.json`。
[两阶段命令、损失与显存设置](finetune.md) · [v6 配方](project-readme/multimodal.md#reproduce)

<a name="swanlab-reporting"></a>

### SwanLab 记录

联合训练可在 rank 0 记录本地或在线指标。
[安装与监控](finetune.md#resume-and-monitor) · [SwanLab 教程](project-readme/multimodal.md#swanlab)

<a name="calibration-and-evaluation"></a>

## 校准与评测

在校准分片上拟合温度，再用另一份开发集检查。从训练报告读取选中的检查点，并准备一份与
开发集格式相同的独立 `calibration` 分片。

```bash
checkpoint=$(python -c 'import json, sys; print(json.load(sys.stdin)["checkpoint"])' \
  < multimodal-runs/example/sft/training_report.json)
python scripts/multimodal/calibrate.py --model "$checkpoint" \
  --calibration multimodal-data/example/native/calibration.jsonl \
  --validation multimodal-data/example/native/decision_dev.jsonl \
  --out multimodal-runs/example/final
```

校准验收同时检查整体及各模态的退化。未通过时保留原概率。校准集和最终测试集应独立。
启用有序评分通道后，按模态和档位数应用校准。缺少该通道时使用对应模态温度，或在没有
评分拟合的模态上使用原分布。

冻结检查点后，用清单指定的决策与单框样本评测。原生评测器输出准确率、校准指标、标准化评分
MAE 和 RPS，也提供记录中存在的模态、类型、任务和来源切片。串行评测可用
`--video-cache-size 24` 复用解码结果。共用采样设置的同一视频请求可一起处理，最后恢复数据顺序。
[准确率与延迟](benchmarks.md)

<a name="source-and-local-workspace-layout"></a>

## 源码与本地目录

运行和训练模块在 `laya_multimodal/`，通用命令在 `scripts/multimodal/`，入门配方在
`configs/multimodal/`。原生多模态 CI 在 CPU 上运行小型本地样本。

本地数据、权重和训练输出分别位于 `multimodal-data/`、`models/` 和 `multimodal-runs/`。
私有准备脚本、对比实验和日志放在 Git 忽略的 `_local/`。
