# 动手运行、训练与评测

[English](../../project-readme/multimodal.md) · **简体中文**

从仓库根目录执行 Shell 命令。已有本地 `.venv-mm` 时，可以先用 `source .venv-mm/bin/activate` 激活它。首次配置环境可在激活的 Python 虚拟环境中安装项目。视频需要系统里的 FFmpeg 和 ffprobe，HTTP 示例需要 `serve` 依赖。

```bash
python -m pip install -e '.[multimodal]'
python -c 'import torch, laya; print(torch.__version__); print(laya_multimodal.__file__)'
```

第二条命令应打印 PyTorch 版本和当前仓库的 `laya_multimodal/__init__.py` 路径。首次阅读可以依次完成 CPU 演示、Python 推理和 HTTP 调用，理解数据格式后再展开四卡训练。 [教程首页](index.md) · [基座支持矩阵](../multimodal.md#backbones-and-input-types)

<a name="cpu-walkthrough"></a>

## 1. 在 CPU 上跑通两个头部

从这个练习开始可以快速确认安装，然后对照架构图和训练图。脚本用程序生成的隐藏状态，调用实际的候选评分头、单框定位头、损失函数和反向传播。运行这一节只需代码和 Python 依赖。

```bash
CUDA_VISIBLE_DEVICES='' python docs/project-readme/assets/walkthrough.py
```

读输出时先找以下字段。

| 字段 | 预期结果 | 含义 |
| --- | --- | --- |
| `heads.logits_shape` | `[3, 3]` | 三条决策请求，每条在批中补齐到三个候选。 |
| `heads.box_shape` | `[1, 1, 4]` | 一条定位请求、一个候选框、四个归一化坐标。 |
| `worked_calculations.choice_warmup_loss` | 约 `0.4267` | 选中目标的 soft CE 与加权 Brier 之和。 |
| `worked_calculations.score_parts.ordinal_rps` | 约 `0.005` | 评分档位的累积概率误差。 |
| `worked_calculations.single_box_loss` | 约 `2.0132` | L1、GIoU 和定位质量项的加权和。 |

有本地 v6 数据时，再加一次 `--data`，核对两份附录的样例索引。输出中的 `verified_v6_examples` 应为 `{"io": 30, "tasks": 192}`。脚本按字节位置逐条读取，检查记录 ID、题型、标签和媒体引用。

```bash
CUDA_VISIBLE_DEVICES='' python docs/project-readme/assets/walkthrough.py \
  --data multimodal-data/open-jev-omni-qwen-filtered-v6/native
```

接下来挑一条图像样本，沿图确认每个维度来自哪里，再用三个数字例子复算损失。 [样本进入模型](multimodal-current.md#input-to-model) · [模型架构](multimodal-current.md#architecture) · [损失计算](multimodal-current.md#worked-losses)

<a name="python-inference"></a>

## 2. 运行 Python 决策与单框定位

以下使用训练得到的 v6 原生检查点。也可以把 `checkpoint` 改成已发布的 v1 原生 Safetensors 目录。v6 的 LoRA 检查点还会从其配置记录的路径加载原始基座。先确认检查点里有 `multimodal_config.json`。 [检查点路径和下载方式](multimodal-release.md#native-checkpoint)

```bash
checkpoint="multimodal-runs/qwen4b-v6-fresh/sft/best"
test -s "$checkpoint/multimodal_config.json"
CUDA_VISIBLE_DEVICES='' python - "$checkpoint" <<'PY'
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw
from laya_multimodal import MultimodalAgent

agent = MultimodalAgent(sys.argv[1], device="cpu")
result = agent.predict(
    "I was charged twice for the same invoice and would like a refund.",
    {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {"billing": "invoices and refunds", "technical": "software issues"},
        },
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "urgency": {
            "type": "score",
            "instructions": "Rate the urgency of the request.",
            "criteria": ["routine", "time-sensitive", "critical"],
        },
    },
)
print(json.dumps(result["answers"], indent=2))
print("Text input tokens:", result["usage"]["input_tokens"])
print("Generated tokens:", result["usage"]["output_tokens"])

work = Path("_local/project-readme-tutorial")
work.mkdir(parents=True, exist_ok=True)
image = Image.new("RGB", (384, 384), "white")
ImageDraw.Draw(image).rectangle((96, 96, 288, 288), fill="red")
image.save(work / "scene.png")

question = {"target": {"type": "box", "instructions": "the red square"}}
target = agent.predict(image=image, questions=question)["answers"]["target"]
print(json.dumps(target, indent=2))
box = target["box"]
width, height = image.size
pixels = (box["left"] * width, box["top"] * height,
          box["right"] * width, box["bottom"] * height)
ImageDraw.Draw(image).rectangle(pixels, outline="blue", width=4)
image.save(work / "prediction.png")
PY
```

按题目 ID 对照结果。先看 `route.choice` 选中了哪个键，再读 `route.probabilities` 了解模型如何在两个选项间分配概率。`refund.noul` 是回答“是”的概率，`urgency.score` 是档位索引的期望，`urgency.legend` 给出档位含义。三档评分中，如果分布为 `[0.1, 0.5, 0.4]`，期望就是 `0×0.1 + 1×0.5 + 2×0.4 = 1.3`。`output_tokens=0` 表示模型直接从隐藏状态读出答案。

用已发布的 v1 原生权重在 CPU 运行这段例子时，`route.choice` 为 `billing`，`refund.noul` 为 `0.9967`，`urgency.score` 为 `0.8301`。换用 v6 检查点时，按相同字段读取它的结果。

打开 `_local/project-readme-tutorial/prediction.png`，用蓝色预测框和红色方块比较位置。`box` 的四个坐标在 `[0, 1]` 内，横坐标乘图像宽，纵坐标乘图像高。`target.confidence` 估计定位 IoU。该图可用于验证输出格式，任务准确率用带标注的评测集测量。 [Python API](../reference/multimodal.md) · [单框契约](../grounding.md)

<a name="systemone-http"></a>

## 3. 发送 System One HTTP 请求

在一个终端启动服务。这里使用 CPU 和环回地址，便于在同一台机器上调用。已有 `laya-multimodal-serve` 时可从设置 `checkpoint` 开始。

```bash
python -m pip install -e '.[multimodal,serve]'
checkpoint="multimodal-runs/qwen4b-v6-fresh/sft/best"
CUDA_VISIBLE_DEVICES='' LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=cpu LAYA_PRELOAD=0 \
  LAYA_MULTIMODAL_MODEL="$PWD/$checkpoint" laya-multimodal-serve
```

另开一个终端，从仓库根目录执行健康检查和文本请求。请求里的 `"model": "multimodal"` 选中刚才配置的原生模型。首次请求会加载检查点。

```bash
curl -fsS --noproxy 127.0.0.1 http://127.0.0.1:8000/health
curl -fsS --noproxy 127.0.0.1 http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @- <<'JSON'
{
  "model": "multimodal",
  "state": "I was charged twice for the same invoice.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {"billing": "invoices and refunds", "technical": "software issues"}
    }
  }
}
JSON
```

根对象中的 `model` 告诉你服务使用的模型，`answers.route` 对应请求中的同名题目，`usage` 记录输入和输出 token。单条请求默认使用 TypeSafe 输出。`choice` 和 `score` 的 `confidence` 按 TypeSafe 的定义计算，`noul` 直接返回真值概率。 [完整响应字段](../http-api.md#strict-jev-contract-laya_jev_strict)

图片还是发到 `/v1/systemone`。用第二步生成的 PNG，把 `state` 中的 `scene` 引用绑定至 `assets.scene`，再把它的 `part` 与 multipart 字段 `photo` 对应起来。

```bash
curl -fsS --noproxy 127.0.0.1 http://127.0.0.1:8000/v1/systemone \
  -H 'TypeSafe-Danger-Zone-Images: true' \
  -F 'request={"model":"multimodal","state":["scene","Find the object"],"assets":{"scene":{"type":"image","part":"photo"}},"questions":{"target":{"type":"box","instructions":"the red square"}}};type=application/json' \
  -F 'photo=@_local/project-readme-tutorial/scene.png;type=image/png'
```

查看 `answers.target.box`、`answers.target.confidence` 和 `usage.output_tokens`。单框请求引用一张图片，PNG 或 JPEG 的边长需在 280 至 1,568 像素之间。启用鉴权的服务还需要按配置发送 Bearer 凭据。 [图片资产协议](../http-api.md#image-assets-and-single-box-questions) · [客户端示例](../multimodal-clients.md)

<a name="audio-and-video"></a>

## 4. 换成视频或音频

v6 Qwen3.5 读取视频的画面和时间戳。将本地短视频路径传给 `video`，设置 `video_audio=False`。在 Python 里可以调整 `video_frames`。把以下片段中的检查点和视频路径改成自己的本地文件。

```python
from laya_multimodal import MultimodalAgent

agent = MultimodalAgent("multimodal-runs/qwen4b-v6-fresh/sft/best", device="cpu")
result = agent.predict(
    {"text": "A workbench recording", "video": "clips/workbench.mp4",
     "video_audio": False, "video_frames": 8, "video_encoding": "native"},
    {"event": {"type": "noul", "instructions": "Is a package opened during the clip?"}},
)
print(result["answers"]["event"]["noul"])
print(result["media"]["video"]["frame_times"])
```

第一行结果是真值概率，第二行记录实际采样时间。对视频输出有疑问时，先核对画面中的关键事件是否落在采样帧附近。

HTTP 视频或音频使用 `POST /v1/multimodal`。JSON 中将本地文件字节编码后放进 `video.base64` 或 `audio.base64`，同一请求提供 `state` 与 `questions`。例如向已启动的 Qwen 服务提交一段视觉视频。

```bash
python - clips/workbench.mp4 <<'PY' | curl -fsS --noproxy 127.0.0.1 http://127.0.0.1:8000/v1/multimodal \
    -H 'Content-Type: application/json' --data-binary @-
import base64
import json
from pathlib import Path
import sys

video = base64.b64encode(Path(sys.argv[1]).read_bytes()).decode("ascii")
print(json.dumps({
    "state": "A workbench recording",
    "video": {"base64": video},
    "questions": {"event": {"type": "noul", "instructions": "Is a package opened?"}},
}))
PY
```

HTTP 的采样帧数和音轨策略取自服务加载的检查点。做音频决策时，用训练过的 Gemma 4 等音频检查点，Python 传 `audio="clips/speech.wav"` 或传波形和 `sampling_rate`。HTTP 把文件放在 `audio.base64`。先做一道“是否有人说话”的 `noul` 判断，再依照相同契约尝试单选或评分。 [音频真实样例](multimodal-current-io-examples.md#gemma-audio) · [原生媒体客户端](../multimodal-clients.md)

<a name="reproduce"></a>

## 5. 复现 v6 两阶段训练

联合训练实现位于 `laya_multimodal/train.py`。仓库训练脚本和
`laya-multimodal train` 共用同一个命令入口。
[训练接口参考](../reference/train.md)

### 准备本地基座、数据和可用 GPU

```bash
base="$PWD/models/pretrained/Qwen3.5-4B"
data="$PWD/multimodal-data/open-jev-omni-qwen-filtered-v6/native"
recipes="$PWD/docs/project-readme/assets"
run="$PWD/multimodal-runs/qwen4b-v6-reproduction"

ls -lh "$base/config.json" "$data/"{warmup,train,dev,decision_dev,calibration}.jsonl
nvidia-smi
```

`warmup.jsonl` 有 30,720 条，用于训练两个头。`train.jsonl` 有 1,500,000 条，用于联合微调。`dev.jsonl` 和 `decision_dev.jsonl` 用于选模。概率校准留到训练完成后使用 `calibration.jsonl`。父目录的 `manifest.json` 保存分片哈希和数量。v6 的准入记录绑定原始 Qwen 处理器路径和 16,384 token 的预算，训练时还会检查实际处理长度。

把下面的 `training_devices` 换成这次复现可用的四张物理卡。v6 正式运行使用的是 2、3、4、5。第一次复现使用尚未占用的 `$run` 目录，续训沿用原目录。

```bash
training_devices=2,3,4,5  # Select four free physical GPUs.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$training_devices"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$run"
set -o pipefail
```

### 阶段 A：从原始 Qwen 初始化并预热头部

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module laya_multimodal.train_cli --visual-only --require-original-base \
  --model "$base" --data "$data" --train-split warmup \
  --config "$recipes/head-config.json" --out "$run/head" 2>&1 | tee "$run/head.log"
```

这一步从原始基座创建候选评分头和单框头，冻结骨干。全局 batch 为 64，共 480 个更新步。开始时查看 `head/initialization.json`，其中 `original_base` 为 `true`，`world_size` 为 4。完成后应看到 `head/best/multimodal_config.json`，这是下一阶段的起点。 [预热配置](../../project-readme/assets/head-config.json) · [训练方式](multimodal-current.md#training)

### 阶段 B：联合 LoRA 微调

在阶段 A 成功完成后运行。全局样本数除以 batch 后向上取整，计划共 23,438 步。四个 rank 负责同一个全局窗口，分别累积本卡微批的梯度。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module laya_multimodal.train_cli --visual-only --qwen-training-delta \
  --model "$run/head/best" --data "$data" \
  --config "$recipes/sft-config.json" --out "$run/sft" 2>&1 | tee "$run/sft.log"
```

此阶段使用新优化器更新语言 LoRA、视觉 merger 和两个头。开始时 `sft/initialization.json` 应指向 `head/best`。训练中查看 `sft/best` 的选模结果和 `checkpoint_latest.json` 的恢复点。结束后 `sft/training_report.json` 的 `status` 为 `complete`，`sft/last` 是最后一次保存的模型，`sft/best` 是按开发集规则选中的模型。 [SFT 配置](../../project-readme/assets/sft-config.json) · [一次参数更新](multimodal-current.md#optimizer-step) · [当前训练入口](../reference/train.md)

<a name="resume"></a>

## 6. 读进度、选检查点和续训

要查看正式 v6 的进度，给 `active_run` 使用以下目录。查看你复现的运行时，将它替换为自己的输出目录。

```bash
active_run="$PWD/multimodal-runs/qwen4b-v6-fresh"
rg '^grounding step=' "$active_run/sft.log" | tail -5
cat "$active_run/sft/checkpoint_latest.json"
```

- `step` 是已经完成的优化器更新。用 `step / 23438` 得到本轮 SFT 进度。如果只想估算剩余时间，用跨度较大的两行 `elapsed` 之差除以步数差，再乘以剩余步数。跨过开发集评测和保存的窗口会把这些耗时计入估算。
- `loss` 是当前全局窗口的平均训练损失。样本的任务、长度和难度会影响单步数值，趋势可用一段窗口的平均值看。
- `GROUNDING DEV` 是独立的开发集评测记录。SFT 每 500 步评测，检查 `decisions.overall.accuracy`、`decisions.by_modality` 和 `acc_iou50`。定位与决策同时用于选模。
- 优化器快照每 250 步保存，在开发集评测和最终步也会保存。最新指针给出可恢复的步数，期间已写到日志的步数可能更多。

查看最后一次开发集评测可只打印主要字段。

```bash
python - "$active_run/sft.log" <<'PY'
import json
from pathlib import Path
import sys

prefix = "GROUNDING DEV "
lines = Path(sys.argv[1]).read_text().splitlines()
line = next((line for line in reversed(lines) if line.startswith(prefix)), None)
if line is None:
    print("The run has not logged a development evaluation yet.")
else:
    metrics = json.loads(line.removeprefix(prefix))
    print("Step:", metrics["step"])
    print("Decision accuracy:", metrics["decisions"]["overall"]["accuracy"])
    print("Box Acc@IoU>=0.5:", metrics["acc_iou50"])
PY
```

当原运行已经停止且需续训时，使用原运行的变量、配置、设备数量和输出目录，在同一条 SFT 启动命令中加上 `--resume "$run/sft"`。恢复使用快照里的权重、优化器、采样顺序和随机状态。原 SFT 的 `--model` 仍指向它启动时的 `head/best`，用于核验起始模型。下面沿用第五步复现时设置的变量。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module laya_multimodal.train_cli --visual-only --qwen-training-delta \
  --model "$run/head/best" --data "$data" \
  --config "$recipes/sft-config.json" --out "$run/sft" \
  --resume "$run/sft" 2>&1 | tee -a "$run/sft.log"
```

开始后日志应出现 `Resumed joint training at optimizer step ...`。推理和导出选择 `sft/best` 或已完成校准的模型目录，续训选择包含优化器快照的运行目录。 [检查点用途](multimodal-release.md) · [选模与同集结果](multimodal-current.md#matched-development)

<a name="swanlab"></a>

### 启动训练时直接记录到 SwanLab

在训练使用的 Python 环境安装 SwanLab 可选依赖。首次连接账号时，在启动训练的目录执行 `swanlab login --local`，根据终端提示完成授权。 [SwanLab 账号设置](https://swanlab.cn/settings) · [登录命令](https://docs.swanlab.cn/en/api/cli-swanlab-login.html)

```bash
python -m pip install -e '.[multimodal,swanlab]'
swanlab login --local
```

在第五步的头部预热和联合 SFT 命令中追加 `--swanlab online --swanlab-project laya-multimodal-v6`。下面是 SFT 阶段的完整例子，沿用第五步定义的 CUDA 设备和绝对路径。 `--swanlab-name` 可为实验指定名字。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module laya_multimodal.train_cli --visual-only --qwen-training-delta \
  --model "$run/head/best" --data "$data" \
  --config "$recipes/sft-config.json" --out "$run/sft" \
  --swanlab online --swanlab-project laya-multimodal-v6 2>&1 | tee "$run/sft.log"
```

从项目根目录用 `torchrun --module laya_multimodal.train_cli` 启动下一次训练。新建输出目录，并用绝对路径指定基座、数据和配置。若 SwanLab 已在其他目录登录，可从该目录调用当前项目的训练脚本。

在终端打印的 SwanLab 实验链接中打开面板。主进程按优化器步直接记录 `train/loss`、`train/gradient_norm`、实际窗口大小、步速和预计剩余时间。`lr` 区域显示实际使用的学习率。`dev` 区域记录初始及各轮开发集评测，包含模态和题型切片、单框 IoU 和选模分。`final` 单独显示训练结束后的评测。预热和 SFT 按输出目录分组，`--resume` 时在同组创建新的续训实验，曲线沿用原优化器步数。

本地实验数据写到输出目录下的 `swanlog/`。使用 `--swanlab local` 可在自己的机器上记录，再用 `swanlab watch "$run/sft/swanlog"` 打开。在线初始化遇到暂时的网络故障时使用离线记录，连接恢复后可通过 SwanLab 同步。 [本地面板](https://docs.swanlab.cn/en/api/cli-swanlab-watch.html) · [同步离线记录](https://docs.swanlab.cn/en/api/cli-swanlab-sync.html)

### 补录已经启动的旧训练

需要给旧版本的训练补齐曲线时，使用 `monitor_swanlab.py` 从现有日志导入。默认一次性导入，旧训练仍在写日志时可加 `--follow`。下一次启动用上面的训练参数直接记录。

```bash
python scripts/multimodal/monitor_swanlab.py "$run" \
  --stage sft --mode online --project laya-multimodal-v6 --follow
```

<a name="evaluate"></a>

## 7. 评测并读结果

以下命令同时评测决策验证集和完整单框验证集。使用第五步定义的 `data` 和 `run`。如需评测正式运行，`run` 可指向该运行。完整定位分片有 23,269 条，CPU 评测适合低资源验证，完整横评需为推理时间留出预算。

```bash
python scripts/multimodal/evaluate.py \
  --model "$run/sft/best" --device cpu \
  --decisions "$data/decision_dev.jsonl" --grounding "$data/dev.jsonl" \
  --max-len 16384 --max-micro-tokens 16384 --micro-batch 1 --grounding-micro-batch 4 \
  --out "$run/dev-metrics.json"
```

| 在 JSON 中查看 | 怎样理解 |
| --- | --- |
| `decisions.overall.accuracy` | 所有决策样本合并后计算的准确率。 |
| `decisions.by_modality` 和 `decisions.by_type` | 展开文本、图像、视频及各决策类型，结合各组样本数寻找短板。 |
| `grounding.single_n`、`grounding.miou`、`grounding.acc_iou50` | 正框数量、平均 IoU 与 IoU 至少达到 0.5 的比例。 |
| `harmonic_accuracy` | 全体决策准确率与单框 `acc_iou50` 的调和平均。SFT 在线选模的决策项用按模态宏平均。 |
| `datasets`、`max_len`、`video_audio` | 保存输入分片哈希和相关推理条件，便于同集横评。 |

决策准确率检查预测概率最大的选项是否属于目标分布的最高概率档。评分题额外查看 `score_mae`，它衡量预测等级期望与目标等级期望的差。比如预测分布 `[0.1, 0.5, 0.4]` 和目标 `[0, 0.6, 0.4]` 都以等级 1 为最高概率档，这一题在准确率中计为正确，期望分数分别为 `1.3` 和 `1.4`，这一题的绝对误差为 `0.1`。概率质量可结合 NLL、Brier 和 ECE 判断。 [评分损失例子](multimodal-current.md#worked-losses)

在线选模的定位分片固定抽取 512 条。完整分片和抽样分片分别报告。若比较两个模型，把上述命令中的 `--model` 和 `--out` 换成另一个检查点及另一个报告路径，保持数据、采样范围、推理预算和聚合方法一致。v6 的冻结测试分片与来源记录在数据根目录的 `evaluation/manifest.json`。 [同集结果](multimodal-current.md#matched-development) · [基准评测](../benchmarks.md)

### 在独立分片上校准概率

```bash
python scripts/multimodal/calibrate.py \
  --model "$run/sft/best" \
  --calibration "$data/calibration.jsonl" --validation "$data/decision_dev.jsonl" \
  --out "$run/final" --device cpu
```

打开 `final/calibration_report.json`。先看 `accepted` 和 `failures`，再比较 `before` 与 `candidate` 的 NLL、ECE 和评分误差。报告里的 `candidate` 是候选校准方案，`accepted` 为 `false` 时输出模型保留原校准参数。有了可用的 `final` 检查点后，可导出或与相同分片上的未校准检查点比较。 [导出与部署](multimodal-release.md)

## 定位常见问题

| 看到的现象 | 从哪里查 |
| --- | --- |
| Python 找不到 `laya_multimodal`，或服务命令不可用 | 检查当前虚拟环境，用开头的导入命令定位项目路径。HTTP 需要 `serve` 依赖。 |
| LoRA 检查点无法加载基座 | 打开 `multimodal_config.json`，确认 `backbone` 所指本地基座仍可读。要移动模型，按发布教程导出完整权重。 |
| 图片请求返回 422 | 查看响应的 `detail`，核对图片请求头、`state` 中的引用、`assets` 的 `part`、multipart 字段名和图片大小。 |
| 视频打不开或时间点出乎预期 | 用 `ffprobe` 查看该文件。确认 Qwen 路径使用视觉视频，再根据响应里的 `frame_times` 查看采样位置。 |
| 训练报处理器或 token 预算不匹配 | 对照 v6 准入记录、基座处理器和配置的 `max_len`。复现时使用配套的训练数据和基座。 |
| 最后一条日志停留在评测附近 | 对照 `eval_every` 和 `checkpoint_every`，检查是否还在评测或保存，再结合训练进程与最新快照判断进度。 |

[图片契约](../http-api.md#image-assets-and-single-box-questions) · [复现准备](#reproduce) · [续训读法](#resume)
