# v6 本地运行与训练

以下命令从仓库根目录执行。运行环境需要 PyTorch 和项目的 `multimodal` 可选依赖。视频采样需要本地 FFmpeg 和 ffprobe。启用 HTTP 服务时安装 `serve`。

```bash
python -m pip install -e '.[multimodal,serve]'
```

训练数据放在 `multimodal-data/open-jev-omni-qwen-filtered-v6/native`，原始基座放在 `models/pretrained/Qwen3.5-4B`。v6 的可推理检查点在 `multimodal-runs/qwen4b-v6-fresh/sft/best`。LoRA 检查点读取其配置中记录的原始基座路径。 [架构与训练协议](multimodal-current.md) · [导出可独立加载的权重](multimodal-release.md)

## Python 推理

将示例的媒体路径换成本地图片或视频文件。可以用下面的输入测试候选评分和单框输出。

```python
from laya import MultimodalAgent

agent = MultimodalAgent("multimodal-runs/qwen4b-v6-fresh/sft/best", device="cpu")
answers = agent.predict(
    "Packing station",
    {
        "action": {
            "type": "choice",
            "instructions": "Decide what to do next.",
            "criteria": {"continue": "continue packing", "inspect": "inspect the package"},
        },
        "target": {"type": "box", "instructions": "the package on the table"},
    },
    image="images/package.jpg",
)["answers"]
print(answers["action"]["probabilities"])
print(answers["target"]["box"])
```

文本任务传入状态和题目即可。图片请求可以同时包含若干决策问题和单框问题。每个单框问题定位一个指定对象。输出归一化到图像尺寸。 [单框坐标与置信度](../grounding.md) · [Python API](../reference/multimodal.md)

## System One HTTP

```bash
LAYA_HOST=127.0.0.1 LAYA_PRELOAD=0 \
  LAYA_MULTIMODAL_MODEL="$PWD/multimodal-runs/qwen4b-v6-fresh/sft/best" laya-serve
```

文本 JSON 请求发到 `POST /v1/systemone`，使用 `"model": "multimodal"` 选中本地检查点。图片与单框也走这个端点。通过 multipart 传图时，JSON 中 `assets` 的 `part` 对应上传文件的字段名，并发送图片能力请求头。

```bash
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'TypeSafe-Danger-Zone-Images: true' \
  -F 'request={"model":"multimodal","state":["scene","Packing station"],"assets":{"scene":{"type":"image","part":"photo"}},"questions":{"target":{"type":"box","instructions":"the package on the table"}}};type=application/json' \
  -F 'photo=@images/package.jpg;type=image/jpeg'
```

服务器返回 `model`、按原题目 ID 命名的 `answers` 和 `usage`。内联 JSON 图片资产也受图片接口的大小和路径约束。启用鉴权后由调用方传入配置的 Bearer 凭据。 [HTTP 字段与置信度](../http-api.md) · [客户端示例](../multimodal-clients.md)

<a name="audio-and-video"></a>

## 音频和视频

Qwen3.5 v6 使用无声的原生视频路径。Python 可以在状态中控制采样帧数和音轨策略。

```python
result = agent.predict(
    {"text": "A workbench recording", "video": "clips/workbench.mp4",
     "video_audio": False, "video_frames": 8, "video_encoding": "native"},
    {"event": {"type": "noul", "instructions": "Is a package opened during the clip?"}},
)
print(result["answers"]["event"]["noul"])
```

HTTP 视频或音频通过 `POST /v1/multimodal` 上传。JSON 例如将视频字节放入 `"video": {"base64":"..."}`，并同时提供 `state` 和 `questions`。端点每次接收一个原生媒体文件，视频帧数与音轨策略由所加载模型的配置决定。

Gemma 4 等支持原生音频的检查点可在 Python 中传音频路径，或将波形数组和 `sampling_rate` 一同传入。对应的训练记录及输出类型见[音频参考样例](multimodal-current-io-examples.md#gemma-audio)。 [基座支持矩阵](../multimodal.md#backbones-and-input-types)

## 复现 v6 两阶段训练

以下命令使用独立输出目录。将 `training_devices` 设为本机可用于这次复现的四张卡。正式 v6 运行的物理卡编号为 2、3、4、5。

```bash
training_devices=2,3,4,5
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$training_devices"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

base="$PWD/models/pretrained/Qwen3.5-4B"
data="$PWD/multimodal-data/open-jev-omni-qwen-filtered-v6/native"
recipes="$PWD/docs/project-readme/assets"
run="$PWD/multimodal-runs/qwen4b-v6-reproduction"

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/multimodal/train.py --visual-only --require-original-base \
  --model "$base" --data "$data" --train-split warmup \
  --config "$recipes/head-config.json" --out "$run/head"

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/multimodal/train.py --visual-only --qwen-training-delta \
  --model "$run/head/best" --data "$data" \
  --config "$recipes/sft-config.json" --out "$run/sft"
```

v6 的准入收据绑定本地 Qwen 处理器和 16,384 token 预算。开始前可在数据根目录的 `manifest.json` 核对文件 SHA-256。训练时每个输入仍按实际处理长度检查。 [实际配置与损失](multimodal-current.md#training) · [训练入口](https://github.com/2351548518/laya/blob/4f7e045725adf8c7c7067dfc404de1e54f68b05b/scripts/multimodal/train.py)

<a name="resume"></a>

## 续训与检查点

`head/best` 供联合 SFT 初始化。`sft/best` 供推理和导出。`sft/checkpoint_latest.json` 指向恢复优化器、采样顺序和训练状态的快照。用原 SFT 参数执行续训，将以下参数附到 SFT 命令末尾。

```bash
--resume "$run/sft"
```

恢复时保持原来的 `--model`、`--data`、`--config`、`--out` 和 rank 数。快照会核验训练文件、起始模型和配置。正式运行的进度记录在 `multimodal-runs/qwen4b-v6-fresh/sft.log`。

<a name="evaluate"></a>

## 校准与评测

离线评测可读取当前决策验证集和完整定位验证集。

```bash
python scripts/multimodal/evaluate.py \
  --model "$run/sft/best" --device cpu \
  --decisions "$data/decision_dev.jsonl" --grounding "$data/dev.jsonl" \
  --max-micro-tokens 16384 --micro-batch 1 --grounding-micro-batch 4 \
  --out "$run/dev-metrics.json"
```

训练中用于选模的定位评测固定抽取 512 条。横评时要统一定位样本、决策数据和聚合方式。命令行评测的 `harmonic_accuracy` 使用全体决策准确率，SFT 的 `harmonic` 选模分使用决策模态宏平均。

训练完成后用校准集拟合概率，再用决策验证集检查 NLL、ECE 与评分误差。结果和校准接受状态记在输出目录。

```bash
python scripts/multimodal/calibrate.py \
  --model "$run/sft/best" \
  --calibration "$data/calibration.jsonl" --validation "$data/decision_dev.jsonl" \
  --out "$run/final" --device cpu
```

逐套评测保留了数据哈希和运行配置。测试分片和来源保存在 v6 数据根目录的 `evaluation/manifest.json`。 [同集结果与可比性](multimodal-current.md#matched-development)
