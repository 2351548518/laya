# Run inference, training and evaluation

**English** · [简体中文](../../docs_zh/project-readme/multimodal.md)

Run these commands from the repository root. If you already have `.venv-mm`, activate it with
`source .venv-mm/bin/activate`. Video inputs need FFmpeg and ffprobe. Install the `serve` extra
for the HTTP examples.

```bash
python -m pip install -e '.[multimodal]'
python -c 'import torch, laya; print(torch.__version__); print(laya_multimodal.__file__)'
```

The import check prints the PyTorch version and the current checkout's Python package path.
Start with the CPU exercise, then try Python and HTTP with a real checkpoint.
[Tutorial home](index.md) · [Supported inputs](../multimodal.md#backbones-and-input-types)

<a name="cpu-walkthrough"></a>

## 1. Exercise both heads on CPU

The script constructs hidden states and runs the actual candidate scorer, box head, losses
and backpropagation. It needs the Python dependencies and source checkout.

```bash
CUDA_VISIBLE_DEVICES='' python docs/project-readme/assets/walkthrough.py
```

| Output field | Expected | Meaning |
| --- | --- | --- |
| `heads.logits_shape` | `[3, 3]` | Three decisions with up to three candidates each |
| `heads.box_shape` | `[1, 1, 4]` | One image, one box, four normalized coordinates |
| `worked_calculations.choice_warmup_loss` | About `0.4267` | Soft cross-entropy and weighted Brier loss |
| `worked_calculations.score_parts.ordinal_rps` | About `0.005` | Cumulative probability error across levels |
| `worked_calculations.single_box_loss` | About `2.0132` | Weighted coordinate, GIoU and quality loss |

With local v6 data, add `--data` to verify both sample indexes. Expect
`verified_v6_examples = {"io": 30, "tasks": 192}`. The script seeks to each row and checks its ID,
question type, target and media references.

```bash
CUDA_VISIBLE_DEVICES='' python docs/project-readme/assets/walkthrough.py \
  --data multimodal-data/open-jev-omni-qwen-filtered-v6/native
```

[Follow a row](multimodal-current.md#input-to-model) · [Tensor shapes](multimodal-current.md#architecture) · [Worked losses](multimodal-current.md#worked-losses)

<a name="python-inference"></a>

## 2. Predict text and a single box in Python

Set `checkpoint` to a trained local v6 directory or the merged v1 safetensors. Confirm
`multimodal_config.json` is present. A LoRA directory reads the original backbone from the path
recorded in that file. [Checkpoint paths](multimodal-release.md#native-checkpoint)

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

Look at the answer for each question ID. `route.choice` is the selected key, `route.probabilities`
gives its distribution, and `refund.noul` is P(true). `urgency.score` is an expected level
index. For probabilities `[0.1, 0.5, 0.4]`, it is `0×0.1 + 1×0.5 + 2×0.4 = 1.3`.
`usage.output_tokens` is zero. A recorded CPU run using v1 returned `billing`, `0.9967`
and `0.8301` for the route, refund and urgency respectively.

Open `_local/project-readme-tutorial/prediction.png` to compare the blue box with the red square.
The box is normalized to the image. Multiply x coordinates by width and y by height.
`target.confidence` estimates IoU. Use a labelled suite for accuracy.
[Python API](../reference/multimodal.md) · [Single-box contract](../grounding.md)

<a name="systemone-http"></a>

## 3. Send a System One request

Start the service on the loopback address with the chosen checkpoint.

```bash
python -m pip install -e '.[multimodal,serve]'
checkpoint="multimodal-runs/qwen4b-v6-fresh/sft/best"
CUDA_VISIBLE_DEVICES='' LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=cpu LAYA_PRELOAD=0 \
  LAYA_MULTIMODAL_MODEL="$PWD/$checkpoint" laya-multimodal-serve
```

From a second terminal, check liveness and send a text decision. The first prediction loads
the checkpoint when lazy loading is enabled.

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

Read `model`, `answers.route` and `usage`. Single requests default to the TypeSafe response
shape, with TypeSafe confidence for Choice and Score and P(true) for Noul.
[Response format](../http-api.md#strict-jev-contract-laya_jev_strict)

Use the generated PNG in a multipart request. The `scene` reference in the state resolves to
`assets.scene.part` and then the file part named `photo`.

```bash
curl -fsS --noproxy 127.0.0.1 http://127.0.0.1:8000/v1/systemone \
  -H 'TypeSafe-Danger-Zone-Images: true' \
  -F 'request={"model":"multimodal","state":["scene","Find the object"],"assets":{"scene":{"type":"image","part":"photo"}},"questions":{"target":{"type":"box","instructions":"the red square"}}};type=application/json' \
  -F 'photo=@_local/project-readme-tutorial/scene.png;type=image/png'
```

Check `answers.target.box`, `answers.target.confidence` and `usage.output_tokens`. Each PNG or
JPEG side must be between 280 and 1,568 pixels. Send a bearer token on a configured deployment.
[Image assets](../http-api.md#image-assets-and-single-box-questions) · [SDK examples](../multimodal-clients.md)

<a name="audio-and-video"></a>

## 4. Use video or audio

Qwen3.5 reads visual frames and timestamps. Set `video_audio=False` and adjust `video_frames`.
Use your own short clip and checkpoint paths.

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

The answer is a probability. Inspect the sampled frame times when a relevant event is brief.
To send a clip via HTTP, put base64 file bytes in `video.base64` on `POST /v1/multimodal`.

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

The service uses its checkpoint's frame and audio settings. For waveform decisions, load a
trained Gemma audio checkpoint. In Python pass `audio="clips/speech.wav"` or a waveform and
`sampling_rate`. In HTTP pass `audio.base64`. Start with a Noul question about speech, then try
a choice or rating. [Audio examples](multimodal-current-io-examples.md#gemma-audio) · [Media clients](../multimodal-clients.md)

<a name="reproduce"></a>

## 5. Reproduce the two v6 stages

The packaged `laya-multimodal train` and the repository training script use
`laya_multimodal/train.py`. [Training API](../reference/train.md)

### Prepare the original backbone, data and four GPUs

```bash
base="$PWD/models/pretrained/Qwen3.5-4B"
data="$PWD/multimodal-data/open-jev-omni-qwen-filtered-v6/native"
recipes="$PWD/docs/project-readme/assets"
run="$PWD/multimodal-runs/qwen4b-v6-reproduction"

ls -lh "$base/config.json" "$data/"{warmup,train,dev,decision_dev,calibration}.jsonl
nvidia-smi
```

`warmup.jsonl` contains 30,720 rows. `train.jsonl` has 1,500,000. Use `dev.jsonl` and
`decision_dev.jsonl` for selection and a separate `calibration.jsonl` after training. The
parent manifest records counts and checksums. The admitted Qwen rows use the original
processor and the 16,384-token context budget.

Set `training_devices` to four free physical GPU indices. The recorded run used 2, 3, 4, 5.
Use a fresh output directory for a new run and the original one for a continuation.

```bash
training_devices=2,3,4,5  # Select four free physical GPUs.
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$training_devices"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$run"
set -o pipefail
```

### Stage A · Warm up both heads

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module laya_multimodal.train --visual-only --require-original-base \
  --model "$base" --data "$data" --train-split warmup \
  --config "$recipes/head-config.json" --out "$run/head" 2>&1 | tee "$run/head.log"
```

The original backbone stays frozen. A global batch of 64 makes 480 optimizer updates.
Check `head/initialization.json` for `original_base: true` and `world_size: 4`.
The recorded run uses the selected `head/best` checkpoint for the next stage.
[Warmup recipe](assets/head-config.json) · [Training diagram](multimodal-current.md#training)

### Stage B · Joint LoRA fine-tuning

Start this once the warmup selection exists. The planned training budget is 23,438 updates.
Four ranks accumulate together over each global window.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module laya_multimodal.train --visual-only --qwen-training-delta \
  --model "$run/head/best" --data "$data" \
  --config "$recipes/sft-config.json" --out "$run/sft" 2>&1 | tee "$run/sft.log"
```

The optimizer updates language adapters, the visual merger and both heads.
`sft/initialization.json` records the warmup source. `checkpoint_latest.json` points to
recoverable training state. After completion, use `sft/training_report.json` to identify the
selected checkpoint. `sft/last` holds the final update.
[SFT recipe](assets/sft-config.json) · [One update](multimodal-current.md#optimizer-step)

<a name="resume"></a>

## 6. Read a run and resume training

Set `active_run` to the run you want to inspect.

```bash
active_run="$PWD/multimodal-runs/qwen4b-v6-fresh"
rg '^(grounding|[^ ]+ TRAIN) step=' "$active_run/sft.log" | tail -5
cat "$active_run/sft/checkpoint_latest.json"
```

New joint training logs start with an ISO 8601 timestamp in the machine's local time, with
an explicit UTC offset. The filter above also reads the archived v6 log.

```text
2026-11-28T13:46:08+08:00 TRAIN step=5180 loss=0.48978 elapsed=99270.1s ranks=4
```

- `step` counts completed optimizer updates. The planned denominator for this recipe is 23,438.
  Use the difference between two `elapsed` values to estimate the rate, including checkpoint
  saves and development runs within that interval.
- `loss` averages the training objective for the current global window.
- `DEV` holds development metrics as JSON. `EVAL` shows the decision evaluation progress
  for the current worker. The SFT recipe evaluates every 500 steps.
- The optimizer snapshot interval is 250 steps, with additional saves after evaluations
  and at the final step. The recoverable step can trail the most recent log entry.

```bash
python - "$active_run/sft.log" <<'PY'
import json
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
records = re.findall(r"^(?:GROUNDING DEV|(\d{4}-\S+) DEV) (\{.*\})$", text, re.MULTILINE)
if not records:
    print("The run has not logged a development evaluation yet.")
else:
    timestamp, payload = records[-1]
    metrics = json.loads(payload)
    if timestamp:
        print("Time:", timestamp)
    print("Step:", metrics["step"])
    print("Decision accuracy:", metrics["decisions"]["overall"]["accuracy"])
    print("Box Acc@IoU>=0.5:", metrics["acc_iou50"])
PY
```

To continue a stopped run, keep its original model, data, configuration, rank count and
output directory. `--model` still points to the SFT's original `head/best`. The snapshot
restores the actual step, model, optimizer, order and RNG states.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module laya_multimodal.train --visual-only --qwen-training-delta \
  --model "$run/head/best" --data "$data" \
  --config "$recipes/sft-config.json" --out "$run/sft" \
  --resume "$run/sft" 2>&1 | tee -a "$run/sft.log"
```

Look for the timestamped `RESUME step=...` record. Use a selected checkpoint for
prediction or export, and a snapshot for continuation.
[Checkpoint roles](multimodal-release.md) · [Matched dev results](multimodal-current.md#matched-development)

<a name="swanlab"></a>

### Record metrics from the training process

Install the optional extra in the training environment, then authenticate through the terminal.
[SwanLab login reference](https://docs.swanlab.cn/en/api/cli-swanlab-login.html)

```bash
python -m pip install -e '.[multimodal,swanlab]'
swanlab login --local
```

Add `--swanlab online --swanlab-project laya-multimodal-v6` to either stage. Here is the
SFT command with the same absolute paths and devices as above. `--swanlab-name` sets a
display name.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  --module laya_multimodal.train --visual-only --qwen-training-delta \
  --model "$run/head/best" --data "$data" \
  --config "$recipes/sft-config.json" --out "$run/sft" \
  --swanlab online --swanlab-project laya-multimodal-v6 2>&1 | tee "$run/sft.log"
```

Use the experiment link printed to the terminal. Rank zero records `train/loss`,
`train/gradient_norm`, window size, rate, optimizer group learning rates and estimated
remaining time. `dev` tracks the evaluation slices, box metrics and selected score. `final`
holds the endpoint evaluation. A continuation creates an experiment in the same group and
continues the original step numbering.

`--swanlab local` writes to the run's `swanlog/` directory. Open the dashboard with
`swanlab watch "$run/sft/swanlog"`. The offline records can also be synchronized after
network recovery. [Local dashboard](https://docs.swanlab.cn/en/api/cli-swanlab-watch.html) · [Sync](https://docs.swanlab.cn/en/api/cli-swanlab-sync.html)

<a name="evaluate"></a>

## 7. Evaluate and read results

Use the same `data` and `run` paths to evaluate the decision development split and the complete
box development split. The latter contains 23,269 records. Use an appropriate device for the
size of the evaluation.

```bash
python scripts/multimodal/evaluate.py \
  --model "$run/sft/best" --device cpu \
  --decisions "$data/decision_dev.jsonl" --grounding "$data/dev.jsonl" \
  --max-len 16384 --max-micro-tokens 16384 --micro-batch 1 --grounding-micro-batch 4 \
  --out "$run/dev-metrics.json"
```

| Report field | Meaning |
| --- | --- |
| `decisions.overall.accuracy` | Decision accuracy pooled across rows |
| `decisions.by_modality` and `decisions.by_type` | Per-input and per-output slices, with counts |
| `grounding.single_n`, `miou` and `acc_iou50` | Positive boxes, mean IoU and share above the 0.5 IoU threshold |
| `harmonic_accuracy` | Harmonic mean of pooled decision accuracy and box Acc@IoU≥0.5 |
| `datasets`, `max_len` and `video_audio` | Split hashes and inference conditions |

For decision accuracy, the predicted argmax must be among the highest-probability target
levels or options. Score MAE compares expected indexes. A prediction `[0.1, 0.5, 0.4]` and
target `[0, 0.6, 0.4]` both peak at 1, and their expected indexes 1.3 and 1.4 differ by 0.1.
Check NLL, Brier and ECE alongside accuracy.

Online selection uses a fixed 512-row grounding subset and modality-macro decision accuracy.
Report those measurements separately from full-split evaluation. For final testing, keep the
rows, sampling, budget and aggregation fixed for all models. The frozen v6 suites are listed
under the dataset root's `evaluation/manifest.json`.
[Matched measurements](multimodal-current.md#matched-development) · [Benchmark protocol](../benchmarks.md)

### Fit probability calibration on a separate split

```bash
python scripts/multimodal/calibrate.py \
  --model "$run/sft/best" \
  --calibration "$data/calibration.jsonl" --validation "$data/decision_dev.jsonl" \
  --out "$run/final" --device cpu
```

Read `final/calibration_report.json`. Inspect `accepted` and `failures`, then compare
`before` and `candidate` on NLL, ECE and Score error. When `accepted` is false,
the saved model retains its previous calibration. [Export and deployment](multimodal-release.md)

## Troubleshooting

| Symptom | Where to look |
| --- | --- |
| Import or service command fails | Activate the correct Python environment. Install the `serve` extra for HTTP. |
| LoRA checkpoint cannot find its backbone | Check `backbone` in `multimodal_config.json`. Export merged weights to move the model. |
| Image request returns 422 | Inspect `detail`, header, state reference, asset `part`, file name and dimensions. |
| A clip fails or misses an event | Use `ffprobe` and inspect `frame_times`. Use the visual-only path for Qwen. |
| Processor or context mismatch while training | Compare the admission records with the actual processor path and `max_len`. |
| Training log pauses near an evaluation | Compare `eval_every` and `checkpoint_every` with the latest optimizer log and snapshot. |
