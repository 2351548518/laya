# Native multimodal commands

These are reusable local commands. Source-specific collection, data preparation and historical
experiments belong in the ignored `_local/` workbench.

| Command | Purpose |
| --- | --- |
| `train.py` | Fresh head warmup and joint supervised LoRA/full adaptation |
| `calibrate.py` | Fit held-out calibration and check it on independent development records |
| `prepare_dataset.py` | Convert native JSONL and Kev records to a shared request and supervision layout |
| `evaluate.py` | Evaluate native decisions and single-box geometry |
| `prepare_systemone.py` | Export frozen text and image benchmarks for System One and Kev |
| `export.py` | Safely merge native LoRA, save all heads/projectors, and reload-check synthetic fixtures |
| `export_gguf.py` | Use a compatible local llama.cpp-jev converter and preserve decision tensors in FP32 |
| `quantize_gguf.py` | Create smaller language GGUFs and a Q8 visual projector; optionally test Q8 head matrices |

Install the native dependencies with `python -m pip install -e '.[multimodal]'`. The runtime
and training implementation live in `laya/`. Store local checkpoints in `models/` or
`multimodal-runs/`. Use a compatible `llama.cpp-jev` checkout built in `build-laya/` for GGUF
conversion and quantization. [Runtime setup](../../docs/multimodal-release.md#run-the-decision-server)

Export a frozen benchmark directory containing `manifest.json` and its verified JSONL files.

~~~bash
python scripts/multimodal/prepare_systemone.py \
  --source /path/to/frozen-benchmark --out /path/to/new/systemone
~~~

`systemone/` contains labelled request envelopes. Send the contained `request` as the JSON
body for text or the first multipart part for images. `media_files` supplies the image
parts from local storage. Labels and provenance stay in the envelope. Box targets are
remapped for images that need padding. Audio and video cases remain available in the
source benchmark directory.

The `kev/` JSONL files use Kev's `--data` format. For native Qwen text decisions, set
`LAYA_MULTIMODAL_MODEL` when serving and pass `--remote-model multimodal` to the local Kev
evaluator. [Image request contract](../../docs/http-api.md#image-assets-and-single-box-questions)

## Prepare and evaluate local splits

```sh
python scripts/multimodal/prepare_dataset.py --training \
  --split train=path/to/train.jsonl --split warmup=path/to/warmup.jsonl \
  --split dev=path/to/grounding-dev.jsonl --split decision_dev=path/to/decision-dev.jsonl \
  --out multimodal-data/example/native
python scripts/multimodal/evaluate.py --model path/to/checkpoint --device cuda:0 \
  --suite-manifest multimodal-data/example/native/manifest.json --suite decision_dev \
  --out multimodal-runs/example/decision-dev.json
```

The output manifest records hashes, counts, modalities, supervision types, and content usable
under the System One wire schema. Media paths are relative to the generated JSONL. The HTTP
benchmark exporter prepares capped PNG/JPEG assets where required.

## Export checkpoints

Example merged export:

```bash
python scripts/multimodal/export.py --checkpoint /path/to/native/lora/final \
  --out /path/to/new/merged --base-id Qwen/Qwen3.5-4B \
  --device cuda:0 --fixtures /path/to/synthetic-fixtures.json
```

Fixtures are a JSON list of `{id, state, questions}` cases. They contain synthetic local inputs,
not training or benchmark labels. Probability, selected-choice, box-coordinate and quality
drift checks must pass before the staged directory becomes the output directory.

Example GGUF export after merging:

```bash
python scripts/multimodal/export_gguf.py --checkpoint /path/to/new/merged \
  --llama-cpp /path/to/llama.cpp-jev --out /path/to/new/gguf \
  --name jev-multimodal-qwen3.5-4b --quantize Q8_0
```

The GGUF command writes BF16, optional quantized language weights, and the required visual
`mmproj-BF16.gguf`. It checks all decision/localization tensors stay FP32. Inference parity is
a separate release check: a structurally valid GGUF alone does not prove behavioral equivalence.

Mobile quantization starts from the floating-point export, without converting or merging again:

```bash
python scripts/multimodal/quantize_gguf.py \
  --model /path/to/gguf/jev-multimodal-qwen3.5-4b-BF16.gguf \
  --mmproj /path/to/gguf/mmproj-BF16.gguf \
  --llama-cpp /path/to/llama.cpp-jev --out /path/to/new/mobile-gguf \
  --variants Q4_K_M Q4_0 Q3_K_M
```

The default keeps all seven decision/localization tensors bitwise unchanged in FP32.
`--head-quantization` supports Q8_0, Q4_0, Q2_0 and Q1_0 for the three head matrices; normalization and
bias tensors remain FP32. These options create separately named experimental files and require
a fresh behavioral comparison. The command rejects already quantized inputs rather than
requantizing them. Phone compatibility and memory must be checked with the matching runtime.

For the minimum-size compatibility experiment:

```bash
python scripts/multimodal/quantize_gguf.py \
  --model /path/to/gguf/jev-multimodal-qwen3.5-4b-BF16.gguf \
  --mmproj /path/to/gguf/mmproj-BF16.gguf \
  --llama-cpp /path/to/llama.cpp-jev --out /path/to/new/minimum \
  --variants Q1_0 --pure --head-quantization Q1_0 --mmproj-quantization Q1_0
```

This produces about 639 MiB including vision for this model, but the measured decision accuracy
falls to 28.67% and box Acc@IoU >= 0.5 to 1.56%. It is an extreme experiment with severe capability loss.

IQ formats require an actually measured `--imatrix`; pure IQ also requires an explicit
`--token-embedding-type`, since text-only imatrix collection omits embedding-lookup data.
`--tensor-type 'REGEX=type'` permits higher-precision sensitive backbone tensors; the explicit
head precision rules take precedence. For example, the IQ3 recipe uses IQ3_XXS with Q8 heads,
`ssm_out.weight=q8_0`, `ssm_beta_alpha.weight=q8_0` and the measured training-only imatrix.
The release guide lists the exact published model pairs and development comparisons.
[Models and GGUF deployment](../../docs/multimodal-release.md)
