# Models and GGUF deployment

The published checkpoint is the completed Qwen3.5-4B v1 run, trained on text, images and
visual-only video. The native release has its LoRA merged and contains the trained visual
merger, candidate scorer and single-box head. Load it with `laya.MultimodalAgent`.
[Native model and runtime requirements](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b)

## Select a GGUF pair

Use the language GGUF for text inference. For images and video, pair it with the matching
`mmproj` file, which contains the visual encoder and trained merger. The table reports the
combined size on disk.

| Selected file | Size | Distillation rows | Original H | Selected H on prior runtime | Selected H after runtime fix |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | 4.811 GiB | 10,496 | 76.80% | 78.22% | Not rerun |
| Q4_K_M, FP32 heads | 2.902 GiB | 10,496 | 75.57% | 77.53% | **77.83%** |
| Q4_K_M, Q8 heads | 2.897 GiB | 10,496 | 75.57% | 77.19% | Not rerun |
| Q4_0 | 2.718 GiB | 10,496 | 77.04% | 77.13% | Not rerun |
| Q3_K_M | 2.489 GiB | 744 | 74.37% | 75.77% | Not rerun |
| IQ3_XXS / SSM8 | 2.150 GiB | 10,496 | 72.11% | 72.85% | Not rerun |
| IQ2_XS / SSM4 | 1.840 GiB | 744 | 71.41% | 75.58% | **75.92%** |
| Q1_0, size experiment | 0.624 GiB | 744 | 2.96% | 2.98% | Not rerun |

The selected GGUF heads use teacher distillation with training-time head QAT disabled.
H is the harmonic mean of decision accuracy and box accuracy at IoU ≥ 0.5. The Q4_K_M and
IQ2_XS / SSM4 runtime reruns each used 900 decisions and 128 positive single boxes from the
model-selection development set. The fix recovered nine WebM inputs per model, preserved
all 1,019 previously valid answers and produced the same box accuracy. The model card lists
the exact projector pairs and measured desktop memory.
[GGUF model card](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF) ·
[Runtime evaluation](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF/blob/main/recovery/runtime-fixed-evaluation.json)

## Run the decision server

Build the local llama.cpp-jev fork at the tested revision and apply the runtime patch used in
the reruns.
Install a C++ compiler and CMake. For video decoding, put `ffmpeg` and `ffprobe` on PATH.

```bash
git clone https://github.com/2351548518/llama.cpp-jev.git
git -C llama.cpp-jev checkout --detach c649d0fcca794985df25370f89b2aebffc41b23c
curl -fL https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF/resolve/cb71291936b718f2b9e39fe627f000c52c577655/recovery/runtime-fixes.patch -o runtime-fixes.patch
git -C llama.cpp-jev apply --check ../runtime-fixes.patch
git -C llama.cpp-jev apply ../runtime-fixes.patch
cmake -S llama.cpp-jev -B llama.cpp-jev/build-laya -DCMAKE_BUILD_TYPE=Release -DMTMD_VIDEO=ON
cmake --build llama.cpp-jev/build-laya --target llama-server llama-quantize -j 4
```

Download the IQ2_XS / SSM4 pair below and start the server on localhost. To run Q4_K_M, use
`recovery/jev-multimodal-qwen3.5-4b-Q4_K_M.gguf` with `mmproj-Q8_0.gguf`.

```bash
hf download ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF \
  recovery/jev-multimodal-qwen3.5-4b-IQ2_XS-SSM4-heads-Q8_0.gguf \
  mmproj-Q4_0.gguf --local-dir models/jev-compact

llama.cpp-jev/build-laya/bin/llama-server \
  -m models/jev-compact/recovery/jev-multimodal-qwen3.5-4b-IQ2_XS-SSM4-heads-Q8_0.gguf \
  --mmproj models/jev-compact/mmproj-Q4_0.gguf \
  --host 127.0.0.1 --port 8080 -ngl 0 -c 2048 -b 256 -ub 128 -np 1
```

```bash
curl -fsS http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice for the same invoice.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {"billing": "invoices and refunds", "technical": "software issues"}
    }
  }
}'
```

For images or visual-only video, send local media bytes as a data URL in `files`. See the
runtime guide for the request shape and the model card for mobile integration and desktop
memory measurements. [llama.cpp-jev](https://github.com/2351548518/llama.cpp-jev) ·
[Mobile integration](https://huggingface.co/ZeeoRe/jev-multimodal-qwen3.5-4b-GGUF/blob/main/mobile/README.md)

## Export a local checkpoint

Merge a native LoRA checkpoint with `export.py`, convert the merged Qwen checkpoint with
`export_gguf.py`, and use `quantize_gguf.py` to create smaller profiles. The commands include
structure and checksum checks. Use evaluation fixtures and the target runtime to measure
behavior after conversion. [Export and quantization commands](https://github.com/2351548518/laya/tree/main/scripts/multimodal)
