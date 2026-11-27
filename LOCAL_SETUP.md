# Local setup

Run the native decision engine with a trained Qwen3.5 or Gemma 4 checkpoint. Qwen supports
text, images and visual video frames. A trained Gemma 4 checkpoint can also process audio.

## Linux and macOS

Use Python 3.11 or newer for native models. Install FFmpeg with your system package manager for video input.
On macOS, use an arm64 Python on Apple silicon.

```bash
./setup_laya.sh
source .venv-mm/bin/activate
```

The setup command installs native inference, training and HTTP dependencies. It downloads the
merged Qwen3.5-4B checkpoint to `models/jev-multimodal-qwen3.5-4b`. Use `--skip-models` if
you already have weights. Use `TORCH_INDEX_URL` to select a PyTorch wheel index.

```bash
export LAYA_MULTIMODAL_MODEL="$PWD/models/jev-multimodal-qwen3.5-4b"
laya-multimodal --device cpu --preset triage "I was charged twice."
LAYA_DEVICE=cpu laya-multimodal-serve
```

Set `LAYA_DEVICE=cuda:0` for an NVIDIA GPU or `LAYA_DEVICE=mps` on Apple silicon. The in-process
CLI accepts `--device` and an explicit checkpoint path with `--multimodal`.

## Windows

Set `LAYA_MULTIMODAL_MODEL` to a local trained checkpoint directory. Run `start.bat` to install the
dependencies in `.venv-mm` and start the same HTTP service. The optional positional argument
sets the HTTP port.

## Validate and train

```bash
python verify/multimodal_smoke_test.py --help
python -m laya_multimodal.train_cli --help
```

The packaged trainer uses the files in the installed source checkout. Start a fresh run with a
new output directory. For a continuation, pass `--resume` and the original model, dataset,
configuration and GPU count. [Native training](docs/finetune.md) · [v6 tutorial](docs/project-readme/multimodal.md)

[Container deployment](docs/docker.md) · [Native clients](docs/multimodal-clients.md)
