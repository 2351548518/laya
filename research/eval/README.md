# Text decision diagnostics

Both commands use a trained native Qwen3.5 or Gemma 4 checkpoint. The scripts read option
logits from the same prompt preparation and head used by the native runtime.

```bash
export LAYA_MULTIMODAL_MODEL="$PWD/models/jev-multimodal-qwen3.5-4b"
python -m research.eval.laya_eval --device cpu --langs en,ja --out _local/massive-native.json
python -m research.eval.presentation_checks --device cpu --lang en,ja \
  --out _local/presentation-native.json
```

The MASSIVE harness scores the labelled intent test split and reports accuracy, macro F1,
calibration and per-case option probabilities. It uses the checkpoint's text calibration and
reports the resolved model revision. Control the sample size with `--per-lang` and keep the
seed, labels and option count fixed for comparisons.

The presentation checks use frozen text probes to measure slot bias and sensitivity to
different labels for the same option. They also compare native logits and public prediction
probabilities for each question type under test.

Historical reports under `research/results/` preserve their original configurations.
The archived reports and their reproduction notes are available in
[the original source revision](https://github.com/2351548518/laya/blob/21167295a16ec64de8b68999e17c37fb9a5e8706/research/eval/README.md).

Run the offline checks from the repository root.

```bash
python research/eval/test_laya_eval.py
python research/eval/test_presentation_checks.py
```
