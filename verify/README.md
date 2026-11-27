# Check a native checkpoint

Run the smoke check before serving a checkpoint. It uses the local model processor and
validates typed decisions and single-box responses for the supplied files.

```bash
python verify/multimodal_smoke_test.py --help
```

For a running authenticated service, verify one native HTTP decision and inspect the loaded
checkpoint's device and revision.

```bash
python scripts/http_inference_smoke.py --api-key-file /path/to/bearer-key \
  --checkpoint-revision local --device cpu
```

The default request exercises the three text decision types. Pass `--request` with a System One
image asset request to validate a box, or combine it with `--route multimodal` for inline media.
Use the exact Hub revision in place of `local` for a pinned downloaded checkpoint.
[HTTP request formats](../docs/http-api.md)
