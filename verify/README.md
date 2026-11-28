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

The default request exercises the three text decision types. To validate a box, pass `--request`
with a JSON System One request whose image asset has `format` and `data_base64` fields.
Use `--route multimodal` with a native JSON request for inline image, audio or video.
Use the exact Hub revision in place of `local` for a pinned downloaded checkpoint.
[HTTP request formats](../docs/http-api.md)
