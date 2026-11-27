# Checkpoint integrity

Load a trusted native checkpoint. The runtime uses safetensors for model weights and verifies
an optional SHA-256 map before loading artifacts. It disables Transformers remote code loading.

## Pin a revision

Set `LAYA_REVISION=reviewed` to use the recorded release commit for the published native Qwen
checkpoint. Use an explicit commit for a different Hub checkpoint. Set `revision=` on `load`
to override the environment. [Release files](multimodal-release.md)

## Verify artifact digests

Provide `expected_sha256` to `load` or set `LAYA_SHA256_DIGESTS` to a JSON mapping from
checkpoint-relative paths to hex digests. Include the metadata, model and processor files you
reviewed. Hash the actual files you will serve. An SHA-256 map distributed with untrusted weights
does not establish their provenance.

```python
from laya import load

agent = load("models/jev-multimodal-qwen3.5-4b",
             expected_sha256={"decision_head.safetensors": "<reviewed SHA-256 digest>"})
```

For a multi-model service, use a map keyed by registered checkpoint name. Store bearer and Hub
tokens separately. Bind to loopback or a trusted private network, and use auth and a TLS reverse
proxy for access outside that network. [HTTP controls](http-api.md) · [Docker deployment](docker.md)

`/health` is a liveness check. With an API key configured, send the
`Authorization: Bearer …` header to see the loaded checkpoint and device details.
