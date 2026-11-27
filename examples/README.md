# Local examples

| Example | What it demonstrates |
| --- | --- |
| [native_multimodal.py](native_multimodal.py) | In-process text and media decisions and an image Box |
| [Client examples](clients/README.md) | C++, Go, Java, .NET and Rust requests to the local service |
| [http_route.py](http_route.py) | Inspect the configured checkpoint and run a text request over HTTP |
| [docker/quickstart.py](docker/quickstart.py) | Run a decision from a JSON file in a container |
| [hooks](hooks/) | Apply caching, logging and auditing to a prediction |
| [Integrations](../docs/use-cases.md) | Use typed decisions from agent frameworks |

From the repository root, run the in-process example with a trained checkpoint.

```bash
python examples/native_multimodal.py models/jev-multimodal-qwen3.5-4b --device cpu
```

Add `--image`, `--video` or an audio-capable checkpoint with `--audio`. [Setup](../LOCAL_SETUP.md)
