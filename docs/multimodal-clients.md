# Native media from Python and TypeScript

Run `laya-serve` on your machine with a trained native checkpoint. The server process owns the
model and decodes the media. The two SDKs send file bytes to `POST /v1/multimodal`.

## Start the service

From the repository root, with the [native runtime installed](multimodal.md) and a
[checkpoint downloaded](multimodal-release.md), run:

```sh
python -m pip install -e '.[multimodal,serve]'
LAYA_HOST=127.0.0.1 LAYA_PRELOAD=0 \
  LAYA_MULTIMODAL_MODEL="$PWD/models/jev-multimodal-qwen3.5-4b" laya-serve
```

The first request loads the checkpoint when `LAYA_PRELOAD=0`. If you set `LAYA_API_KEY`,
pass that bearer token to the SDK as `api_key` or `apiKey`. [Server settings](http-api.md)

On Windows PowerShell, select your checkpoint and start the launcher.

```powershell
$env:LAYA_MULTIMODAL_MODEL = "$PWD\models\jev-multimodal-qwen3.5-4b"
.\start.bat --multimodal
```

The launcher uses `.venv-mm` and installs CPU PyTorch on first use. Set `TORCH_INDEX_URL` before
creating the environment to choose a different PyTorch wheel index. Use
`.\start.bat --multimodal 8080` for a different port.

## Python

Install the client with `python -m pip install ./sdk/python` in the application environment.

```python
from pathlib import Path
from laya_client import LayaClient

client = LayaClient(base_url="http://127.0.0.1:8000")
prediction = client.predict_multimodal(
    {"target": {"type": "box", "instructions": "the red cup"}},
    image=Path("scene.jpg").read_bytes(),
    state="Current tabletop",
)
target = prediction["answers"]["target"]
print(target["box"], target["confidence"])
```

## TypeScript

Build or install the package using the [SDK setup guide](https://github.com/2351548518/laya/tree/main/sdk/typescript).

```ts
import { readFile } from "node:fs/promises";
import { Laya } from "laya-client";

const client = new Laya({ baseURL: "http://127.0.0.1:8000" });
const prediction = await client.predictMultimodal(
  { image: await readFile("scene.jpg") },
  { target: { type: "box", instructions: "the red cup" } },
  { state: "Current tabletop" },
);
console.log(prediction.answers.target.box, prediction.answers.target.confidence);
```

Pass exactly one nonempty `image`, `audio`, or `video` file. Optional `state` supplies JSON
context, such as a text instruction. Both SDKs encode media bytes as base64 in the request.
The TypeScript client also accepts a browser `File` converted with
`new Uint8Array(await file.arrayBuffer())`.

The released Qwen3.5 model supports images and visual video frames. For waveform decisions,
serve a trained Gemma 4 E2B checkpoint and send audio file bytes. Video decoding requires
`ffmpeg` and `ffprobe` on the server.

The server caps the complete JSON body at 2 MiB, including the base64 overhead. Use short clips
for this endpoint or use the in-process Python API for larger local files.

The endpoint accepts `choice`, `noul`, `score`, and `box` questions. `box` describes one object
in an image and returns `top`, `left`, `bottom`, and `right` in `[0, 1]`, plus a localization
quality estimate. `usage.forward_passes` counts the executed questions. The service enforces
its request and token budgets. [Single-box grounding](grounding.md)

For pure text, use the SDK's `predict` or `predictBatch`/`predict_batch` on the text endpoints.
For in-process text and media inference with the same native checkpoint, use
`MultimodalAgent`. [Native Python usage](multimodal.md#local-prediction)

## HTTP without an SDK

Send a JSON object to `POST /v1/multimodal`. Encode the original file bytes with standard base64
and put the string in the `base64` property of the corresponding media field. For example:

```json
{
  "image": {"base64": "<base64 of scene.jpg>"},
  "questions": {"target": {"type": "box", "instructions": "the red cup"}}
}
```

Save the filled request as `request.json` and submit it with your HTTP client.

```sh
curl -fsS http://127.0.0.1:8000/v1/multimodal \
  -H 'Content-Type: application/json' --data-binary @request.json
```

Send `Authorization: Bearer …` when server authentication is configured. For a text decision
using the same checkpoint, send `state`, `questions` and `"model": "multimodal"` to
`POST /v1/systemone`. [HTTP API](http-api.md)
