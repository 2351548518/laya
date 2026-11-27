# Native media from Python and TypeScript

**English** · [简体中文](zh-CN/multimodal-clients.md)

Run `laya-multimodal-serve` with a trained native checkpoint. The Python and TypeScript SDKs send file
bytes to `POST /v1/multimodal`. Their image asset methods use `POST /v1/systemone`.

## Start the service

Use the downloaded checkpoint and run these commands from the repository root.
[Native checkpoint download](multimodal-release.md)

```sh
python -m pip install -e '.[multimodal,serve]'
LAYA_HOST=127.0.0.1 LAYA_PRELOAD=0 \
  LAYA_MULTIMODAL_MODEL="$PWD/models/jev-multimodal-qwen3.5-4b" laya-multimodal-serve
```

The first request loads the checkpoint when `LAYA_PRELOAD=0`. If you set `LAYA_API_KEY`,
pass that bearer token to the SDK as `api_key` or `apiKey`. [Server settings](http-api.md)

On Windows PowerShell, select your checkpoint and start the launcher.

```powershell
$env:LAYA_MULTIMODAL_MODEL = "$PWD\models\jev-multimodal-qwen3.5-4b"
.\start.bat
```

The launcher uses `.venv-mm` and installs CPU PyTorch on first use. Set `TORCH_INDEX_URL` before
creating the environment to choose a different PyTorch wheel index. Use
`.\start.bat 8080` for a different port.

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

Build or install the TypeScript client.
[SDK installation](https://github.com/2351548518/laya/tree/main/sdk/typescript)

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

Pass exactly one nonempty `image`, `audio`, or `video` file. Use `state` for optional context,
such as a text instruction. Both SDKs encode media bytes as base64 in the request.
The TypeScript client also accepts a browser `File` converted with
`new Uint8Array(await file.arrayBuffer())`.

The released Qwen3.5 model supports images and visual video frames. For waveform decisions,
serve a trained Gemma 4 E2B checkpoint and send audio file bytes. Video decoding requires
`ffmpeg` and `ffprobe` on the server.

The server caps the complete JSON body at 2 MiB, including the base64 overhead. Use short clips
for this endpoint or use the in-process Python API for larger local files.

The endpoint accepts `choice`, `noul`, `score`, and `box` questions. An image box answer contains
normalized `left`, `top`, `right`, and `bottom` coordinates. Its `confidence` estimates overlap
with the target. `usage.forward_passes` counts the executed questions.
[Single-box grounding](grounding.md)

For pure text, use the SDK's `predict` or `predictBatch`/`predict_batch` on the text endpoints.
For in-process text and media inference with the same native checkpoint, use
`MultimodalAgent`. [Native Python usage](multimodal.md#local-prediction)

## System One image assets

Use `predict_image_assets` in Python or `predictImageAssets` in TypeScript when a decision
references one or more named images in the state, instructions or criteria. Both methods
attach `TypeSafe-Danger-Zone-Images: true` and return named answers with token counts.
Each PNG or JPEG must measure between 280 and 1,568 pixels on each side.

```python
import base64
from pathlib import Path

scene = base64.b64encode(Path("scene.jpg").read_bytes()).decode("ascii")
prediction = client.predict_image_assets(
    ["scene-photo", "Current tabletop"],
    {"target": {"type": "box", "instructions": "Locate the red cup."}},
    {"scene-photo": {"type": "image", "format": "jpeg", "data_base64": scene}},
)
```

```ts
const scene = (await readFile("scene.jpg")).toString("base64");
const result = await client.predictImageAssets(
  ["scene-photo", "Current tabletop"],
  { target: { type: "box", instructions: "Locate the red cup." } },
  { "scene-photo": { type: "image", format: "jpeg", data_base64: scene } },
);
```

A Box answer can carry null confidence when the server omits localization quality.
[System One HTTP contract](http-api.md)

## HTTP without an SDK

Send a JSON object to `POST /v1/multimodal`. Put the base64-encoded file bytes in the media
field's `base64` property.

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
using the same checkpoint, send `state` and `questions` to
`POST /v1/systemone`. [HTTP API](http-api.md)

[C++, .NET, Go, Java and Rust examples](https://github.com/2351548518/laya/tree/main/examples/clients)
