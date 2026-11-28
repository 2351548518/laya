# 从 Python 与 TypeScript 发送原生媒体

[English](../docs/multimodal-clients.md) · **简体中文**

启动加载了训练后检查点的 `laya-multimodal-serve`。Python 与 TypeScript SDK
通过 `POST /v1/multimodal` 发送文件字节，通过 `POST /v1/systemone` 发送命名图像资产。

<a name="start-the-service"></a>

## 启动服务

下载检查点后，在仓库根目录运行。 [检查点下载](multimodal-release.md)

```sh
python -m pip install -e '.[multimodal,serve]'
LAYA_HOST=127.0.0.1 LAYA_PRELOAD=0 \
  LAYA_MULTIMODAL_MODEL="$PWD/models/jev-multimodal-qwen3.5-4b" laya-multimodal-serve
```

`LAYA_PRELOAD=0` 让第一条请求触发加载。配置 `LAYA_API_KEY` 时，请把同一 Bearer 密钥设置为
SDK 的 `api_key` 或 `apiKey`。 [服务参数](http-api.md)

在 Windows PowerShell 中设置检查点路径后启动服务。

```powershell
$env:LAYA_MULTIMODAL_MODEL = "$PWD\models\jev-multimodal-qwen3.5-4b"
.\start.bat
```

启动器首次运行会使用 `.venv-mm` 并安装 CPU 版 PyTorch。创建环境前可用 `TORCH_INDEX_URL`
选择其他 wheel 索引。通过 `.\start.bat 8080` 可改变端口。

## Python

在应用的环境中运行 `python -m pip install ./sdk/python`。

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

构建或安装 TypeScript 客户端。 [SDK 安装](https://github.com/2351548518/laya/tree/main/sdk/typescript)

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

每个请求只提交一个非空的 `image`、`audio` 或 `video` 文件。用 `state` 附带文本上下文。
SDK 会把媒体字节编码成 base64。浏览器 `File` 可用 `new Uint8Array(await file.arrayBuffer())`
转为 TypeScript 客户端接受的字节数组。

已发布的 Qwen3.5 模型支持图片和视频画面。音频要选择经过训练的 Gemma 4 E2B 检查点，再传音频
文件字节。服务器用 `ffmpeg` 和 `ffprobe` 解码视频。整个 JSON 请求体上限为 2 MiB，
包含 base64 开销。较大的本地文件可以用进程内的 Python 接口。

接口支持 `choice`、`noul`、`score` 与 `box`。单框返回归一化的 `left`、`top`、`right`、
`bottom`，`confidence` 估计与目标框的重合度。`usage.forward_passes` 记录实际前向次数。
纯文本可调用 SDK 的 `predict` 或 `predictBatch`／`predict_batch`。
[单框定位](grounding.md) · [进程内推理](multimodal.md#local-prediction)

<a name="system-one-image-assets"></a>

## System One 图像资产

在状态、题目指令或选项说明中引用命名图片时，使用 Python 的 `predict_image_assets` 或
TypeScript 的 `predictImageAssets`。两者都会附上 `TypeSafe-Danger-Zone-Images: true`，
并返回按题目 ID 索引的答案与 token 计数。PNG 或 JPEG 的每条边需在 280 到 1,568 像素之间。

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

如果服务未提供定位质量，`box` 答案的 `confidence` 可为 `null`。
[System One HTTP 格式](http-api.md)

<a name="http-without-an-sdk"></a>

## 直接发送 HTTP 请求

向 `POST /v1/multimodal` 发送 JSON 对象，把 base64 文件字节填到相应媒体字段的 `base64`。

```json
{
  "image": {"base64": "<base64 of scene.jpg>"},
  "questions": {"target": {"type": "box", "instructions": "the red cup"}}
}
```

将填好的 JSON 保存为 `request.json` 后发送。

```sh
curl -fsS http://127.0.0.1:8000/v1/multimodal \
  -H 'Content-Type: application/json' --data-binary @request.json
```

启用认证时带上 `Authorization: Bearer …`。文本决策向 `POST /v1/systemone` 发送 `state`
和 `questions`。 [HTTP API](http-api.md) · [C++、.NET、Go、Java、Rust 示例](https://github.com/2351548518/laya/tree/main/examples/clients)
