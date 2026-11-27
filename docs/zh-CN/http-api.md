# HTTP API

[English](../http-api.md) · **简体中文**

`POST /v1/systemone` 按 TypeSafe System One 格式处理文本、图片决策和单框定位。默认使用
已发布的 Qwen3.5-4B 决策检查点。设置 `LAYA_MULTIMODAL_MODEL` 可选择其他训练后的检查点，
例如支持音频的 Gemma。音频和视频字节通过 `POST /v1/multimodal` 发送。
[TypeSafe API](https://docs.typesafe.ai/api/system-one) · [媒体示例](multimodal-clients.md)

Python HTTP 客户端提供同步单条、批量、媒体推理和存活检查。应用环境可仅安装 `sdk/python`，
由服务进程负责加载推理栈。 [Python SDK](https://github.com/2351548518/laya/tree/main/sdk/python)

```bash
python -m pip install -e '.[multimodal,serve]'
LAYA_HOST=127.0.0.1 laya-multimodal-serve
```

通过 `laya_multimodal.serve.create_app()` 构造 FastAPI 应用，也可用 `create_app(router)`
注入自定义路由器。

<a name="configuration"></a>

## 服务配置

| 环境变量 | 用途 | 默认值 |
| --- | --- | --- |
| `LAYA_HOST` | 监听地址 | `0.0.0.0` |
| `LAYA_PORT` | 监听端口 | `8000` |
| `LAYA_ROOT_PATH` | 反向代理公开的 URL 前缀 | 空 |
| `LAYA_DEVICE` | 检查点的 PyTorch 设备 | `auto` |
| `LAYA_MULTIMODAL_MODEL` | 训练后的目录或 Hub ID | 已发布的 Qwen3.5-4B |
| `LAYA_PRELOAD` | 启动时加载检查点 | `1` |
| `LAYA_MODELS` | 逗号分隔的预加载模型名 | 已注册检查点。指定原生模型时为 `multimodal` |
| `LAYA_THREADS` | CPU 算子线程上限，按物理核心数配置 | PyTorch 默认 |
| `LAYA_IDLE_UNLOAD_SECONDS` | 空闲多少秒后卸载检查点，下次请求重新加载。0 关闭 | `0` |
| `LAYA_MAX_LOADED` | 同时常驻的检查点数，预加载可按所加载的数量提高此上限 | `2` |
| `LAYA_MAX_TOKEN_BUDGET` | 请求中 `max_len` 和 `head_max_len` 的上限。超过返回 `422` | `8192` |
| `LAYA_DEFAULT_MODEL` | 默认检查点名，`default` 解析为 `multimodal` | `multimodal` |
| `LAYA_EXTRA_MODELS` | 从名称到本地目录或 Hub ID 的 JSON 映射 | 空 |
| `LAYA_API_KEY` | 要求预测和路由带 `Authorization: Bearer <key>`，限制健康详情，并关闭 `/docs`、`/redoc` 和 `/openapi.json` | 空 |
| `LAYA_LOG_LEVEL` | Uvicorn 日志级别 | `info` |
| `LAYA_MAX_CONCURRENT` | 通过认证后可同时处理的请求数，超额返回 `503` | `16` |
| `LAYA_BODY_READ_TIMEOUT` | 读取请求体的秒数上限，超时返回 `408` 并释放请求名额 | `10` |
| `LAYA_MAX_BATCH_TOKENS` | `/v1/systemone/batch` 单次前向的 token 预算，大批次可拆成多次前向 | `131072` |
| `LAYA_JEV_STRICT` | 严格 System One 格式，`0` 为扩展文本响应。图像请求使用严格格式 | 单条 `1`，批量 `0` |
| `LAYA_TOURNAMENT_GROUP` | 大候选集的淘汰分组大小，有效范围 2–100。开启后最多 255 个选择项 | `0` |

经反向代理公开 `/laya` 时，设 `LAYA_ROOT_PATH=/laya` 并让代理在转发前去掉该前缀。
应用内部仍接收 `/health`、`/v1/systemone` 等原始路径。为临时使用配置空闲卸载时，
推理结束会更新空闲计时，下一条请求承担重新加载耗时。释放检查点和设备缓存后，
进程分配器仍可能保留部分内存。 [Docker](docker.md)

<a name="endpoints"></a>

## 接口

| 方法和路径 | 用途 |
| --- | --- |
| `GET /health` | 存活及经认证的模型状态 |
| `POST /v1/route` | 查看路由 |
| `POST /v1/systemone` | 文本和命名图像，包括单框 |
| `POST /v1/systemone/batch` | 多条状态共用题目 |
| `POST /v1/multimodal` | 一个内联媒体文件 |

设置 `min_confidence` 时，扩展响应中的 `gate` 提供是否应暂缓使用答案的依据。

### `GET /health`

存活检查在推理时仍可访问。配置 Bearer 密钥时，未认证的探针收到 200 与 `{"status": "ok"}`，
通过认证可读到完整信息。未配置认证时，所有调用者都可读到下面的结构。

```json
{"status": "ok", "loaded": ["multimodal"], "revisions": {"multimodal": "..."},
 "device": "cuda", "device_is_preference": false,
 "checkpoint_devices": {"multimodal": "cuda"},
 "cpu_fallbacks": {"multimodal": {"count": 0, "last_reason": null}}}
```

- `status` 值为 `ok` 时表示进程能应答。
- `loaded` 列出常驻检查点，惰性加载前或空闲卸载后可为空。
- `revisions` 对应每个常驻检查点的修订号。
- `device` 返回常驻检查点的实际设备，未加载时返回配置偏好。
- `device_is_preference` 为 `true` 表示当前返回的是设备偏好。
- `checkpoint_devices` 按模型名给出实际设备。顶层 `device` 使用列表中的第一个可用值。
- `cpu_fallbacks` 是后端的回退次数和最后原因。原生多模态后端报告 0 与 `null`。

空闲卸载开启时，详情还包含 `idle_unload_seconds` 与 `idle_seconds`。健康检查和路由快照
都不会重新开始推理空闲计时。

### `POST /v1/route`

查看服务器当前的检查点选择，无需执行模型前向或加载检查点。

```bash
curl -s localhost:8000/v1/route -H 'content-type: application/json' -d '{
  "state": {"message": "Turn lights off"},
  "questions": {},
  "explain": true
}'
```

```json
{"model": "multimodal", "repo": "ZeeoRe/jev-multimodal-qwen3.5-4b",
 "reason": "default native checkpoint",
 "diagnostic": {"stage": "whole_state", "reason": "insufficient_words", "source": null}}
```

请求需要 `state`、`questions`，题目可以为空对象。可选 `model`、`task`、`lang`、
`lang_guess` 遵循预测接口的规则。`explain` 默认 `false`，显式提供时须为 JSON 布尔值。
路由接口拒绝 token 预算、置信度、批量控制和钩子参数，返回 `422`。这些控制有各自的预测接口
或部署配置。

默认返回 `model`、`repo` 与 `reason`。`explain: true` 还返回最终 `on_route`
结果的 `diagnostic`，其中 `stage`、`reason` 和 `source` 描述定位。没有诊断时为 `null`。
`source.path` 可能含调用方的字段名，路由理由可能包含请求值。应用按实际内容处理日志。
自定义路由器或钩子可改变选择。

认证、并发名额、读取期限和大小上限与预测接口共用。路由可能等待前序推理。路由快照反映执行时
的配置和钩子状态。对于不提供 `explain` 的自定义路由器，显式请求解释时返回 `422`。
预料外的服务错误写入日志并返回 `500`。

<a name="try-a-configuration-mismatch-locally"></a>

#### 核对客户端和服务器的配置

在仓库根目录执行离线示例。示例比较默认 `multimodal` 与自定义服务器的 `support` 路由，
也检查显式指定检查点的结果。

```bash
python -m pip install -e ".[serve]" httpx
python examples/http_route.py
```

### `POST /v1/systemone`

同一请求的所有问题共用 `state`。逐条处理多条状态时使用批量接口的 `states` 数组。

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice this month, I want my money back",
  "questions": {
    "queue":   {"type": "choice", "instructions": "Which team?",
                "criteria": {"billing": "billing and refunds", "tech": "login and app issues",
                             "other": "everything else"}},
    "urgency": {"type": "score",  "instructions": "How urgent?",
                "criteria": ["calm", "firm", "angry", "furious"]}
  }
}'
```

| 字段 | 必需 | 含义 |
| --- | --- | --- |
| `state` | 是 | 文本或 JSON 上下文，缺失或 `null` 返回 `400` |
| `questions` | 是 | 按 ID 索引的 `choice`、`score`、`noul` 或含图片的 `box` |
| `model` | 否 | 按名称选择检查点 |
| `assets` | 否 | 内联 PNG/JPEG 或 multipart 图片的引用映射 |
| `task` | 否 | 省略 `model` 时指定已注册名称 |
| `lang` | 否 | 后端使用的语言信息 |
| `lang_guess` | 否 | 路由诊断的语言提示 |
| `max_len` | 否 | 请求 token 预算，受 `LAYA_MAX_TOKEN_BUDGET` 限制 |
| `head_max_len` | 否 | 支持单独候选预算的后端使用。原生检查点使用 `max_len` |
| `min_confidence` | 否 | `[0, 1]` 的阈值，或按候选数桶的映射，例如 `{"choice:3-5": 0.9, "default": 0.4}` |

省略的控制沿用部署配置。钩子参数 `hooks`、`on_predict_start`、`on_predict_end`、
`hooks_raise`、`hooks_timeout` 由服务端管理，通过请求发送会返回 `422`。

可以指定 `multimodal`、`default`、已发布的 `ZeeoRe/jev-multimodal-qwen3.5-4b`
或 `LAYA_EXTRA_MODELS` 中的名称。客户端模型 ID 如 `jev-1` 使用服务默认检查点。
不可在请求中指定任意路径或未注册的 Hub ID，它们会返回 `422`。

<a name="image-assets-and-single-box-questions"></a>

### 命名图像与单框

发送 `TypeSafe-Danger-Zone-Images: true`，把 PNG 或 JPEG 图片绑定到 `assets`。
在 `state`、`instructions` 或 `criteria` 的字符串值中引用名字。图像边长均须为
280–1,568 像素。数组中先列图片引用再放文本，可让原生提示按此顺序放入图像和文字。
单框题使用一张独立的引用图片，返回 0 到 1 的坐标与定位质量。

```bash
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'TypeSafe-Danger-Zone-Images: true' \
  -F 'request={"model":"multimodal","state":{"photo":"scene-ref"},"assets":{"scene-ref":{"type":"image","part":"scene"}},"questions":{"target":{"type":"box","instructions":"Locate the red mug."}}};type=application/json' \
  -F 'scene=@scene.jpg;type=image/jpeg'
```

JSON 上传用 `format` 和 `data_base64` 取代 `part`。multipart 的第一个 part
是 JSON 请求，后续文件按资产引用对应。单文件最大 16 MiB，上传总大小和展开后 RGB
缓冲均受 32 MiB 限制。检查点的 token 预算同时生效。
[TypeSafe 图像说明](https://docs.typesafe.ai/images) · [SDK 示例](multimodal-clients.md#system-one-image-assets)

<a name="response"></a>

### 响应

单条请求默认返回 `model`、`answers` 和 `usage`。下面是 Qwen 检查点的文本示例。

```json
{
  "model": "Qwen/Qwen3.5-4B",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.9967,
      "probabilities": {
        "billing": 0.9978,
        "technical": 0.0018,
        "sales": 0.0004
      }
    },
    "urgency": {
      "type": "score",
      "score": 0.9502,
      "confidence": 0.5125,
      "probabilities": {
        "0": 0.1874,
        "1": 0.675,
        "2": 0.1376
      },
      "legend": {
        "0": "not urgent",
        "1": "needs attention soon",
        "2": "critical deadline or blocking issue"
      }
    },
    "refund_requested": {
      "type": "noul",
      "noul": 0.998
    }
  },
  "usage": {
    "input_tokens": 286,
    "output_tokens": 0
  }
}
```

答案使用原题目 ID。`choice` 给出选项和分布，`score` 给出期望档位和含义映射，
`noul` 是肯定概率，`box` 给出归一化框与定位质量。选项概率四舍五入至四位小数。
`input_tokens` 汇总实际准备的提示长度，`output_tokens` 为 0。扩展响应还有
`usage.forward_passes` 和回答该请求的 `routing`。 [概率精度](questions-and-answers.md#probability-precision)

启用 `min_confidence` 后，扩展字段包括 `gate`、`abstention_threshold`，
`abstention` 为 `passed`、`abstained` 或 `unevaluated`。低于阈值的答案保留原值并
标记 `low_confidence: true`。

<a name="confidence-three-numbers-not-interchangeable"></a>

### 三种置信度字段

| 字段 | 含义 |
| --- | --- |
| `answer_confidence` | 最大候选概率 `max(p)`。在独立数据上验证温度之后用作决策门槛。 |
| `confidence` | 扩展的 `choice`、`score` 为 `1 - H(p)/log(k)`，`noul` 为 `max(p_true, 1 - p_true)`。框为 IoU 估计。 |
| `x_jev_confidence` | 扩展响应中按 TypeSafe 公式计算的 `choice` 或 `score` 置信度。 |

TypeSafe 的选择置信度为 `(p_max - 1/n)/(1 - 1/n)`。评分置信度是到最可能档位的概率加权
距离，按均匀分布对应的距离缩放后用 1 减去该值。进程内可用
`laya_multimodal.jev_confidence(answer)` 得到同一量。迁移已有 TypeSafe 阈值时，始终确认
读的是同一公式。 [本地置信度](questions-and-answers.md#reading-confidence)

<a name="strict-jev-contract-laya_jev_strict"></a>

### 严格 System One 响应

单条默认只返回 `model`、`answers` 和两项 token 计数。

- `choice` 保留 `type`、`choice`、`probabilities`、`confidence`。
- `score` 保留 `type`、`score`、`probabilities`、`confidence`、`legend`。
- `noul` 保留 `type`、`noul`。
- `box` 保留 `type`、`box`、`confidence`。

选择和评分用 TypeSafe 置信度，`legend` 保留结构化档位和原始图片引用。要在文本响应中看到
路由和截断细节，用 `LAYA_JEV_STRICT=0`。批量默认是扩展响应，启用批量的严格字段投影时，
仍沿用批量的置信度。成功响应带 `Server-Timing: inference;dur=<ms>` 和 `X-Inference-Time-Ms`。

<a name="large-choices-with-laya_tournament_group"></a>

### 大候选集的分组淘汰

`LAYA_TOURNAMENT_GROUP` 接收 2 到 100 的分组大小，每轮胜者进入下一轮。扩展响应有
`x_tournament.finalists` 和 `x_tournament.rounds`，已淘汰候选概率为 0。批量中分别对每条
状态执行，累计各轮 token。若淘汰流程需要移除调用方给出的顺序，返回 `option_order_dropped`。
[分组评测](evals.md#attributing-tournament-errors)

<a name="limits"></a>

## 请求上限

| 项目 | 默认上限 |
| --- | --- |
| 普通 JSON 请求体 | 2 MiB |
| 启用图像请求头的 System One 请求体 | 32 MiB |
| 单张 System One 图像文件 | 16 MiB |
| 序列化文本状态 | 50,000 字符 |
| 单请求题目 | 64 |
| 每批状态 | 64 |
| 每道选择题候选 | 普通文本 100。图片、显式 `model: "multimodal"` 的单条请求及分组淘汰可到 255 |
| 每道评分题档位 | 普通文本 32，System One 图像题 10 |
| 全部答案选项 | 512 |
| 同时接入的请求 | `LAYA_MAX_CONCURRENT`，默认 16 |

超出检查点 `max_len` 或服务端限制时，缩短文本、媒体或候选说明，或在
`LAYA_MAX_TOKEN_BUDGET` 范围内提高预算。`LAYA_MAX_BATCH_TOKENS` 用于划分 HTTP 批量，
原生 Agent 再按自身 token 预算打包兼容的题目。 [输入预算](multimodal.md#local-prediction)

<a name="errors"></a>

## 错误

| 状态码 | 情况 |
| --- | --- |
| `400` | JSON 形状非法、缺少状态或题目、存在未配对的 Unicode 代理码位 |
| `401` | 服务启用认证，Bearer 密钥缺失或错误 |
| `408` | 请求体读取超时 |
| `413` | 超过服务的大小或记录数上限 |
| `422` | 题目、媒体或请求控制不符合接口要求 |
| `500` | 推理或路由异常，详情记入服务日志 |
| `503` | 请求名额已满或原生媒体检查点未配置 |

<a name="concurrency-model"></a>

## 并发

认证通过后，服务先检查接入名额。请求体读取完毕后才进入推理队列。同步模型前向在一个
工作线程中执行，事件循环仍可响应健康检查。读取超时会释放名额。客户端取消等待后，已开始的
服务端工作可能继续。

<a name="batch-and-media-requests"></a>

## 批量与内联媒体

`POST /v1/systemone/batch` 接收 `states` 数组和共用的 `questions`，按输入顺序给出
`results`，并在 `total_usage` 汇总 token。`POST /v1/multimodal` 接收一个媒体文件，
以 `{"base64": "..."}` 放在 `image`、`audio` 或 `video` 字段。命名 PNG 和 JPEG
图片资产使用 `POST /v1/systemone`。 [Python、TypeScript 与 curl 示例](multimodal-clients.md)
