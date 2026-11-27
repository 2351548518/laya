# 用 Docker 部署原生决策模型

[English](../docker.md) · **简体中文**

同一镜像可以执行一次性决策，也可以提供 System One HTTP 服务。镜像包含 Qwen3.5、
Gemma 4 的运行时、FFmpeg 和媒体依赖。

<a name="serve-a-native-multimodal-checkpoint"></a>

## 启动 HTTP 服务

准备包含 `multimodal_config.json`、已保存的处理器和模型权重的合并检查点。在仓库根目录运行。

```bash
export LAYA_MULTIMODAL_MODEL_DIR="$PWD/models/jev-multimodal-qwen3.5-4b"
docker compose -f compose.yaml -f compose.http.yaml -f compose.multimodal.yaml \
  up --build laya-multimodal-serve
```

此配置以只读方式挂载检查点并启用离线加载。如果要从 Hub 加载，省略 `compose.multimodal.yaml`，
把仓库 ID 设为 `LAYA_MULTIMODAL_MODEL`。空值使用发布的 Qwen3.5-4B 模型。
命名缓存卷 `model-cache` 在重建容器后保留已下载文件。 [已发布检查点](multimodal-release.md)

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/docker/request.json
```

`/health` 检查进程是否可用。向决策接口发送请求可检验实际推理。
`LAYA_PRELOAD=1` 让服务在启动时加载检查点。宿主机端口默认绑定环回地址。
用 `LAYA_PORT` 修改端口，用 `LAYA_BIND_ADDRESS` 修改宿主机绑定地址。
通过 `LAYA_API_KEY` 启用 Bearer 认证，或把密钥文件挂载进容器并在 `LAYA_API_KEY_FILE`
填写容器内路径。文件需可供容器内 UID 10001 读取。
[HTTP 设置](http-api.md) · [部署安全](security.md)

Qwen3.5 处理文本、图像和视频画面。音频波形需要具备音频能力的 Gemma 4 检查点。
[媒体请求](multimodal-clients.md)

<a name="platforms"></a>

## 平台配置

| 宿主机 | 配置 | 默认 wheel 索引 |
| --- | --- | --- |
| Linux AMD64 或 ARM64 CPU | 上面的基础配置 | `cpu` |
| NVIDIA GPU | 在 `up` 前加 `-f compose.cuda.yaml` | `cu126` |
| DGX Spark | 在 `up` 前加 `-f compose.spark.yaml` | `cu130` |

NVIDIA 宿主机需安装兼容驱动和 NVIDIA Container Toolkit，用 `LAYA_GPU_ID` 选择物理显卡。
仓库的 CUDA 12.6 配置适用于 570 系列驱动，CUDA 13.0 配置需要 580 系列或更新驱动。
核对部署机器和容器 wheel 的兼容范围。
[NVIDIA 驱动兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html) · [PyTorch CUDA 12.6 wheel](https://download.pytorch.org/whl/cu126/torch/)

Apple 芯片上的 MPS 推理使用宿主机安装方式。
[本机安装](https://github.com/2351548518/laya/blob/main/LOCAL_SETUP.md)

<a name="run-one-local-request"></a>

## 执行一次请求

```bash
docker compose -f compose.yaml -f compose.http.yaml -f compose.multimodal.yaml \
  run --build --rm laya-multimodal
```

使用自己的请求文件时，将 `LAYA_INPUT_DIR` 指向文件所在目录并增加 `-f compose.example.yml`。
容器读取 `/inputs/request.json`，请求包含 `state` 和 `questions`。媒体文件放在 JSON
旁边，并使用容器内路径，例如 `"state": {"text": "Localize the red cup", "image": "/inputs/cup.png"}`。
[原生输入](multimodal.md#local-prediction)
