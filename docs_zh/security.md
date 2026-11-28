# 检查点完整性

[English](../docs/security.md) · **简体中文**

从可信来源加载原生检查点。运行时使用 safetensors 保存的权重，加载前可校验 SHA-256，
并关闭 Transformers 的远程代码加载。

<a name="pin-a-revision"></a>

## 固定模型修订

设置 `LAYA_REVISION=reviewed` 可使用发布的原生 Qwen 检查点已记录的提交。对其他 Hub
检查点，填写对应的提交 ID。`load(revision=...)` 优先于环境变量。
[已发布模型](multimodal-release.md)

<a name="verify-artifact-digests"></a>

## 校验文件摘要

通过 `load(expected_sha256=...)` 或 `LAYA_SHA256_DIGESTS` 提供从检查点相对文件路径到十六进制
SHA-256 的映射。清单应包含实际审核过的元数据、权重和处理器文件，并对部署时使用的文件计算摘要。
从可靠渠道取得校验值，才能验证目标文件的来源。

```python
from laya_multimodal import load

agent = load("models/jev-multimodal-qwen3.5-4b",
             expected_sha256={"decision_head.safetensors": "<reviewed SHA-256 digest>"})
```

多模型服务使用以已注册检查点名称为键的摘要映射。分别存放 Bearer 密钥和 Hub 凭据。
服务应绑定环回地址或可信私网。需要从网络外部访问时，配置认证及 TLS 反向代理。
[HTTP 服务设置](http-api.md) · [Docker 部署](docker.md)

`/health` 可用于存活检查。启用 API 密钥后，发送 `Authorization: Bearer …` 才能读取
已加载检查点及设备详情。
