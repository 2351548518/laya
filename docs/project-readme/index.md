# v6 项目说明

这组文档面向使用 v6 数据训练和部署 laya-multimodal 的读者。示例和训练配置对应 Qwen3.5-4B 的 150 万条训练记录，说明项目对文本、图像、原生视频和单框定位的实现。

| 要看的内容 | 文档 |
| --- | --- |
| 数据来源、模型架构图、训练流程图、源码与损失示例 | [v6 数据、模型架构与训练](multimodal-current.md) |
| Python 与 HTTP 推理、启动训练、续训、评测和校准 | [v6 本地运行与训练](multimodal.md) |
| 按模态和输出类型查看完整请求与监督目标 | [真实输入输出样例](multimodal-current-io-examples.md) |
| 每个细任务的数量、模态内占比与三个样例 | [64 类细任务](multimodal-current-examples.md) |
| 本地检查点、LoRA 合并、GGUF 与已有发布结果 | [模型导出与发布](multimodal-release.md) |

仓库提供代码、配置和图表。运行训练及媒体示例时，请准备本地数据和权重。CPU 损失演示直接生成隐藏状态，可独立运行。已发布的权重来自 v1。
[CPU 损失演示](multimodal-current.md#optimizer-step) · [模型发布](multimodal-release.md) · [通用安装与基座支持](../multimodal.md)
