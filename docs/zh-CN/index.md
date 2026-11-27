# laya-multimodal 文档

[English](../index.md) · **简体中文**

laya-multimodal 在本地对文本和模型支持的图片、视频、音频做 System One 决策。每道题返回
`choice`、`noul`、`score` 或 `box` 格式的答案，生成的答案 token 数为零。Qwen3.5 支持
文本和视觉输入。处理音频波形时，使用具备音频能力的 Gemma 4 检查点。

## 从仓库根目录开始

安装项目并下载已发布的 Qwen 检查点。先跑通文本决策，再用同一个 Python Agent 处理图片或
无音轨的视频画面。 [中文 README](https://github.com/2351548518/laya/blob/main/README.zh-CN.md#安装) · [本地推理](multimodal.md)

## 按任务查阅

| 要做什么 | 文档 |
| --- | --- |
| 设计题目、读取答案和概率 | [问题与答案](questions-and-answers.md) · [结构化输出](structured.md) |
| 在图片中定位一个目标 | [单框定位](grounding.md) |
| 选择检查点或使用终端命令 | [路由](routing.md) · [CLI 与 MCP](cli-mcp.md) |
| 复现 v6 或用自己的 JSONL 训练 | [v6 教程](project-readme/index.md) · [微调](finetune.md) · [数据格式](multimodal.md#local-jsonl-data-loading) |
| 比较准确率、校准和延迟 | [原生评测](benchmarks.md) · [文本评测](evals.md) |
| 从 Python、TypeScript 或其他进程调用服务 | [System One HTTP](http-api.md) · [媒体客户端](multimodal-clients.md) |
| 用 Docker 部署或运行更小的量化模型 | [Docker](docker.md) · [GGUF 部署](multimodal-release.md) · [检查点完整性](security.md) |
| 接入应用流程 | [决策应用](use-cases.md) · [预测钩子](hooks/index.md) · [分阶段上线](staged-adoption.md) |
| 查询 Python 接口 | [Python API](reference/index.md) |

v6 教程提供标有张量维度的流程图、训练配置和原始样本索引。已有指标注明对应检查点和评测分片。
[模型与损失](project-readme/multimodal-current.md) · [按模态查样本](project-readme/multimodal-current-io-examples.md)
