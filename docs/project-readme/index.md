# laya-multimodal v6 教程

从一条请求开始，学会读懂多模态决策的输入、模型输出和训练目标，再用仓库里的代码复现推理、训练、评测和导出。教程以 v6 的 Qwen3.5-4B 和 150 万条训练记录为主线，覆盖文本、图像、视频和图像单框定位。音频示例采用支持原生音频的 Gemma 检查点。

## 第一次怎么读

| 顺序 | 打开哪一节 | 读完后核对什么 |
| --- | --- | --- |
| 1. 先运行 | [CPU 损失演示](multimodal.md#cpu-walkthrough) | 看到 `[3, 3]` 候选 logits、`[1, 1, 4]` 预测框和一次可复算的损失。使用 Python 环境即可。 |
| 2. 看一条数据 | [图像单选如何进入模型](multimodal-current.md#input-to-model) | 能从 `state` 的图像引用找到媒体文件，从 `questions.action` 找到 `expected.action`。 |
| 3. 沿着模型图走一遍 | [架构图与张量形状](multimodal-current.md#architecture) | 从图底部的图像和文本向上读，找到视觉 token、共享骨干和两个任务头。 |
| 4. 沿着训练图走一遍 | [训练图与两阶段训练](multimodal-current.md#training) · [一次参数更新](multimodal-current.md#optimizer-step) | 从图顶部向下读，理解哪些权重参与更新，64 条记录如何在四张卡上完成一个优化器步。 |
| 5. 看真实输入 | [读取原始 JSONL](multimodal-current-io-examples.md#read-original-row) | 对照一道题的请求、标签和来源。已有 v6 本地数据时，用索引直接读取同一行。 |

## 按任务跳转

| 想做什么 | 从这里开始 | 需要准备 |
| --- | --- | --- |
| 用 Python 得到第一个答案，画出预测框 | [本地推理](multimodal.md#python-inference) | 已训练的原生检查点。教程会生成一张可用的测试图。 |
| 从客户端提交文本、图片、视频或音频 | [System One 请求](multimodal.md#systemone-http) · [音频和视频扩展](multimodal.md#audio-and-video) | 本地 HTTP 服务和对应模态的检查点。 |
| 对照每种模态和输出类型的完整样本 | [30 条 v6 请求和 Gemma 音频样例](multimodal-current-io-examples.md) | 可直接阅读请求和监督标签。 |
| 检查任务来源、数量和多样性 | [64 类细任务与 192 条摘录](multimodal-current-examples.md) | 按模态和任务名查找。 |
| 启动四卡训练或查看已有训练 | [复现两阶段训练](multimodal.md#reproduce) · [读日志与续训](multimodal.md#resume) · [SwanLab 实时曲线](multimodal.md#swanlab) | 原始 Qwen3.5-4B、v6 数据和可用 GPU。查看已有训练时使用它的日志。 |
| 解释指标、比较检查点 | [评测与校准](multimodal.md#evaluate) · [同集结果](multimodal-current.md#matched-development) | 相同的分片、样本范围和指标定义。 |
| 合并 LoRA、转 GGUF、选择量化版本 | [导出与部署](multimodal-release.md) | 本地训练检查点或已发布权重。 |

## 在哪里查看

在 VS Code 中打开本页的 Markdown 预览，可以点击表格中的小节链接。查看架构图或训练图时，点击图片本身可打开 SVG，图下也提供 PNG。执行代码块时，以包含 `laya/` 和 `pyproject.toml` 的仓库根目录作为工作目录。

喜欢用浏览器阅读时，可在安装文档依赖后从仓库根目录启动预览。已有 `zensical` 的环境可直接执行第二条命令。

```bash
python -m pip install -r requirements-docs.txt
zensical serve --dev-addr 127.0.0.1:8001
```

打开浏览器中的教程首页，从侧栏进入运行、原理、发布或样例小节。 [本地教程预览](http://127.0.0.1:8001/project-readme/)

## 先辨认本地文件的用途

| 路径 | 用途 |
| --- | --- |
| `docs/project-readme/assets/` | 教程图、真实样例索引、处理器形状记录、两阶段配置和 CPU 演示。 |
| `multimodal-data/open-jev-omni-qwen-filtered-v6/native/` | 本地 v6 的训练、预热、验证、校准 JSONL 和对应的媒体引用。 |
| `models/pretrained/Qwen3.5-4B/` | 从头训练时的原始基座。 |
| `multimodal-runs/qwen4b-v6-fresh/` | v6 本地训练的日志、配置、可推理检查点和优化器快照。 |

目前教程中的公开下载链接对应 v1 已发布权重，本地 v6 检查点的位置及用途列在发布章节。 [检查点与版本](multimodal-release.md)
