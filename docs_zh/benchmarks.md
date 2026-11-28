# 原生决策评测

[English](../docs/benchmarks.md) · **简体中文**

固定一份与训练、检查点选择和校准分开的有标签分片。模型之间使用相同的请求、媒体预算
和目标。记录选用的检查点修订号、套件校验值、帧采样方式及上下文上限。

<a name="accuracy-and-calibration"></a>

## 准确率与校准

将准备数据时生成的清单交给原生评测器。

```bash
model=models/jev-multimodal-qwen3.5-4b
data=multimodal-data/example/native

python scripts/multimodal/evaluate.py --model "$model" \
  --suite-manifest "$data/manifest.json" --suite test \
  --device cuda:0 --out multimodal-runs/eval/test.json

python benchmarks/bench_multimodal.py --help
```

| 输出 | 报告中查看的指标 |
| --- | --- |
| `choice` 和 `noul` | 准确率与 NLL、Brier、ECE 等概率指标 |
| `score` | 准确率、等级误差 MAE、归一化 MAE 与 RPS |
| 单个正例 `box` | 平均 IoU，以及 IoU ≥ 0.5 的比例 |
| 同时含决策与正例框 | 总决策准确率与单框 Acc@IoU≥0.5 的调和平均 |

当报告包含模态、任务、来源标签时，查看对应的细分指标。读百分比时同时看样本数。
记录开发集选模所用的指标和分母，单独报告最终测试结果。文本评测器可在同一组文本
样本上比较外部决策接口给出的概率。
[文本评测器](evals.md) · [v6 同开发集比较](project-readme/multimodal-current.md#matched-development)

<a name="latency"></a>

## 延迟

在支持所选输入的硬件上分别测量文本、图像、视频和音频。记录设备、数值精度、检查点、输入长度、
首次预测、中位数和 P95。注明请求数以及是否缓存重复视频。比较不同运行时时，固定媒体和题目。
[延迟命令](https://github.com/2351548518/laya/blob/main/benchmarks/README.md) · [已发布的对比](https://github.com/2351548518/laya/blob/main/BENCHMARKS.md)
