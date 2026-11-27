# 历史守卫评测

[English](../guardrail-limitations.md) · **简体中文**

早期 Laya 0.3.21 和 0.3.22 曾在 CPU 上评测 53 条合成样本。报告记录了模型漏判包含嵌入式
指令的文档，以及误判引用攻击措辞的正常文本。当前原生模型应使用上线场景中经过复核的样本单独评测。
[原始样本与评测记录](https://github.com/NandhaKishorM/laya/issues/781)

<a name="what-was-measured"></a>

## 报告结果

| 阈值为 0.5 的筛选规则 | 召回率 | 假阳性率 |
| --- | ---: | ---: |
| `jailbreak` 或 `prompt_injection` | 0.72 | 0.21 |
| `prompt_injection` | 0.60 | 0.18 |
| 完整的 `LayaGuardrail` 规则 | 0.76 | 0.29 |

报告的 c042 在 c043 的基础上加入一条嵌入式指令，两条的 `prompt_injection` 都是 0.000。
安全课程中引用攻击措辞的 c038 则得到 0.619。建立评测时可保留此类成对对照。

<a name="what-this-does-and-does-not-say"></a>

## 按指标的单位读分数

`prompt_injection.noul` 是概率。四档 `harm_severity.score` 是 0–3 档位索引的期望值。
有完整概率分布时，守卫使用上半档的概率之和。每个上半档对期望的贡献至少为其概率的两倍，
因此期望为 0.82 时，上半档的合计概率至多为 0.41。

用独立验证数据选应用阈值，对不确定或影响较大的请求在执行前复核。
[概率与置信度](questions-and-answers.md#reading-confidence) · [分阶段上线](staged-adoption.md)
