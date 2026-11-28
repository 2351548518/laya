# 联合训练接口

[English](../../docs/reference/train.md) · **简体中文**

`laya_multimodal.train` 提供决策和单框定位的联合训练循环与命令入口。训练和独立评测
共用 `data` 的数据加载和 `evaluation.native_metrics` 的决策指标。`models.modality_calibration`
使用独立分片拟合各模态的温度和序数校准。`training` 包提供损失函数、优化器快照、
跨卡更新和 SwanLab 上报。
[运行联合训练](../finetune.md#run-the-joint-trainer)

<a name="training-and-validation"></a>

## 训练和验证

::: laya_multimodal.train.GroundingTrainConfig

::: laya_multimodal.train.finetune_grounding

::: laya_multimodal.train.load_grounding_rows

::: laya_multimodal.train.grounding_metrics

<a name="decision-labels"></a>

## 决策标签

::: laya_multimodal.train.decision_target

::: laya_multimodal.train.target_from_gold

::: laya_multimodal.train.target_from_expected
