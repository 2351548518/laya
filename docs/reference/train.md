# Fine-tuning

**English** · [简体中文](../../docs_zh/reference/train.md)

`laya_multimodal.train` provides the joint trainer and the command entry point for typed
decisions and single-box grounding. The trainer and standalone evaluation share the dataset
loader in `data` and metrics in `evaluation.native_metrics`. The `models.modality_calibration` module
fits modality-specific temperatures and ordinal calibration on held-out data. The `training`
package supplies loss functions, optimizer snapshots, distributed updates and SwanLab reporting.
[Fine-tuning guide](../finetune.md#run-the-joint-trainer)

## Training and validation

::: laya_multimodal.train.GroundingTrainConfig

::: laya_multimodal.train.finetune_grounding

::: laya_multimodal.train.load_grounding_rows

::: laya_multimodal.train.grounding_metrics

## Decision labels

::: laya_multimodal.train.decision_target

::: laya_multimodal.train.target_from_gold

::: laya_multimodal.train.target_from_expected
