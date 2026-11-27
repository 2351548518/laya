# 文本决策评测

[English](../evals.md) · **简体中文**

`laya_multimodal.evals` 对有标签的文本决策评分，支持按切片查看指标，与已审核的基线做
回归比较。解析、指标计算和报告比较可离线运行。实际预测需要可用的检查点或调用方提供的 runner。
图片、音视频与单框标签使用原生评测器。 [原生评测](benchmarks.md)

<a name="quickstart"></a>

## 快速开始

```bash
# check the format without a model
laya-multimodal eval validate research/evals/fixture.jsonl

# score a labelled set on one checkpoint, with thresholds and a baseline
laya-multimodal eval run data.jsonl --model multimodal --device cpu \
    --min-accuracy 0.8 --max-ece 0.05 --score-within 0.25 --slice language \
    --json report.json --markdown report.md

# compare a saved report to a baseline
laya-multimodal eval compare report.json --baseline baseline.json --tolerance choice_accuracy=0.02
```

成功退出码为 `0`，指标或基线门槛未通过为 `1`，参数和数据用法错误为 `2`。
用 `--json` 保存完整报告，用 `--markdown` 保存摘要。终端显示总体及请求的切片。

<a name="comparing-a-student-with-repeated-teacher-decisions"></a>

## 比较学生与教师的重复决策

用一次教师观察作固定参考。第二次教师独立作答，学生也单独作答。`agreement` 用三个已保存的
运行计算相对于同一参考的一致率。复采样种子只控制置信区间。

```bash
laya-multimodal eval agreement reference.json repeat.json student.json \
    --bootstrap-samples 2000 --seed 0 --json agreement.json
```

输入的 JSON 结构如下。

```json
{
  "schema": "laya-agreement-run/1",
  "run_id": "teacher-reference",
  "provenance": {
    "model": "teacher-checkpoint",
    "revision": null,
    "settings": {"temperature": 0.7, "prompt_template": "Choose one label."},
    "seed": 42
  },
  "cases": [{
    "id": "ticket-001",
    "state": "I was charged twice.",
    "questions": {"route": {
      "type": "choice",
      "instructions": "Select the team responsible for this request.",
      "criteria": ["billing", "support"]
    }},
    "answers": {"route": "billing"},
    "language": "en",
    "tags": ["duplicate-charge"]
  }]
}
```

为每次观察提供不同 `run_id`。三份文件须包含相同的样本 ID、状态、问题及其选项顺序、语言和
标签，可改变记录行顺序。每道题都有答案。教师给出合法标签，学生可用 `null` 表示暂缓回答。
两次教师观察使用相同模型、revision 和 `settings`，采样种子单独记录。来源未知时用 `null`。

也可校验教师记录后直接运行本地学生。

```python
import json
import laya_multimodal
from laya_multimodal.evals import evaluate_agreement

with open("reference.json", encoding="utf-8") as handle:
    reference = json.load(handle)
with open("repeat.json", encoding="utf-8") as handle:
    repeat = json.load(handle)
revision = "YOUR_CHECKPOINT_COMMIT"
with laya_multimodal.Router(device="cpu", revision=revision) as runner:
    report = evaluate_agreement(
        runner, reference, repeat, model="multimodal",
        student_provenance={
            "model": "ZeeoRe/jev-multimodal-qwen3.5-4b", "revision": revision,
            "settings": {"device": "cpu"},
        },
    )
print(report["overall"])
```

`evaluate_agreement` 使用 `predict(state, questions, model=...)`。运行器错误会上报。
标为 `abstained` 的有效答案记为 `null`。`laya-agreement-report/1` 保留来源、输入哈希、
逐题结果及按题目、语言和标签的切片。两组一致率的分母均包括未作答项，覆盖率单独报告。
95% 分位区间按完整样本 ID 重采样，保留同一记录中的所有题。少于两个样本时区间为 `null`。
一致率反映当前配对观测，正确性和校准仍使用有真值的评测。
[教师一致性代码](https://github.com/2351548518/laya/blob/main/examples/evals/teacher_agreement.py)

<a name="attributing-shortlist-errors"></a>

## 诊断候选缩减

`evaluate_shortlist` 把候选筛选与最后的选择分别统计。

```python
import laya_multimodal
from laya_multimodal.evals import Dataset
from laya_multimodal.evals_shortlist import evaluate_shortlist
from laya_multimodal.shortlist import embed_fn_from_agent

agent = laya_multimodal.load()  # use the native checkpoint input embeddings
dataset_path = "intents.jsonl"
dataset = Dataset.from_jsonl(dataset_path)
report = evaluate_shortlist(
    agent, dataset, embed_fn_from_agent(agent), k=20,
    checkpoint_id="my-checkpoint@revision", embedder_id="my-encoder@revision",
    dataset_path=dataset_path,
)
print(report.overall)
print(report.cases[0]["shortlist_status"])
```

每题为 `correct`、`retrieval_miss` 或 `decision_miss`。
`shortlist_recall_at_k` 是正确标签进入候选的比例，`shortlist_accuracy_on_recalled`
是保留下正确标签的样本中选对的比例。`choice_accuracy` 仍覆盖全部样本。`k >= n` 时保留
原题，召回为 1。延迟包含候选嵌入与决策。使用部署实际的嵌入函数与检查点，在
`checkpoint_id`、`embedder_id` 中记录修订，并用 `dataset_path` 保存数据 SHA-256。

<a name="attributing-tournament-errors"></a>

## 诊断分组淘汰

```python
import laya_multimodal
from laya_multimodal.evals import Dataset
from laya_multimodal.evals_tournament import evaluate_tournament

agent = laya_multimodal.load()
dataset_path = "intents.jsonl"
report = evaluate_tournament(
    agent, Dataset.from_jsonl(dataset_path), group_size=16,
    checkpoint_id="my-checkpoint@revision", dataset_path=dataset_path,
)
print(report.overall)
print(report.cases[0]["tournament_status"])
```

报告列出决赛选项和轮数，区分 `correct`、`elimination_miss` 与 `final_choice_miss`。
`tournament_survival_rate` 按全部选择题计算正确标签进入决赛的比例。
`tournament_accuracy_on_survivors` 仅计算正确标签进决赛的题，在没有此类样本时留空。
`choice_accuracy` 保留端到端结果，延迟计入全部轮次。

在开发集比较 `group_size`，再选定设置做最终测试。末轮概率只覆盖入围选项。
`laya-tournament-evals-report/1` 与分组大小用于核对基线的评测策略。

<a name="native-checkpoint"></a>

## 选择原生检查点

用 `laya-multimodal eval run bench.jsonl --checkpoint /path/to/checkpoint` 评测有标签的
文本。媒体和单框使用 `scripts/multimodal/evaluate.py`。 [原生评测流程](benchmarks.md)

<a name="dataset-format"></a>

## 数据格式

每行一个 JSON 对象，允许空行及以 `#` 开头的注释。先用 `eval validate` 检查数据。

| 字段 | 必需 | 含义 |
| --- | --- | --- |
| `state` | 是 | 文本或 JSON 输入 |
| `questions` | 是 | 与 `Router.predict` 相同的题目映射 |
| `expected` | 是 | 按题目 ID 的真值，选择用标签，评分用数字，判断用布尔值 |
| `tags` | 否 | 用于切片的标签 |
| `language` | 否 | 用于切片的语言 |
| `model` | 否 | 单行使用的模型，`--model` 可以覆盖 |

`Dataset.from_jsonl` 和 `Example` 都校验问题与监督对象及题目键的一致性。格式问题在模型预测
前报 `EvalError`。 [JSONL 模板](https://github.com/2351548518/laya/blob/main/research/evals/dataset.template.jsonl)

<a name="metrics"></a>

## 指标

| 指标 | 计算方式 |
| --- | --- |
| `choice_accuracy` | 选中真值标签的比例 |
| `noul_accuracy` | 用 `p >= 0.5` 得出的布尔值与真值一致的比例 |
| `score_mae` | 评分与数字目标的平均绝对误差 |
| `score_within_<tol>` | 绝对误差不超过 `--score-within` 的比例 |
| `ece` | 按 15 个置信度桶统计校准偏差 |
| `brier` | 按 `P(correct)` 计算 `mean((confidence - correct)**2)` |
| `aurc` | 按不同置信度及其覆盖的样本数加权积分风险 |
| `selective_accuracy@50`、`selective_accuracy@80` | 取目标覆盖率处的阈值，统计不低于阈值的准确率 |
| `mean_confidence` | 用于统计的置信度均值 |
| `latency_p50_ms`、`latency_p95_ms` | 每条请求等待的整次调用时间 |
| `cost_per_decision_p50_ms`、`cost_per_decision_p95_ms` | 一次调用耗时除以其处理的记录数 |

置信度与正确性的联合指标使用可得到布尔正确性和置信度的选择或是非题。纯评分数据可计算
MAE 和容差命中率，`--max-ece` 会因缺少 `ece` 而失败。可重复传 `--score-within`，
或用 Python 的 `ScoreWithin(0.25)` 注册容差指标。数据里没有评分题时，容差指标缺失，
以该指标设定门槛会明确失败。

<a name="which-confidence-a-metric-reads"></a>

## 指标读取哪种置信度

`laya_multimodal.evals._answer_confidence` 优先读 `answer["answer_confidence"]`，
再依次读取 `confidence`、是非题的 `max(p, 1 - p)`、`max(probabilities)`。
扩展决策的选择和评分 `confidence` 使用归一化熵，单条 System One 严格响应使用 TypeSafe
公式。`LAYA_JEV_STRICT` 省略 `answer_confidence`，因此对照实验应固定响应格式。
[HTTP 置信度](http-api.md#confidence-three-numbers-not-interchangeable) · [原生校准](multimodal.md#calibration-and-evaluation)

<a name="coverage-cuts-and-ties"></a>

### 覆盖率与同分样本

覆盖率按置信度阈值切分。同分样本一起接收，实际覆盖率可高于目标。`aurc` 按每个置信度组
覆盖的样本数加权。`config.coverage_metric_definition` 记录指标定义，跨不同定义的覆盖率或
AURC 相对门槛会拒绝比较。重新生成需要对照的基线。

<a name="batching-and-timing"></a>

## 批量与计时

`--batch-size N` 将连续且检查点与问题结构相同的记录成批调用。每条请求等待整次调用，其
平均计算成本为调用时间除以该批行数。`config.timing` 记录实际分批方式。
默认比较会忽略计时指标，除非显式为其设容差。`--on-error fail` 上报异常，
`--on-error skip` 将失败项记入 `config.errored`，计时仅用成功调用。

<a name="grouping-the-rows-inside-a-batch"></a>

### 按长度分组

`--sort-by-length` 在多次前向间把接近长度的记录分组，减少 padding，返回顺序与输入一致。
`sort_by_length` 是请求配置，`sort_by_length_sent` 表示运行器收到该选项。
没有可重排的批次时，检查报告而不是从命令行参数推断效果。

<a name="the-abstention-gate-at-a-threshold"></a>

### 按阈值标记暂缓

`--min-confidence` 接收 `[0, 1]` 的有限数字。它给低于门槛的答案标记 `low_confidence`
与 `abstention`，原答案用于准确率等指标。运行器缺少相应参数时，显式请求阈值会报
`EvalError`。`config.timing.min_confidence` 与 `min_confidence_sent` 对应请求值和传递情况。

<a name="slices"></a>

## 切片

通过 `--slice` 按 `language`、`model`、`qid` 或 `tag` 输出细分指标。模型切片根据实际
回答每条记录的检查点归类。

<a name="opt-in-slice-gates"></a>

### 切片验收规则

把需要约束的切片写入 JSON 策略。

```json
{
  "version": 1,
  "rules": [
    {"slice": {"language": "zh"}, "metric": "choice_accuracy",
     "min_count": 50, "max_drop": 0.05},
    {"slice": {"qid": "intent"}, "metric": "ece",
     "min_count": 50, "max": 0.10}
  ]
}
```

```bash
laya-multimodal eval run data.jsonl --baseline baseline.json --tolerance choice_accuracy=0.02 \
    --gate-policy gates.json --json report.json
laya-multimodal eval compare report.json --baseline baseline.json \
    --tolerance choice_accuracy=0.02 --gate-policy gates.json
```

每条规则选择一个切片值、一个指标和正整数 `min_count`，并使用 `min`、`max`、
`max_drop` 或 `max_increase` 之一。后两个规则需要基线且两份报告均达到指标的有效样本数，
并有匹配的运行标识。缺少切片或指标、有效样本不足、跳过样本或发生错误时，门槛失败。
`config.gate_policy` 记录当时的规则，`compare` 可另外应用明确指定的策略。

<a name="run-identity"></a>

## 运行标识

| 配置键 | 记录 |
| --- | --- |
| `schema` | 报告结构 `laya-evals-report/1` |
| `dataset`、`dataset_sha256` | 文件路径与字节摘要 |
| `questions_sha256` | 问题 ID、类型、指令、选项、标签及显式的 `option_order` |
| `laya_version` | 评分程序版本 |
| `coverage_metric_definition` | 覆盖率与 AURC 的指标定义 |
| `thresholds`、`gate_policy` | 总体门槛与切片规则 |
| `revisions`、`timing` | 模型修订与实测计时配置 |

换指令、候选顺序或标签都可能改变问题摘要。`REPORT_SCHEMA`、`file_fingerprint` 和
`questions_fingerprint` 可在 Python 中复用。使用 `--revision SHA` 或
`--revision multimodal=SHA` 固定 Hub 修订，本地目录缺少对应修订元数据时记为 `null`。

<a name="baseline-and-ci-gate"></a>

## 基线与 CI 验收

一起维护数据、审阅后的基线和容差。`--tolerance METRIC=VALUE` 限制对应指标的绝对漂移，
`EvalReport.compare` 与 `assert_regression` 提供程序接口。
`EvalReport.comparable_to` 检查 schema、数据字节摘要及题目摘要，冲突时报告双方值。

```text
FAIL: baseline is not comparable: dataset_sha256 (dataset bytes): baseline is <sha>, this run is <sha>
```

旧报告中缺失的标识属于未知。要求可复现对照时，应生成含完整标识的新报告。
[仓库评测样例](https://github.com/2351548518/laya/tree/main/research/evals)

<a name="adding-the-real-labelled-set"></a>

## 接入真实标签集

按模板准备 JSONL，先验证，再运行同一检查点和预算。保留数据和审阅后的基线，再把评测命令接入 CI。

<a name="evidence-inspection"></a>

## 查看已有证据

```bash
laya-multimodal eval evidence --checkpoint ./my-checkpoint --report report.json
```

命令读取 `multimodal_config.json` 的 `training.laya_train_calibration` 和已有评测报告。
校准记录可为 `PRESENT`、`MISSING`、`INSUFFICIENT` 或 `UNKNOWN`，取决于样本数和
已保存问题。已知的数据摘要或问题摘要冲突会给出 `INCOMPARABLE`。缺少共享证据时，
检查点与报告的关系为 `UNKNOWN`。命令在本地检查文件，无需加载模型。
