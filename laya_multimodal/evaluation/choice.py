"""Evaluate shortlisting and tournament elimination for large choice sets.

The shared evaluator handles answer scoring, timing and slices. Each strategy
adds metrics derived from its decision and candidate metadata.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

from .evaluator import Dataset, EvalError, EvalReport, _group_cases, evaluate, file_fingerprint
from ..decisions.shortlist import (
    DEFAULT_SHORTLIST_K, DEFAULT_TOURNAMENT_GROUP, predict_shortlist, predict_tournament,
)

SHORTLIST_REPORT_SCHEMA = "laya-shortlist-evals-report/1"
TOURNAMENT_REPORT_SCHEMA = "laya-tournament-evals-report/1"


def _labels(question: Dict[str, Any], where: str) -> list[str]:
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        labels = list(criteria)
    elif isinstance(criteria, list):
        labels = criteria
    else:
        raise EvalError("%s: choice criteria must be a dict or list" % where)
    if not labels or any(not isinstance(label, str) for label in labels):
        raise EvalError("%s: choice criteria must contain string labels" % where)
    if len(set(labels)) != len(labels):
        raise EvalError("%s: choice criteria contain duplicate labels" % where)
    return labels


def _validate_choice_dataset(dataset: Dataset, strategy: str) -> None:
    if not isinstance(dataset, Dataset) or not dataset.examples:
        raise EvalError("%s evaluation needs a non-empty Dataset" % strategy)
    for index, example in enumerate(dataset.examples):
        if not example.expected:
            raise EvalError("row %d: expected must contain a choice label" % index)
        for qid, expected in example.expected.items():
            where = "row %d question %r" % (index, qid)
            question = example.questions.get(qid)
            if not isinstance(question, dict) or question.get("type") != "choice":
                raise EvalError("%s: %s evaluation requires a choice question" % (where, strategy))
            labels = _labels(question, where)
            if not isinstance(expected, str) or expected not in labels:
                raise EvalError("%s: expected label %r is not in criteria" % (where, expected))


class _ShortlistRunner:
    def __init__(self, agent: Any, embed_fn: Callable[[Sequence[str]], Any], k: int):
        self.agent, self.embed_fn, self.k = agent, embed_fn, k

    def predict(self, state: Any, questions: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
        kwargs = {"model": model} if model is not None else {}
        result = predict_shortlist(self.agent, state, questions, self.embed_fn, self.k, **kwargs)
        meta = result.get("shortlist")
        if not isinstance(meta, dict):
            raise EvalError("predict_shortlist returned no shortlist metadata")
        answers = result.get("answers")
        for qid, question in questions.items():
            if not isinstance(question, dict) or question.get("type") != "choice":
                continue
            where = "question %r" % qid
            original = _labels(question, where)
            item = meta.get(qid)
            if not isinstance(item, dict):
                raise EvalError("%s: missing shortlist metadata" % where)
            kept = item.get("labels")
            if (not isinstance(kept, list) or len(kept) != min(self.k, len(original))
                    or any(not isinstance(label, str) or label not in original for label in kept)
                    or len(set(kept)) != len(kept)
                    or item.get("n") != len(original) or item.get("k") != self.k
                    or item.get("passthrough") is not (self.k >= len(original))):
                raise EvalError("%s: inconsistent shortlist metadata" % where)
            scores = item.get("scores")
            if ((self.k >= len(original) and scores is not None)
                    or (self.k < len(original) and
                        (not isinstance(scores, list) or len(scores) != len(kept)))):
                raise EvalError("%s: inconsistent shortlist scores" % where)
            answer = answers.get(qid) if isinstance(answers, dict) else None
            if isinstance(answer, dict) and (answer.get("type") != "choice" or answer.get("choice") not in kept):
                raise EvalError("%s: choice answer is outside the shortlist" % where)
        return result


def _shortlist_metrics(cases: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not cases:
        return {}
    recalled = sum(case["shortlist_status"] != "retrieval_miss" for case in cases)
    correct = sum(case["shortlist_status"] == "correct" for case in cases)
    metrics = {"shortlist_recall_at_k": recalled / len(cases)}
    # There is no decision-stage denominator when retrieval recalls nothing.
    if recalled:
        metrics["shortlist_accuracy_on_recalled"] = correct / recalled
    return metrics


def evaluate_shortlist(
    agent: Any,
    dataset: Dataset,
    embed_fn: Callable[[Sequence[str]], Any],
    *,
    k: int = DEFAULT_SHORTLIST_K,
    checkpoint_id: str,
    embedder_id: str,
    dataset_path: Optional[str] = None,
) -> EvalReport:
    """Evaluate labelled choice questions with retrieval/decision error attribution.

    ``checkpoint_id`` and ``embedder_id`` are caller-supplied experiment identifiers,
    ideally immutable revisions. ``dataset_path`` adds the SHA256 of the JSONL bytes when
    the caller loaded ``dataset`` from that file. The same ``embed_fn`` used at inference
    is used here. Timing includes both embedding and the decision call.

    ``shortlist_recall_at_k`` counts a gold label kept by retrieval. Conditional
    ``shortlist_accuracy_on_recalled`` is absent when recall is zero. The existing
    ``choice_accuracy`` remains the end-to-end accuracy over every scored case.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise EvalError("k must be a positive integer, got %r" % k)
    if not callable(embed_fn):
        raise EvalError("embed_fn must be callable")
    for name, value in (("checkpoint_id", checkpoint_id), ("embedder_id", embedder_id)):
        if not isinstance(value, str) or not value.strip():
            raise EvalError("%s must be a non-empty string" % name)
    _validate_choice_dataset(dataset, "shortlist")
    config: Dict[str, Any] = {"shortlist": {"k": k, "checkpoint_id": checkpoint_id,
                                          "embedder_id": embedder_id}}
    if dataset_path is not None:
        config["dataset"] = dataset_path
        config["dataset_sha256"] = file_fingerprint(dataset_path)
    report = evaluate(_ShortlistRunner(agent, embed_fn, k), dataset, config=config)
    for case in report.cases:
        item = case.get("shortlist")
        if not isinstance(item, dict) or not isinstance(item.get("labels"), list):
            raise EvalError("question %r: missing shortlist metadata" % case["qid"])
        if case["expected"] not in item["labels"]:
            case["shortlist_status"] = "retrieval_miss"
        elif case["correct"]:
            case["shortlist_status"] = "correct"
        else:
            case["shortlist_status"] = "decision_miss"
    report.overall.update(_shortlist_metrics(report.cases))
    for dimension in ("language", "model", "qid", "tag"):
        for value, group in _group_cases(report.cases, dimension).items():
            report.slices[dimension][value].update(_shortlist_metrics(group))
    report.config["schema"] = SHORTLIST_REPORT_SCHEMA
    report.config["shortlist"]["counts"] = {
        status: sum(case["shortlist_status"] == status for case in report.cases)
        for status in ("correct", "retrieval_miss", "decision_miss")
    }
    return report


def _shape(n: int, group_size: int) -> tuple[int, int]:
    rounds = 0
    while n > group_size:
        n = -(-n // group_size)
        rounds += 1
    return rounds, n


class _TournamentRunner:
    def __init__(self, agent: Any, group_size: int):
        self.agent, self.group_size = agent, group_size

    def predict(self, state: Any, questions: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
        kwargs = {"model": model} if model is not None else {}
        result = predict_tournament(self.agent, state, questions, self.group_size, **kwargs)
        meta = result.get("tournament")
        choice_ids = {qid for qid, question in questions.items()
                      if isinstance(question, dict) and question.get("type") == "choice"}
        if not isinstance(meta, dict) or set(meta) != choice_ids:
            raise EvalError("predict_tournament returned incomplete tournament metadata")
        answers = result.get("answers")
        for qid in choice_ids:
            original = _labels(questions[qid], "question %r" % qid)
            item = meta[qid]
            rounds, finalists = _shape(len(original), self.group_size)
            kept = item.get("labels") if isinstance(item, dict) else None
            if (not isinstance(kept, list) or len(kept) != finalists
                    or any(not isinstance(label, str) or label not in original for label in kept)
                    or len(set(kept)) != len(kept)
                    or kept != [label for label in original if label in kept]
                    or type(item.get("n")) is not int or item["n"] != len(original)
                    or type(item.get("rounds")) is not int
                    or item["rounds"] != rounds):
                raise EvalError("question %r: inconsistent tournament metadata" % qid)
            answer = answers.get(qid) if isinstance(answers, dict) else None
            if (not isinstance(answer, dict) or answer.get("type") != "choice"
                    or answer.get("choice") not in kept):
                raise EvalError("question %r: choice answer is outside the finalists" % qid)
        return result


def _tournament_metrics(cases: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not cases:
        return {}
    survived = sum(case["tournament_status"] != "elimination_miss" for case in cases)
    correct = sum(case["tournament_status"] == "correct" for case in cases)
    metrics = {"tournament_survival_rate": survived / len(cases)}
    if survived:
        metrics["tournament_accuracy_on_survivors"] = correct / survived
    return metrics


def evaluate_tournament(
    agent: Any,
    dataset: Dataset,
    *,
    group_size: int = DEFAULT_TOURNAMENT_GROUP,
    checkpoint_id: str,
    dataset_path: Optional[str] = None,
) -> EvalReport:
    """Score labelled choices and attribute misses to elimination or the final choice.

    ``checkpoint_id`` is a caller-supplied identifier, ideally an immutable revision.
    ``dataset_path`` adds the JSONL file's SHA256. The existing choice accuracy is
    end-to-end; tournament confidence and probabilities describe finalists only.
    ``tournament_survival_rate`` includes choices that needed no elimination round.
    """
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2:
        raise EvalError("group_size must be an integer of at least 2")
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise EvalError("checkpoint_id must be a non-empty string")
    _validate_choice_dataset(dataset, "tournament")
    config: Dict[str, Any] = {
        "tournament_group_size": group_size,
        "tournament": {"group_size": group_size, "checkpoint_id": checkpoint_id},
    }
    if dataset_path is not None:
        config["dataset"] = dataset_path
        config["dataset_sha256"] = file_fingerprint(dataset_path)
    report = evaluate(_TournamentRunner(agent, group_size), dataset, config=config)
    for case in report.cases:
        item = case.get("tournament")
        if not isinstance(item, dict) or not isinstance(item.get("labels"), list):
            raise EvalError("question %r: missing tournament metadata" % case["qid"])
        if case["expected"] not in item["labels"]:
            case["tournament_status"] = "elimination_miss"
        elif case["correct"]:
            case["tournament_status"] = "correct"
        else:
            case["tournament_status"] = "final_choice_miss"
    report.overall.update(_tournament_metrics(report.cases))
    for dimension in ("language", "model", "qid", "tag"):
        for value, group in _group_cases(report.cases, dimension).items():
            report.slices[dimension][value].update(_tournament_metrics(group))
    report.config["schema"] = TOURNAMENT_REPORT_SCHEMA
    report.config["tournament"]["counts"] = {
        status: sum(case["tournament_status"] == status for case in report.cases)
        for status in ("correct", "elimination_miss", "final_choice_miss")
    }
    return report
