"""Question formatting, probability losses and confidence helpers."""
import json
import math
import threading
from numbers import Integral
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}
_DEFAULT_NOUL_LABELS = {"false": "false", "true": "true"}


_TOKENIZE_LOCK = threading.RLock()


def encode_text(tok, text, **kwargs):
    """Tokenize under a lock because a fast tokenizer changes padding and truncation settings."""
    with _TOKENIZE_LOCK:
        return tok(text, **kwargs)


def serialize_state(state: Union[str, dict, list]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value) -> str:
    """Render one criterion value as text.

    Strings pass through; anything structured (dict, list, number) becomes a single-line JSON
    document with the default separators -- ``", "`` between members, ``": "`` before a value --
    so a rubric reads as JSON rather than a Python repr. Without this a dict-valued criterion
    crashed `noul` outright and leaked `{'desc': ...}` into `choice` and `score` prompts.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _resolve_noul_labels(labels=None):
    if labels is None:
        labels = _DEFAULT_NOUL_LABELS
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = labels["false"], labels["true"]
    if not isinstance(false_label, str) or not isinstance(true_label, str):
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    false_label, true_label = false_label.strip(), true_label.strip()
    if not false_label or not true_label or false_label == true_label:
        raise ValueError("noul labels must map exactly 'false' and 'true' to distinct non-empty strings")
    return false_label, true_label


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul semantic order is always [false, true]."""
    t, crit = q["t"], q.get("crit")
    if t != "noul" and "labels" in q:
        raise ValueError("labels is only supported for noul questions")
    if t == "choice":
        # only None/"" mean "no description"; 0 and False are legitimate criterion values.
        # `str(k)` unconditionally: a label with no description is rendered as itself, so an int
        # label used to come back as an int from a function annotated `-> List[str]` and then
        # reached `build_sequence`, which calls `.replace` on it and raised an AttributeError
        # naming neither the question nor the label. With a description the same label already
        # went through `"%s: %s" %` and was a str, which is why only the undescribed form broke.
        # `structured._enum_field` stringifies labels the same way; the returned answer still
        # carries the caller's original label, which is unchanged.
        return [str(k) if v is None or v == "" else "%s: %s" % (k, render_criterion(v))
                for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_label, true_label = _resolve_noul_labels(q.get("labels"))
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        false_label + ": "
        + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        true_label + ": "
        + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def check_batch_size(batch_size: Any) -> Optional[int]:
    """Validate the per-call `batch_size` cap before the chunker has to spend it.

    `batch_size` counts states per forward pass, and the batch loop uses it as a `range()` step.
    A `float` -- what a JSON `2.0` parses to, and what `n / 2` yields -- reached `range()` and came
    back as the interpreter's own `'float' object cannot be interpreted as an integer`; a `str` was
    compared to zero first and leaked `'>' not supported between instances of 'str' and 'int'`;
    `True` is an `int` to `range()`, so it silently capped every pass at one state and switched
    `sort_by_length` off, because that knob's own gate is `1 < chunk`. `laya/shortlist.py`,
    `laya/mcp/tools.py` and `laya/serve.py` already refuse these shapes; the two `predict_batch`
    entry points and `window_batch_cap` did not, so a Python caller -- the documented route, and
    the one with no schema in front of it -- met the raw `TypeError` instead of the named refusal.

    `None`, zero and negative integers denote an uncapped batch. The chunker uses
    `len(states)` and `window_batch_cap` returns `None`. Integral counts such as
    `numpy.int64` are accepted. Booleans raise `ValueError`.
    """
    if batch_size is None:
        return None
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise ValueError("batch_size must be an integer count of states, got %r" % (batch_size,))
    return int(batch_size)


def unpermute_probs(p: np.ndarray, option_order: Optional[List[int]]) -> np.ndarray:
    """Put a slot-ordered probability row back into the caller's option order.

    `build_sequence` puts option `option_order[s]` in slot `s`, so a model row comes back
    indexed by slot. Everything downstream indexes by option -- `zip(keys, p)` for a choice,
    `arange(k) * p` for a score level, `p[1]` for noul-true -- so the row has to be inverted
    first or the probabilities end up attached to the wrong options, which is silent.

    A missing or mismatched order returns `p` untouched, so the canonical path is unaffected.
    """
    if option_order is None or len(option_order) != len(p):
        return p
    canonical = np.empty_like(p)
    canonical[np.asarray(option_order, dtype=int)] = p
    return canonical


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Historical RLCD reward: clipped log score + spherical score + ranked probability score.

    The finite log floor is a numerical surrogate. Unlike the unclipped scoring rules, this
    combination is not globally strictly proper for arbitrarily small target probabilities.
    Supervised decision losses use stable log-softmax and differentiable Brier/RPS terms.

    q: [..., N, K] reported distributions
    target: [N, K] (one-hot or soft target distributions)
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def td_lambda_targets(p_true: torch.Tensor, batch: Dict, lam: float = 1.0) -> torch.Tensor:
    """TD(lambda) targets for multi-turn conversation trajectories."""
    target = batch["target"].clone()
    groups = batch.get("ep_group")
    if groups is None:
        return target
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(batch["ep_step"][idx])]
        y = batch["target"][idx[-1], 1]
        G = y
        for j in range(len(idx) - 1, -1, -1):
            if j < len(idx) - 1:
                G = (1 - lam) * p_true[idx[j + 1]] + lam * G
            target[idx[j], 0], target[idx[j], 1] = 1 - G, G
    return target


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error across confidence bins.

    `bins` is the number of histogram buckets and must be an integer of at least 1. It is refused
    rather than misread because `np.linspace(0, 1, bins + 1)` yields no interior edge for `0` and
    `-1`: the loop body never runs and the function returns `0.0`, the best score ECE can report,
    for a distribution that is anything but well calibrated. A count that arrives from a config
    file as `0` therefore turns a calibration gate into one that always passes. `True` is refused
    with the same argument -- it would silently histogram into a single bucket. Any other
    `Integral` is accepted, a `numpy.int64` among them, since the arrays here are numpy's own.
    """
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise ValueError("bins must be an integer of at least 1, got %r" % (bins,))
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def answer_confidence(p: np.ndarray, k: int) -> float:
    """Probability mass on the answer being reported: max(p).

    This is the quantity temperature scaling fits, and the quantity every calibration figure in
    this repository is computed on -- both benchmark harnesses take `conf = max(probs)` before
    calling `ece_score`. The README's gating section relies on the property that goes with it:
    of the answers returned at confidence c, about c of them are right. That property is
    conditional, and the condition is not met by default -- it holds only after the temperatures
    have been fitted and validated on held-out data for this checkpoint and this option count.
    The shipped checkpoints are over-confident: `choice:11+` is fitted at 0.1006, a ~10x
    sharpener that would publish a genuine 0.24 top probability as 0.9999. `clamp_temperature`
    refuses it and applies `TEMP_MIN` (0.5) instead, which still sharpens 2x and publishes that
    same 0.24 as 0.4993 -- so a threshold applied to these numbers still selects below model
    accuracy (issue #394). The clamp bounds the damage; it does not calibrate anything.

    `confidence_from_probs` below reports a different quantity on a different scale and carries
    no such guarantee, so the two must not be compared against the same threshold.
    """
    if k < 1:
        return 1.0
    val = float(p[:k].max())
    return 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy confidence: 1 - H(p) / log(k).

    How concentrated the whole distribution is. Useful, but not calibrated: it is not what
    temperature scaling fits and not what the reported ECE measures. See `answer_confidence`.

    The formula is reported only for ``k >= 2``: ``log(1)`` and ``log(0)`` are zero, so a
    distribution over fewer than two options has no entropy scale to divide by. Those inputs
    return ``1.0``, the *maximum* score this function can produce, whatever ``p`` holds -- an
    empty or single-option question reads as perfectly concentrated, and a caller comparing the
    reported `confidence` against any threshold sees a pass. That is not the same as the
    abstention gate passing it: `laya.confidence` gates on `answer_confidence` first and reaches
    this number only when that field is absent. The gate does pass a real one-option answer, for
    a different reason -- a softmax over one logit is exactly ``[1.0]``, so both numbers are
    ``1.0``. `laya.structured` builds such an answer from ``{"enum": ["solo"]}``, and the option
    count is bounded only above (`MAX_OPTIONS`).
    """
    if k < 2:
        return 1.0
    # The scalar `np.clip` on the score is the part worth replacing (a native comparison is ~4x
    # the dispatcher); the entropy sum itself stays vectorised. A Python loop over `pk` is faster
    # below roughly k=40 and slower above it -- 0.9 us vs 2.4 us at k=11, but 14.6 us vs 3.0 us at
    # k=256 -- and summing in a different order moves the last bits (1.7e-15 over 136k measured
    # distributions). The public readouts round this value to four decimals.
    pk = p[:k]
    ent = -float((pk * np.log(np.clip(pk, 1e-12, 1.0))).sum())
    score = 1.0 - ent / math.log(k)
    return 0.0 if score < 0.0 else (1.0 if score > 1.0 else score)


def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


# Bound the logit scale used by temperature calibration. Temperatures below one sharpen
# probabilities, and the floor limits the increase in confidence.
TEMP_MIN = 0.5
TEMP_MAX = 5.0


def clamp_temperature(t, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """A usable temperature: `t` confined to [lo, hi], falling back to 1.0 if it is not a number.

    A bool is not a number either: `True`/`False` used to float to 1.0/0.0 here and read as
    fitted/sharpening temperatures, the same class of quiet acceptance `check_min_confidence`
    already refuses. A custom `lo`/`hi` still bounds either way.
    """
    if isinstance(t, bool):
        return 1.0
    try:
        t = float(t)
    except (TypeError, ValueError, OverflowError):
        # `OverflowError` is a number this function cannot read, not a bug in the caller: JSON's
        # grammar has no double limit, so a temperature written as a 401-digit integer literal
        # parses as an `int` and `float()` refuses it. (`1e400` in the same file is fine -- JSON
        # reads that as a float, which is `inf`, and the check below already answers 1.0 for it.)
        # `confidence._finite`, `hooks.validate_timeout` and `calibrate.apply_calibration_payload`
        # all read this exception as "does not read as a number" already; here it used to escape,
        # and with it every `Agent`/`ONNXAgent`/`load_calibration` load that had read the
        # checkpoint off disk by then.
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):    # NaN / inf
        return 1.0
    return min(hi, max(lo, t))
