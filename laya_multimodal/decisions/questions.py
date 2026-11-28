"""Question types, option ordering and input formatting."""
import json
import threading
from numbers import Integral
from typing import Any, Dict, List, Optional, Union

import numpy as np

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
    `sort_by_length` off, because that knob's own gate is `1 < chunk`. `laya_multimodal/decisions/shortlist.py`,
    `laya_multimodal/mcp/tools.py` and `laya_multimodal/serve.py` already refuse these shapes; the two `predict_batch`
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
