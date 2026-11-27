"""Shared decision question validation and local checkpoint loading."""
import json
import os
from typing import Any, Dict, List, Optional, Union

from .common import QTYPES, _resolve_noul_labels
from .hooks import HookRegistry


def _option_count(qdef: Dict) -> int:
    """How many options a validated question definition renders to.

    Mirrors `render_options`: a choice has one option per criterion, a score one per level,
    and a noul is always the pair [false, true].
    """
    if qdef.get("type") == "noul":
        return 2
    crit = qdef.get("criteria")
    return len(crit) if isinstance(crit, (dict, list, tuple)) else 0


class DecisionAgent(HookRegistry):
    """Question validation and structured answers shared by local decision agents."""

    model_id = None
    hooks_raise = True
    hooks_concurrent = True
    hooks_timeout = None
    _hooks_lock = None

    @staticmethod
    def _check_question(qid: str, qdef: Any) -> None:
        """Reject a question that cannot be answered, naming it and what to fix.

        `render_options` reads `criteria` in the shape the question's type expects and the decision
        head needs at least one option, so a malformed definition used to surface from three frames
        down as something that names neither the question nor the problem: `AttributeError:
        'NoneType' object has no attribute 'items'`, `KeyError: 'bool'`, or a `selected index k out
        of range` raised inside the model for a question that ended up with no options at all.
        """
        if qid is None:
            raise ValueError("question id must not be None")
        if not isinstance(qid, (str, int)) or (isinstance(qid, str) and not qid.strip()):
            raise ValueError("question id must be a non-empty string, got %r" % (qid,))
        if not isinstance(qdef, dict):
            raise ValueError("question %r: definition must be a dict, got %s"
                             % (qid, type(qdef).__name__))
        t = qdef.get("type")
        if not isinstance(t, str) or t not in QTYPES:
            raise ValueError("question %r: unknown type %r; use one of %s" % (qid, t, sorted(QTYPES)))
        if "instructions" not in qdef:
            raise ValueError("question %r: no 'instructions'; add the text the model should answer" % (qid,))
        ins = qdef["instructions"]
        if ins is None:
            raise ValueError("question %r: 'instructions' must not be None; add the text the model should answer" % (qid,))
        if isinstance(ins, str) and not ins.strip():
            raise ValueError("question %r: 'instructions' must not be empty; add the text the model should answer" % (qid,))
        if isinstance(ins, (list, dict)) and not ins:
            raise ValueError("question %r: 'instructions' must not be empty; add the text the model should answer" % (qid,))
        if not isinstance(ins, (str, dict, list, int, float)):
            raise ValueError("question %r: 'instructions' must be a string, dict, or list, got %s"
                             % (qid, type(ins).__name__))
        crit = qdef.get("criteria")
        if t == "choice":
            if not isinstance(crit, (dict, list)):
                raise ValueError("question %r: a choice question takes 'criteria' as a dict of "
                                 "label -> description, or a list of labels" % (qid,))
            if not crit:
                raise ValueError("question %r: a choice question needs at least one criterion" % (qid,))
            # A label is used as a dict key when a list of labels is normalised, so an unhashable
            # label raised `TypeError` from three frames down -- which names neither the question nor
            # the label, and which `serve` cannot classify as a caller error, so over HTTP it became
            # a 500 "inference failed" instead of a 422. Requiring the documented scalars exactly
            # also catches the hashable-but-not-scalar shapes (`tuple`, `frozenset`, `bytes`,
            # `complex`) that the old deny-list let through to the same 500. Labels are rendered as
            # option text, so a nested structure has no meaning here.
            for i, label in enumerate(crit if isinstance(crit, list) else crit.keys()):
                if label is not None and not isinstance(label, (str, int, float, bool)):
                    raise ValueError(
                        "question %r: choice label %d is a %s; a label is rendered as option text "
                        "and used as the answer key, so it must be a scalar (a string, number or "
                        "bool), got %r" % (qid, i, type(label).__name__, label))
                if label is None:
                    # The score path rejects a null level with the same argument one question type
                    # over: a label is option text AND the answer key, and a null one can be
                    # neither. `_to_internal` normalises a list of labels to `{label: None}`, so a
                    # null label became the option text "None" (`str(None)`) while the answer key
                    # and the `probabilities` key for that same option were the JSON string "null",
                    # because a dict key has to be a str. A client then cannot tell whether that
                    # option was the *string* `"null"` or JSON `null`, and the answer key is
                    # unreachable: `criteria[answer["choice"]]` yields `None`, and
                    # `answer["choice"] == "null"` is False for it. `""` stays accepted, because
                    # unlike a null it does round-trip.
                    raise ValueError(
                        "question %r: choice label %d is null; a label is rendered as option text "
                        "and used as the answer key, so it must be a string, number or bool -- a "
                        "null label renders as the text \"None\" while its answer key is \"null\""
                        % (qid, i))
            # `_to_internal` normalises the list form to `{label: None}`, so those labels become
            # the answer keys. Two entries that land on one key made the model score fewer options
            # than the caller wrote, and the response carry fewer probabilities than their list,
            # without a word -- the same silent-shape class as the two checks above. Python
            # treats values as one key whenever they compare equal, so `[1, 1.0]` and `[True, 1]`
            # collapse as well as an exact repeat. An unhashable label raised
            # `TypeError: cannot use 'tuple' as a dict key` from `_to_internal`, three frames down,
            # which names neither the question nor the label -- and which `serve` cannot classify
            # as a caller error, so over HTTP it became a 500 "inference failed" instead of a 422.
            if isinstance(crit, list):
                keys: Dict[Any, int] = {}
                for i, label in enumerate(crit):
                    try:
                        first = keys[label]
                    except TypeError as exc:
                        raise ValueError(
                            "question %r: choice label %d (%r) cannot be an answer key because it is "
                            "unhashable; a label is rendered as option text and used as the answer "
                            "key" % (qid, i, label)) from exc
                    except KeyError:
                        keys[label] = i
                    else:
                        raise ValueError(
                            "question %r: choice label %d (%r) repeats label %d; the labels are the "
                            "answer keys, so every option needs its own (1, 1.0 and True are one "
                            "key)" % (qid, i, label, first))
        elif t == "score":
            if not isinstance(crit, list):
                raise ValueError("question %r: a score question takes 'criteria' as a list of level "
                                 "descriptions, index 0 first" % (qid,))
            if not crit:
                raise ValueError("question %r: a score question needs at least one level" % (qid,))
            if None in crit:
                raise ValueError("question %r: score level %d is null; give every level a description, "
                                 "index 0 first" % (qid, crit.index(None)))
        elif crit is not None and not isinstance(crit, dict):
            raise ValueError("question %r: a noul question takes 'criteria' as a dict with optional "
                             "'true'/'false' descriptions, or omits it" % (qid,))
        elif isinstance(crit, dict):
            # `render_options` reads these two descriptions out by name -- `crit.get("false")` and
            # `crit.get("true")` -- so a dict keyed any other way is not a noul description at all.
            # It used to be substituted with the default pair without a word, so a caller saw their
            # descriptions accepted and never reach the model (#156). `labels` just below has
            # rejected the same mistake since #163; this is the same rule on the other parameter,
            # and a noul is a boolean question either way, so those are the only two keys it can have.
            keys = {str(k).lower() for k in crit}
            if not keys <= {"true", "false"}:
                raise ValueError(
                    "question %r: a noul question takes 'criteria' keyed only 'true'/'false' (either "
                    "or both, and omitted is fine), got %s. Those keys are the option texts the model "
                    "reads; any other key was silently dropped and replaced with the defaults. If you "
                    "want the answer worded differently, keep 'criteria' keyed 'true'/'false' and set "
                    "'labels' instead." % (qid, sorted(keys)))
        if "option_order" in qdef:
            # Slot s shows option `order[s]`. Anything other than a permutation of the option
            # indices would either drop an option or show one twice, so reject it here rather
            # than let it reach the encoder.
            order = qdef["option_order"]
            n = _option_count(qdef)
            if (not isinstance(order, (list, tuple))
                    or len(order) != n
                    or sorted(int(i) for i in order if isinstance(i, int) and not isinstance(i, bool)) != list(range(n))):
                raise ValueError(
                    "question %r: 'option_order' must be a permutation of range(%d) -- one slot per "
                    "option, each option once -- got %r" % (qid, n, order))
        if "labels" in qdef:
            if t != "noul":
                raise ValueError("question %r: 'labels' is only supported for noul questions" % (qid,))
            try:
                _resolve_noul_labels(qdef["labels"])
            except ValueError as e:
                raise ValueError("question %r: %s" % (qid, e)) from e

    @staticmethod
    def _to_internal(qdef: Dict) -> Dict:
        t = qdef["type"]
        crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        elif t == "noul" and isinstance(crit, dict):
            # Normalize boolean literal keys to string keys ("true"/"false")
            crit = {str(k).lower(): v for k, v in crit.items()}
        ins = qdef["instructions"]
        if not isinstance(ins, str):
            # `ensure_ascii=False`, matching `serialize_state` and `render_criterion` in
            # common.py and the instructions path in shortlist.py. The default escaped
            # non-ASCII to literal `\uXXXX`, which the tokenizer then read as escape text:
            # on the English checkpoint one German question answered noul=0.1652 as a dict
            # and noul=0.2650 as the identical plain string.
            ins = json.dumps(ins, ensure_ascii=False)
        q = {"t": t, "ins": ins, "crit": crit}
        if "labels" in qdef:
            q["labels"] = qdef["labels"]
        if "option_order" in qdef:
            q["option_order"] = [int(i) for i in qdef["option_order"]]
        return q

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def decide(self, state: Union[str, dict, list], schema: Any = None, *,
               questions: Optional[Dict[str, Any]] = None, return_details: bool = False,
               min_confidence: Optional[float] = None,
               **predict_kwargs) -> Any:
        """Answer `state` against a schema (JSON schema or pydantic model) and return typed values.

        See `laya.structured`. Pass exactly one of `schema` or `questions`; extra keyword arguments
        are forwarded to `predict` / `system_one`.
        """
        from .structured import decide as _decide
        return _decide(self, state, schema, questions=questions,
                       return_details=return_details, min_confidence=min_confidence, **predict_kwargs)

    def decide_batch(self, states: List[Union[str, dict, list]], schema: Any = None, *,
                     questions: Optional[Dict[str, Any]] = None,
                     return_details: bool = False, min_confidence: Optional[float] = None,
                     **predict_kwargs) -> List[Any]:
        """Answer many states against one schema (JSON schema or pydantic model) in one batched call.

        The throughput form of :meth:`decide`: the schema is planned once and its questions
        run over every state through :meth:`predict_batch` (shared forward passes, results
        in input order), then each state's answers are projected as ``decide`` does. Extra
        keyword arguments (``batch_size=``, ``lang=``, ``hooks=``, ...) are forwarded to
        ``predict_batch``. See `laya.structured`.
        """
        from .structured import decide_batch as _decide_batch
        return _decide_batch(self, states, schema, questions=questions,
                             return_details=return_details, min_confidence=min_confidence,
                             **predict_kwargs)

    def __repr__(self) -> str:
        return "%s(model_id=%r, device=%s)" % (type(self).__name__, self.model_id, getattr(self, "device", None))


def load(model_id_or_path=None, device=None, *, vegaml=False, **options):
    """Load a native decision checkpoint or an explicitly selected local Vega reader."""
    model = os.fspath(model_id_or_path) if model_id_or_path is not None else None
    if vegaml:
        from .vega import VegaAgent

        return VegaAgent(model, device=device, vegaml=vegaml, **options)

    from .multimodal import MultimodalAgent
    from .router import canonical_name, MULTIMODAL_MODEL

    if model is not None and canonical_name(model) == MULTIMODAL_MODEL:
        model = None
    return MultimodalAgent(model, device=device, **options)


def __getattr__(name):
    if name == "Agent":
        from .multimodal import MultimodalAgent

        return MultimodalAgent
    raise AttributeError(name)
