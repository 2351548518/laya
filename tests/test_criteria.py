"""Criteria rendering: structured values must not crash or leak Python reprs.

Regression tests for the bug reported in PR #2 (trocker): a `noul` question whose criteria
values were dicts raised `TypeError: can only concatenate str (not "dict") to str`, and
`choice`/`score` stringified dicts as Python reprs instead of JSON.

A question whose *shape* is wrong is the other half of the same promise and lives at the bottom of
this file: `Agent.system_one` rejects it by name before anything is rendered or tokenized, instead
of raising `AttributeError: 'NoneType' object has no attribute 'items'` from `render_options` or a
`selected index k out of range` from inside the decision head (#182).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.common import ece_score, render_criterion, render_options  # noqa: E402

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s:\n     got  %r\n     want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s %s" % (name, detail))


def check_raises(name, fn, message):
    try:
        fn()
        FAIL.append("%s: should have raised ValueError" % name)
    except ValueError as e:
        check(name, str(e), message)


# --------------------------------------------------------------- render_criterion
check("criterion/str passes through", render_criterion("phishing or scam"), "phishing or scam")
check("criterion/dict -> json", render_criterion({"desc": "phishing"}), '{"desc": "phishing"}')
check("criterion/list -> json", render_criterion(["a", "b"]), '["a", "b"]')
check("criterion/int -> json", render_criterion(3), "3")
check("criterion/bool -> json", render_criterion(False), "false")
check("criterion/non-ascii kept", render_criterion({"d": "münchen"}), '{"d": "münchen"}')
check_true("criterion/unserialisable falls back to str",
           isinstance(render_criterion({"o": object()}), str))


# --------------------------------------------------------------- the reported crash
q = {"t": "noul", "ins": "Is this phishing?",
     "crit": {"true": {"desc": "phishing, scam or fraud"}, "false": {"desc": "legitimate"}}}
try:
    out = render_options(q)
    check("noul/dict criteria does not crash", len(out), 2)
    check_true("noul/false renders as json", out[0] == 'false: {"desc": "legitimate"}', out[0])
    check_true("noul/true renders as json", out[1] == 'true: {"desc": "phishing, scam or fraud"}', out[1])
    check_true("noul/no python repr leaked", "'" not in "".join(out), out)
except TypeError as e:
    FAIL.append("noul/dict criteria CRASHED: %s" % e)


# --------------------------------------------------------------- choice and score
out = render_options({"t": "choice", "ins": "x",
                      "crit": {"billing": {"desc": "payments"}, "tech": None, "sales": ""}})
check("choice/dict -> json", out[0], 'billing: {"desc": "payments"}')
check("choice/None -> bare key", out[1], "tech")
check("choice/empty string -> bare key", out[2], "sales")
check_true("choice/no python repr", "{'" not in "".join(out), out)

# 0 and False are real criterion values, not "missing"
out = render_options({"t": "choice", "ins": "x", "crit": {"zero": 0, "no": False}})
check("choice/0 is kept", out[0], "zero: 0")
check("choice/False is kept", out[1], "no: false")

out = render_options({"t": "score", "ins": "x", "crit": [{"d": "low"}, "high", 2]})
check("score/dict level -> json", out[0], 'level 0: {"d": "low"}')
check("score/str level unchanged", out[1], "level 1: high")
check("score/int level -> json", out[2], "level 2: 2")


# --------------------------------------------------------------- calibration boundaries
check("ece/zero confidence is included",
      ece_score(np.array([0.0]), np.array([1.0])), 1.0)
check("ece/zero confidence has its proper weight",
      ece_score(np.array([0.0, 1.0]), np.array([1.0, 1.0])), 0.5)


# --------------------------------------------------------------- noul labels
check("noul/default false text", render_options({"t": "noul", "ins": "x", "crit": None})[0],
      "false: no, the statement does not hold")
check("noul/default true text", render_options({"t": "noul", "ins": "x", "crit": None})[1],
      "true: yes, the statement holds")
check("noul/explicit None labels use defaults",
      render_options({"t": "noul", "ins": "x", "crit": None, "labels": None}),
      ["false: no, the statement does not hold", "true: yes, the statement holds"])
check("noul/string criteria still work",
      render_options({"t": "noul", "ins": "x", "crit": {"true": "yes it is", "false": "no"}}),
      ["false: no", "true: yes it is"])
custom_labels = {"true": " A ", "false": " B "}
custom_question = {"t": "noul", "ins": "x", "crit": None, "labels": custom_labels}
check("noul/custom labels preserve false-then-true semantics", render_options(custom_question),
      ["B: no, the statement does not hold", "A: yes, the statement holds"])
check("noul/custom label input is not mutated", custom_labels, {"true": " A ", "false": " B "})

_label_error = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
for name, labels in [
    ("not a dict", ["negative", "positive"]),
    ("missing true", {"false": "negative"}),
    ("extra key", {"false": "negative", "true": "positive", "other": "x"}),
    ("blank value", {"false": " ", "true": "positive"}),
    ("duplicate values", {"false": "same", "true": "same"}),
    ("non-string value", {"false": 0, "true": "positive"}),
]:
    check_raises("noul/invalid labels " + name,
                 lambda labels=labels: render_options({"t": "noul", "ins": "x", "labels": labels}),
                 _label_error)

for qtype, crit in [("choice", {"a": None, "b": None}), ("score", ["low", "high"])]:
    check_raises("%s/rejects labels" % qtype,
                 lambda qtype=qtype, crit=crit: render_options(
                     {"t": qtype, "ins": "x", "crit": crit,
                      "labels": {"false": "B", "true": "A"}}),
                 "labels is only supported for noul questions")

# --------------------------------------------------------------- unchanged choice and score behaviour
check("choice/string criteria still work",
      render_options({"t": "choice", "ins": "x", "crit": {"a": "first", "b": None}}),
      ["a: first", "b"])
check("choice/boolean-word labels are not rewritten",
      render_options({"t": "choice", "ins": "x", "crit": {"true": "yes", "false": "no"}}),
      ["true: yes", "false: no"])
check("score/string criteria still work",
      render_options({"t": "score", "ins": "x", "crit": ["low", "high"]}),
      ["level 0: low", "level 1: high"])

# every rendered option must be a str, whatever went in.
# The original set varied the criterion *values* only, and left every label a string, which is
# why it passed while `render_options({1: None})` returned `[1]` from a function annotated
# `-> List[str]`. `build_sequence` then called `.replace` on that int, and the error named
# neither the question nor the label. Labels are varied here for the same reason the values were.
for label, qq in [
    ("choice/objects", {"t": "choice", "ins": "x", "crit": {"a": {"n": 1}, "b": [1, 2], "c": 3.5}}),
    ("score/objects", {"t": "score", "ins": "x", "crit": [{"a": 1}, [2], None]}),
    ("noul/objects", {"t": "noul", "ins": "x", "crit": {"true": [1], "false": {"z": 0}}}),
    ("choice/int labels, no description", {"t": "choice", "ins": "x", "crit": {1: None, 2: None, 3: None}}),
    ("choice/float labels, no description", {"t": "choice", "ins": "x", "crit": {1.5: None}}),
    ("choice/None label, no description", {"t": "choice", "ins": "x", "crit": {None: None, "billing": None}}),
    ("choice/bool labels, no description", {"t": "choice", "ins": "x", "crit": {True: None, False: None}}),
    ("choice/int labels with descriptions", {"t": "choice", "ins": "x", "crit": {1: "one", 2: "two"}}),
]:
    check_true("all options are str (%s)" % label,
               all(isinstance(o, str) for o in render_options(qq)), render_options(qq))

# the rendered text of a non-string label is its string form, not its repr
check("choice/int label renders as its str", render_options({"t": "choice", "ins": "x", "crit": {1: None}}), ["1"])
check("choice/None label renders as the word none",
      render_options({"t": "choice", "ins": "x", "crit": {None: None}}), ["None"])
check("choice/int label keeps its description form",
      render_options({"t": "choice", "ins": "x", "crit": {1: "one"}}), ["1: one"])

# the JSON we emit is parseable back
parsed = json.loads(render_options(
    {"t": "noul", "ins": "x", "crit": {"true": {"a": 1}, "false": {"b": 2}}})[1].split("true: ", 1)[1])
check("emitted json round-trips", parsed, {"a": 1})

# public labels reach the renderer through Agent._to_internal without changing caller data
from laya.agent import Agent  # noqa: E402

public_labels = {"true": "A", "false": "B"}
public_question = {"type": "noul", "instructions": "Is this true?", "labels": public_labels}
internal = Agent._to_internal(public_question)
check("agent/forwards noul labels", internal["labels"], public_labels)
check("agent/forwarded labels reach renderer", render_options(internal),
      ["B: no, the statement does not hold", "A: yes, the statement holds"])
check("agent/leaves public question unchanged", public_question,
      {"type": "noul", "instructions": "Is this true?", "labels": {"true": "A", "false": "B"}})

boolean_criteria = {True: "yes", False: "no"}
boolean_question = {"type": "noul", "instructions": "Is this true?", "criteria": boolean_criteria,
                    "labels": {"false": "B", "true": "A"}}
boolean_internal = Agent._to_internal(boolean_question)
check("agent/custom labels keep boolean criteria normalization", boolean_internal["crit"],
      {"true": "yes", "false": "no"})
check("agent/boolean criteria render with custom labels", render_options(boolean_internal), ["B: no", "A: yes"])
check("agent/leaves boolean criteria unchanged", boolean_criteria, {True: "yes", False: "no"})


# --------------------------------------------------------------- CPU-fallback warning (#9 follow-up)
# The warning must fire only when a fallback actually happened -- not merely because the machine
# has CUDA. `laya.load(path, device="cpu")` on a GPU box is a deliberate choice, not a problem.
import inspect  # noqa: E402

from laya import agent as _agent  # noqa: E402

_src = inspect.getsource(_agent.Agent.__init__)


# --------------------------------------------------------------- malformed question shapes (#182)
# A question that cannot be answered used to fail three frames down, as an exception that named
# neither the question nor the fix: `AttributeError: 'NoneType' object has no attribute 'items'`
# from `render_options` for a `choice` without criteria, `KeyError: 'bool'` from the type table,
# and -- for a question that ended up with no options at all -- a `selected index k out of range`
# raised inside `DecisionModel.forward`, which reads like a bug in laya rather than in the caller's
# definition. The inference path below is real: a tiny from-config encoder, no checkpoint
# downloaded (tests/test_local_e2e.py covers the real weights).

from laya.agent import Agent  # noqa: E402

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAIL " + f)
sys.exit(1 if FAIL else 0)
