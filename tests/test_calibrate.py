"""Per-bucket temperature fitting. No model weights are loaded.

Records are CPU tuples (qtype, logits, target, k). The fitter is the notebook's NLL+LBFGS
on log T, grouped by `temp_bucket`. Bounds are `TEMP_MIN` / `TEMP_MAX` via `clamp_temperature`.
Logits for a labeled agent come from `_encode_state` and the raw rows `_decode_answers` scales.
"""
import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laya.calibrate import (  # noqa: E402
    ECE_HOLDOUT_FRAC,
    LOG_T_MARGIN,
    MIN_BUCKET_N,
    MIN_TYPE_N,
    MIN_BINNING_BUCKET_N,
    _clamp_entry,
    _ece_report,
    _ece_split,
    _fit_unclamped_temperature,
    _iter_records,
    _pairs_by_width,
    _softmax,
    apply_binning_map,
    fit_binning_map,
    fit_one_temperature,
    fit_temperature_map,
    fit_temperatures,
)
from laya.common import QTYPES, TEMP_MAX, TEMP_MIN, clamp_temperature, ece_score, temp_bucket  # noqa: E402
import laya.calibrate as _calibrate  # noqa: E402

PASS, FAIL = [], []

# Large enough that ECE_HOLDOUT_FRAC still leaves MIN_BUCKET_N rows to fit.
BUCKET_N = 2500


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


def peaked(k, idx, mag):
    z = np.full(k, -mag, dtype=np.float32)
    z[idx] = mag
    t = np.zeros(k, dtype=np.float32)
    t[idx] = 1.0
    return z, t


def overconfident_records(n, k, qtype, seed=0, acc=0.6, mag=8.0):
    rng = np.random.RandomState(seed)
    recs = []
    for i in range(n):
        y = int(rng.randint(0, k))
        pred = y if rng.rand() < acc else int((y + 1) % k)
        z = np.full(k, -mag, dtype=np.float32)
        z[pred] = mag
        t = np.zeros(k, dtype=np.float32)
        t[y] = 1.0
        recs.append((qtype, z, t, k))
    return recs


check("floor/MIN_BUCKET_N", MIN_BUCKET_N, 2000)
check("floor/MIN_TYPE_N", MIN_TYPE_N, 10)
check("clamp/TEMP_MIN", TEMP_MIN, 0.5)
check("clamp/TEMP_MAX", TEMP_MAX, 5.0)
check_true("clamp/no local TEMP_LO", not hasattr(_calibrate, "TEMP_LO"))
check_true("clamp/no local TEMP_HI", not hasattr(_calibrate, "TEMP_HI"))
check_true(
    "setup/holdout leaves a fittable bucket",
    BUCKET_N - int(round(BUCKET_N * ECE_HOLDOUT_FRAC)) >= MIN_BUCKET_N,
)


# --------------------------------------------------------------- overconfident -> T > 1
choice_k2 = overconfident_records(BUCKET_N, 2, QTYPES["choice"], seed=1, acc=0.6, mag=8.0)
t_choice = fit_one_temperature([(z, t) for _, z, t, _ in choice_k2])
check_true("overconfident/T>1", t_choice > 1.0, "T=%r" % t_choice)
check_true("overconfident/T<=TEMP_MAX", t_choice <= TEMP_MAX, "T=%r" % t_choice)


# --------------------------------------------------------------- bucket keys + omit n<MIN_BUCKET_N
noul_k2 = overconfident_records(BUCKET_N, 2, QTYPES["noul"], seed=2, acc=0.6, mag=8.0)
choice_k12 = overconfident_records(BUCKET_N, 12, QTYPES["choice"], seed=3, acc=0.6, mag=8.0)
# Exactly the per-bucket floor: fitted, but too small to hold any rows out for ECE.
choice_k4 = overconfident_records(MIN_BUCKET_N, 4, QTYPES["choice"], seed=6, acc=0.6, mag=8.0)
tiny_score = overconfident_records(5, 4, QTYPES["score"], seed=4, acc=0.6, mag=8.0)
mixed = choice_k2 + noul_k2 + choice_k12 + choice_k4 + tiny_score
fitted = fit_temperature_map(mixed, compute_ece=True)
keys = set(fitted["temperature_by_options"])
check_true("keys/choice:2", "choice:2" in keys, keys)
check_true("keys/noul:2", "noul:2" in keys, keys)
check_true("keys/choice:11+", "choice:11+" in keys, keys)
check_true("keys/choice:3-5 at floor", "choice:3-5" in keys, keys)
check("keys/temp_bucket choice:2", temp_bucket(QTYPES["choice"], 2), "choice:2")
check("keys/temp_bucket noul:2", temp_bucket(QTYPES["noul"], 2), "noul:2")
check("keys/temp_bucket choice:11+", temp_bucket(QTYPES["choice"], 12), "choice:11+")
check_true("omit/n<MIN_BUCKET_N score:3-5 not in map", "score:3-5" not in keys, keys)
check("omit/n_by_bucket still counts tiny", fitted["n_by_bucket"].get("score:3-5"), 5)
check("omit/n_by_bucket counts floor bucket", fitted["n_by_bucket"].get("choice:3-5"), MIN_BUCKET_N)
check_true("omit/not NaN", all(np.isfinite(v) for v in fitted["temperature_by_options"].values()))
check_true("alias/fit_temperatures is fit_temperature_map", fit_temperatures is fit_temperature_map)


# --------------------------------------------------------------- type floor stays below the bucket floor
# 50 is above MIN_TYPE_N and below MIN_BUCKET_N, so the scalar moves and the bucket is omitted.
medium_score = overconfident_records(50, 4, QTYPES["score"], seed=5, acc=0.6, mag=8.0)
med = fit_temperature_map(medium_score, compute_ece=False, seed=0)
med_other_seed = fit_temperature_map(medium_score, compute_ece=False, seed=1)
check_true(
    "type-floor/fitted below bucket floor",
    med["temperature"][QTYPES["score"]] > 1.0,
    med["temperature"],
)
check_true("type-floor/bucket omitted", "score:3-5" not in med["temperature_by_options"])
check_true("ece/off has no report", "report" not in med)
check("ece/off ignores seed", med["temperature"], med_other_seed["temperature"])


# --------------------------------------------------------------- clamp 0.5-5
# 50% accurate, extreme logits: NLL wants a large T, clamped at TEMP_MAX.
rng = np.random.RandomState(7)
hi_pairs = []
for i in range(MIN_BUCKET_N):
    y = int(rng.randint(0, 2))
    pred = y if i % 2 == 0 else 1 - y
    z, t = peaked(2, pred, 80.0)
    t = np.zeros(2, dtype=np.float32)
    t[y] = 1.0
    hi_pairs.append((z, t))
t_hi = fit_one_temperature(hi_pairs)
check_true("clamp/high in [0.5, 5]", TEMP_MIN <= t_hi <= TEMP_MAX, "T=%r" % t_hi)
check_true("clamp/high at TEMP_MAX", abs(t_hi - TEMP_MAX) < 1e-4, "T=%r" % t_hi)

# Always-correct, mild logits: NLL wants a small T, clamped at TEMP_MIN.
lo_pairs = []
for i in range(MIN_BUCKET_N):
    z, t = peaked(2, i % 2, 0.3)
    lo_pairs.append((z, t))
t_lo = fit_one_temperature(lo_pairs)
check_true("clamp/low in [0.5, 5]", TEMP_MIN <= t_lo <= TEMP_MAX, "T=%r" % t_lo)
check_true("clamp/low at TEMP_MIN", abs(t_lo - TEMP_MIN) < 1e-4, "T=%r" % t_lo)

src = inspect.getsource(fit_one_temperature)
check_true("clamp/source uses clamp_temperature", "clamp_temperature" in src)
check_true("clamp/source uses TEMP_MIN", "TEMP_MIN" in src)
check_true("clamp/source uses TEMP_MAX", "TEMP_MAX" in src)
check_true("clamp/n<MIN_BUCKET_N returns 1.0", fit_one_temperature(lo_pairs[:5]) == 1.0)
check_true(
    "clamp/n=MIN_BUCKET_N-1 returns 1.0",
    fit_one_temperature(lo_pairs[: MIN_BUCKET_N - 1]) == 1.0,
)


# --------------------------------------------------- ragged pools, pad columns, runaway scales
# A type-level fit pools every option count that type has seen, so `fit_one_temperature`
# regularly receives rows of different lengths. Two things used to cost such a pool its
# answer, and both surface the same way -- the neutral 1.0, or a bound, instead of a fit:
#   1. squaring off the short rows put the pad value in the same softmax those rows run
#      through, so a row whose own logits sat under it stopped being about that row;
#   2. the search runs on log T with no bound, and a pool whose optimum is below
#      `TEMP_MIN` walks `log_t` past float32's `exp()` underflow, divides the objective
#      by zero, and hands `clamp_temperature` a NaN, which it reports as 1.0.
def _spread_pool(width, scale, n, seed=0):
    """`n` argmax-correct rows at one width, re-seeded per width."""
    rs = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        z = list(rs.randn(width) * scale)
        t = [0.0] * width
        t[int(np.argmax(z))] = 1.0
        out.append((z, t))
    return out


def _mixed_rows(k, rng, acc, mag, n=400):
    """`k`-option rows that are right with probability `acc`, logits at +-mag."""
    out = []
    for _ in range(n):
        y = int(rng.randint(0, k))
        pred = y if rng.rand() < acc else int(rng.randint(0, k))
        z, _t = peaked(k, pred, mag)
        t = np.zeros(k, dtype=np.float32)
        t[y] = 1.0
        out.append((z, t))
    return out


two, nine = _spread_pool(2, 1.0, 24), _spread_pool(9, 1.0, 24)
t_two = fit_one_temperature(two, min_n=10)
t_nine = fit_one_temperature(nine, min_n=10)
t_ragged = fit_one_temperature(two + nine, min_n=10)
check_true("runaway/every width on its own fits the floor", (t_two, t_nine) == (0.5, 0.5),
           "two=%r nine=%r" % (t_two, t_nine))
check_true("runaway/ragged pool is a number", bool(np.isfinite(t_ragged)), "T=%r" % t_ragged)
check_true("runaway/ragged pool is not the neutral 1.0", t_ragged != 1.0, "T=%r" % t_ragged)
check_true("runaway/ragged pool keeps the floor", abs(t_ragged - TEMP_MIN) < 1e-9,
           "T=%r" % t_ragged)

# The same runaway at one width: only the walk to zero was wrong, not the raggedness.
t_soft = fit_one_temperature(_mixed_rows(3, np.random.RandomState(51), 0.7, 0.3), min_n=10)
check_true("runaway/low-spread pool is not the neutral 1.0", t_soft != 1.0, "T=%r" % t_soft)
check_true("runaway/low-spread pool keeps the floor", abs(t_soft - TEMP_MIN) < 1e-9,
           "T=%r" % t_soft)

# Rows whose every logit sits under the pad value: pooled, they used to report the ceiling.
low_rows = [([-20000.0 - i, -20001.0 - i], [1.0, 0.0]) for i in range(12)]
t_low_alone = fit_one_temperature(low_rows, min_n=10)
t_low_pool = fit_one_temperature(low_rows + _spread_pool(4, 1.0, 12), min_n=10)
check_true("pad/low rows alone fit the floor", abs(t_low_alone - TEMP_MIN) < 1e-9,
           "T=%r" % t_low_alone)
check("pad/low rows pooled fit what they fit alone", t_low_pool, t_low_alone)
check_true("pad/pool is not the ceiling", t_low_pool != TEMP_MAX, "T=%r" % t_low_pool)
check("pad/one matrix per width, narrowest first",
     [int(z.shape[1]) for z, _ in _pairs_by_width(low_rows + _spread_pool(4, 1.0, 12))], [2, 4])
check("pad/row totals survive the split",
     sum(int(z.shape[0]) for z, _ in _pairs_by_width(low_rows + _spread_pool(4, 1.0, 12))), 24)

# What a question type stores through the public API for the same rows.
ragged_recs = ([(1, z, t, len(z)) for z, t in low_rows * 34]
               + [(1, z, t, len(z)) for z, t in _spread_pool(5, 1.0, 400, 1)])
t_public = fit_temperature_map(ragged_recs)["temperature"][1]
check_true("public/ragged type keeps the floor", abs(t_public - TEMP_MIN) < 1e-9,
           "T=%r" % t_public)
check_true("public/ragged type is neither neutral nor the ceiling",
           t_public not in (1.0, TEMP_MAX), "T=%r" % t_public)

# Bounding the search must not flatten a minimum that sits inside the band: the fit is
# checked against the objective it is supposed to minimise, in float64, not against a
# hard-coded number.
mid = _mixed_rows(20, np.random.RandomState(293), 0.7, 1.0)
t_mid = fit_one_temperature(mid, min_n=10)


def _nll_at(rows, scale):
    z = np.asarray([r[0] for r in rows], dtype=np.float64) / float(scale)
    t = np.asarray([r[1] for r in rows], dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(1, keepdims=True)
    return float(-(t * np.log(np.clip(p, 1e-300, None))).sum(1).mean())


check_true("band/interior fit stays interior", TEMP_MIN < t_mid < TEMP_MAX, "T=%r" % t_mid)
check_true("band/interior fit beats the floor", _nll_at(mid, t_mid) < _nll_at(mid, TEMP_MIN),
           "T=%r nll=%r floor=%r" % (t_mid, _nll_at(mid, t_mid), _nll_at(mid, TEMP_MIN)))
check_true("band/interior fit beats either side of itself",
           _nll_at(mid, t_mid) < min(_nll_at(mid, t_mid * 1.1), _nll_at(mid, t_mid / 1.1)))
src = inspect.getsource(fit_one_temperature)
check_true("band/search is bounded by LOG_T_MARGIN", "LOG_T_MARGIN" in src)
check_true("band/the bound stops before float32 exp() underflows",
           float(np.log(TEMP_MIN)) - LOG_T_MARGIN > -80.0,
           "lower edge=%r" % (float(np.log(TEMP_MIN)) - LOG_T_MARGIN))

# Grouping the rows must not turn the objective into an average of per-width averages: it
# still averages over rows, so repeating one width's rows pulls the fit toward that width.
wide12 = _mixed_rows(12, np.random.RandomState(11), 0.6, 1.0, 8)
narrow2 = _mixed_rows(2, np.random.RandomState(12), 0.95, 4.0, 8)
t_wide = fit_one_temperature(wide12, min_n=8)
t_narrow = fit_one_temperature(narrow2, min_n=8)
t_even = fit_one_temperature(wide12 + narrow2, min_n=8)
t_heavy = fit_one_temperature(wide12 * 2 + narrow2, min_n=8)
check_true("weight/the two widths want different scales", abs(t_wide - t_narrow) > 0.3,
           "wide=%r narrow=%r" % (t_wide, t_narrow))
check_true("weight/both fits are interior, not a bound",
           all(TEMP_MIN < x < TEMP_MAX for x in (t_wide, t_narrow, t_even, t_heavy)),
           "%r %r %r %r" % (t_wide, t_narrow, t_even, t_heavy))
check_true("weight/doubling a width pulls the fit to that width",
           abs(t_heavy - t_wide) < abs(t_even - t_wide) and abs(t_heavy - t_even) > 0.2,
           "even=%r heavy=%r wide=%r" % (t_even, t_heavy, t_wide))


# ------------------------------------------------- the clamp is reported, not only applied
# A clamped fit and a fit whose optimum really is the bound are the same number, so the
# number cannot say which happened. `temperature_clamped` carries the unclamped scale
# beside it. The value to install is still the clamped one, so these checks also pin that
# the report changes nothing about `temperature` or `temperature_by_options`.
lo_recs = [(QTYPES["choice"], z, t, 2) for z, t in lo_pairs]
hi_recs = [(QTYPES["choice"], z, t, 2) for z, t in hi_pairs]
mid_recs = [(QTYPES["choice"], z, t, len(z)) for z, t in mid]
lo_map, hi_map, mid_map = (fit_temperature_map(r) for r in (lo_recs, hi_recs, mid_recs))

check_true("clamp-report/the key is always present",
           all("temperature_clamped" in m for m in (lo_map, hi_map, mid_map, med, fitted)))
check("clamp-report/an interior fit is not reported", mid_map["temperature_clamped"], {})
check_true("clamp-report/interior fit really was interior",
           TEMP_MIN < mid_map["temperature"][QTYPES["choice"]] < TEMP_MAX,
           "T=%r" % mid_map["temperature"][QTYPES["choice"]])

for _name, _m, _bound, _applied in (("low", lo_map, "min", TEMP_MIN), ("high", hi_map, "max", TEMP_MAX)):
    _rep = _m["temperature_clamped"]
    # Both the bucket and the type scalar are fit from the same rows here, so both are named,
    # the bucket by its `temperature_by_options` key and the type by its question-type name.
    check("clamp-report/%s names the bucket and the type" % _name,
          sorted(_rep), ["choice", "choice:2"])
    check("clamp-report/%s bound" % _name, sorted({e["bound"] for e in _rep.values()}), [_bound])
    check("clamp-report/%s applied is the clamped value" % _name,
          sorted({e["applied"] for e in _rep.values()}), [_applied])
    check("clamp-report/%s applied is what the result installs" % _name,
          _rep.get("choice:2", {}).get("applied"), _m["temperature_by_options"]["choice:2"])
    check_true("clamp-report/%s fitted is outside the band" % _name,
               all(not TEMP_MIN <= e["fitted"] <= TEMP_MAX for e in _rep.values()),
               "%r" % _rep)
    check_true("clamp-report/%s fitted is on the right side" % _name,
               all((e["fitted"] < TEMP_MIN) is (_bound == "min") for e in _rep.values()),
               "%r" % _rep)

# The clamped value stays the public answer: the report is additive, not a new return path.
check("clamp-report/fit_one_temperature is unchanged at the floor",
      fit_one_temperature(lo_pairs), lo_map["temperature_by_options"]["choice:2"])
check("clamp-report/fit_one_temperature is unchanged at the ceiling",
      fit_one_temperature(hi_pairs), hi_map["temperature_by_options"]["choice:2"])
check_true("clamp-report/the unclamped fit is the clamped one's source",
           abs(clamp_temperature(_fit_unclamped_temperature(lo_pairs), TEMP_MIN, TEMP_MAX)
               - fit_one_temperature(lo_pairs)) < 1e-12)
# The underflow case is a NaN, which `clamp_temperature` reports as the neutral 1.0. That is a
# different fact from a clamp and must not be dressed up as one.
check("clamp-report/a NaN fit is not a clamp", _clamp_entry(float("nan"), 1.0), None)
check("clamp-report/an infinite fit is not a clamp", _clamp_entry(float("inf"), TEMP_MAX), None)
check("clamp-report/a fit on the bound itself is not a clamp", _clamp_entry(TEMP_MIN, TEMP_MIN), None)
# The search bound lives with the search now; the prose in `fit_one_temperature` must not be
# what satisfies this.
check_true("clamp-report/the search bound is in the fitting function",
           "LOG_T_MARGIN" in inspect.getsource(_fit_unclamped_temperature))


# --------------------------------------------------------------- ECE on a held-out split, not the fit rows
report = fitted["report"]
check_true("ece/after < before", report["ece_after"] < report["ece_before"],
           "before=%r after=%r" % (report["ece_before"], report["ece_after"]))
check_true("ece/before finite", np.isfinite(report["ece_before"]))
check_true("ece/after finite", np.isfinite(report["ece_after"]))
check("ece/n is the full input", report["n"], float(len(mixed)))

parsed = _iter_records(mixed)
fit_recs, eval_recs, excluded = _ece_split(parsed, 0)
fit_ids = [id(r) for r in fit_recs]
eval_ids = [id(r) for r in eval_recs]
check_true("split/disjoint", set(fit_ids).isdisjoint(eval_ids))
check("split/partition", len(fit_ids) + len(eval_ids), len(parsed))
again_fit, again_eval, again_ex = _ece_split(parsed, 0)
check("split/deterministic eval", [id(r) for r in again_eval], eval_ids)
check("split/deterministic excluded", again_ex, excluded)
other_fit, other_eval, _other_ex = _ece_split(parsed, 1)
check_true("split/seed changes holdout", [id(r) for r in other_eval] != eval_ids)
check_true("split/floor bucket excluded", "choice:3-5" in excluded, excluded)
check_true("split/tiny excluded", "score:3-5" in excluded, excluded)
check_true("split/big bucket held out", "choice:2" not in excluded, excluded)
check("ece/n_eval", report["n_eval"], float(len(eval_recs)))
check_true("ece/n_eval < n", report["n_eval"] < report["n"])
check("ece/excluded noted", report["buckets_excluded_from_eval"], excluded)
# The map itself is the one fit on the non-held-out rows.
refit = fit_temperature_map(fit_recs, compute_ece=False)
check("split/fit ignores eval buckets", fitted["temperature_by_options"], refit["temperature_by_options"])
check("split/fit ignores eval types", fitted["temperature"], refit["temperature"])
holdout_ece = _ece_report(eval_recs, fitted["temperature"], fitted["temperature_by_options"])
check("ece/after is the holdout", report["ece_after"], holdout_ece["ece_after"])
check("ece/before is the holdout", report["ece_before"], holdout_ece["ece_before"])
full_ece = _ece_report(parsed, fitted["temperature"], fitted["temperature_by_options"])
check_true(
    "ece/not scored on the fit pool",
    full_ece["ece_after"] != report["ece_after"],
    "full=%r holdout=%r" % (full_ece["ece_after"], report["ece_after"]),
)
again_fit_map = fit_temperature_map(mixed, compute_ece=True, seed=0)
check("ece/fit deterministic", again_fit_map["report"], fitted["report"])


# --------------------------------------------------------------- histogram-binning recalibration
check("binning/floor", MIN_BINNING_BUCKET_N, 200)
bmap = fit_binning_map(mixed, fitted["temperature"], fitted["temperature_by_options"], bins=10)
check_true("binning/keyed by temp_bucket", {"choice:2", "noul:2", "choice:11+", "choice:3-5"} <= set(bmap), set(bmap))
check_true("binning/omits sub-floor bucket (score:3-5, n=5)", "score:3-5" not in bmap, set(bmap))
_entry = bmap["choice:2"]
check("binning/records bins", _entry["bins"], 10)
check("binning/one value per bin", len(_entry["values"]), 10)
check_true("binning/values are probabilities", all(0.0 <= v <= 1.0 for v in _entry["values"]), _entry["values"])

# apply: a bucket with no map returns the confidence unchanged; edges clamp into range.
check("binning/apply unknown bucket is identity", apply_binning_map(0.73, "choice:6-10", bmap), 0.73)
check_true("binning/apply in range for conf=1.0", 0.0 <= apply_binning_map(1.0, "choice:2", bmap) <= 1.0)
check_true("binning/apply in range for conf=0.0", 0.0 <= apply_binning_map(0.0, "choice:2", bmap) <= 1.0)

# --------------------------------------------------------------- an empty bin is a centre, not a pass-through
# The line above is the ONLY identity this map has: a bucket it never saw. An empty *bin* -- a
# confidence range the calibration set never produced, inside a bucket it did -- is not identity, and
# `fit_binning_map`'s docstring said it was ("maps to its own midpoint, i.e. leaves that region
# unchanged"). `values.append((b + 0.5) / bins)` rewrites every value in the region to one number, so
# a confidence in an untouched bin moves by up to half a bin width. Someone reasoning about a gate
# near the cut from the old wording would expect the number to survive it. These checks hold the
# rewritten-to-the-centre behaviour, its bound, and the one thing the sentence was right to promise
# (never a fabricated 0), and pin the docstring against them.
def _at_conf(k, conf):
    """One record whose temperature-1.0 confidence is exactly `conf`, with argmax 0 and gold 0."""
    L = float(np.log(conf * (k - 1) / (1 - conf)))
    z = np.zeros(k, dtype=np.float32)
    z[0] = L
    t = np.zeros(k, dtype=np.float32)
    t[0] = 1.0
    return (QTYPES["choice"], z, t, k)


_EB_BINS = 10
_EB_BUCKET = temp_bucket(QTYPES["choice"], 3)  # "choice:3-5"
# Every record at conf 0.65: bin 6 of 10 holds them all, the other nine are empty by construction.
emap = fit_binning_map([_at_conf(3, 0.65)] * MIN_BINNING_BUCKET_N, [1.0, 1.0, 1.0], {},
                       bins=_EB_BINS, min_bucket_n=MIN_BINNING_BUCKET_N)
_eb = emap[_EB_BUCKET]["values"]
check("empty-bin/filled bin holds its empirical accuracy", _eb[6], 1.0)
check("empty-bin/an untouched bin holds its centre", _eb[0], (0 + 0.5) / _EB_BINS)
check_true("empty-bin/exactly the filled bin is not a centre",
           [b for b, v in enumerate(_eb) if v == (b + 0.5) / _EB_BINS] == [0, 1, 2, 3, 4, 5, 7, 8, 9], _eb)
for _b, _probe in ((0, 0.01), (0, 0.0999), (9, 0.91), (9, 0.9999)):
    _got = apply_binning_map(_probe, _EB_BUCKET, emap)
    check("empty-bin/conf %g in bin %d comes back as the centre" % (_probe, _b),
          _got, (_b + 0.5) / _EB_BINS)
    check_true("empty-bin/conf %g in bin %d is not the value it came in as" % (_probe, _b),
               _got != _probe, "unchanged at %.6f" % _got)
# The bound the centre gives, inside a bin the map never filled: no value moves further than half a
# bin width, and none lands on 0. Bin 6 is excluded because it IS filled -- its value is an empirical
# accuracy that can sit anywhere in [0, 1], which is the whole point of the map (measured below by
# "binning/lowers ECE vs temperature on the bucket").
def _bin_of(c):
    return min(_EB_BINS - 1, max(0, int(c * _EB_BINS)))


_empty_probes = [i / 1000.0 for i in range(1, 1000) if _bin_of(i / 1000.0) != 6]
check_true("empty-bin/an untouched bin moves a value by at most half a bin width",
           all(abs(apply_binning_map(c, _EB_BUCKET, emap) - c) <= 0.5 / _EB_BINS + 1e-12
               for c in _empty_probes))
check_true("empty-bin/no confidence is recalibrated to 0",
           all(apply_binning_map(c, _EB_BUCKET, emap) > 0.0 for c in _empty_probes))

# The docstring, held against the behaviour above (the same way
# `tests/test_conformal_abstention.py` holds `check_min_confidence_map`'s prose to its refusals).
# Both directions: a reworded sentence fails the positive checks, and a re-adopted "unchanged" claim
# fails the negative one while the behaviour checks above keep failing too.
_eb_doc = " ".join(inspect.getdoc(fit_binning_map).split())
check_true("doc/the docstring says an empty bin maps to the bin centre",
           "maps to that bin's centre" in _eb_doc, _eb_doc)
check_true("doc/the docstring says the value is rewritten", "the value is still rewritten" in _eb_doc, _eb_doc)
check_true("doc/the docstring promises no pass-through for an empty bin",
           "leaves that region unchanged" not in _eb_doc and "unchanged, so an unseen value" not in _eb_doc,
           _eb_doc)
check_true("doc/the docstring names the missing bucket as the identity case",
           "only a bucket the map has no entry for" in _eb_doc, _eb_doc)

# Binning reduces ECE where temperature alone cannot: recalibrate the temperature-scaled
# confidences of a bucket and compare ECE on the same data.
_qt = QTYPES["choice"]
_tscale = fitted["temperature_by_options"].get("choice:2", fitted["temperature"][_qt])
_cal_conf, _corr = [], []
for qt, z, t, k in _iter_records(choice_k2):
    p = _softmax(z[:k], _tscale)
    _cal_conf.append(float(p.max()))
    _corr.append(bool(int(p.argmax()) == int(np.argmax(t[:k]))))
_binned = [apply_binning_map(c, "choice:2", bmap) for c in _cal_conf]
_ece_temp = ece_score(np.asarray(_cal_conf), np.asarray(_corr))
_ece_binned = ece_score(np.asarray(_binned), np.asarray(_corr))
check_true("binning/lowers ECE vs temperature on the bucket",
           _ece_binned < _ece_temp, "temp=%.4f binned=%.4f" % (_ece_temp, _ece_binned))

try:
    fit_binning_map(mixed, fitted["temperature"], fitted["temperature_by_options"], bins=0)
    check_true("binning/bins must be >= 1", False, "no ValueError")
except ValueError:
    check_true("binning/bins must be >= 1", True)

# `bins` and `min_bucket_n` are counts, and this module already has a count rule -- `min_n` in
# `fit_one_temperature`, `k` in `_validated_pair`, and the `"bins"` of every `binning_map` entry a
# calibration payload carries. Only `bins < 1` was applied to these two, so everything below either
# escaped the guard as a `TypeError` from `range()` / a numpy cast, or -- the arms that were
# *accepted* -- came back as a map. Checked message and all, because "raised something" was exactly
# what was wrong.
for _name, _bad in [("bins", 0), ("bins", -3), ("bins", 2.5), ("bins", 10.0), ("bins", float("nan")),
                    ("bins", float("inf")), ("bins", True), ("bins", False), ("bins", None),
                    ("bins", "10"), ("min_bucket_n", 0), ("min_bucket_n", -3),
                    ("min_bucket_n", 2.5), ("min_bucket_n", float("nan")), ("min_bucket_n", True),
                    ("min_bucket_n", None)]:
    try:
        fit_binning_map(mixed, fitted["temperature"], fitted["temperature_by_options"],
                        **{_name: _bad})
        _res = "no ValueError"
    except ValueError as _exc:
        _res = str(_exc)
    except Exception as _exc:
        _res = "raised %s" % type(_exc).__name__
    check("binning/%s refuses %r" % (_name, _bad), _res,
          "%s must be a positive integer, got %r" % (_name, _bad))

# The two arms the old guard let through, and what they cost. `numpy.int64` is an `Integral`, so it
# passed `bins < 1`, fitted happily, and put a numpy scalar in `"bins"` that `save_calibration`'s
# `json.dump` refuses to serialise. `True` is an `Integral` too: one bin, and a map

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for failure in FAIL:
    print("  FAIL " + failure)
sys.exit(1 if FAIL else 0)
