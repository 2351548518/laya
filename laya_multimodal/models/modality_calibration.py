"""Per-modality temperatures and ordinal calibration for native decisions."""
from collections import Counter, defaultdict
import hashlib
import math

import numpy as np
import torch

from ..decisions.calibration import MIN_TYPE_N, fit_temperature_map
from ..decisions.questions import QTYPES


def validate_ordinal_calibration(profile):
    if not isinstance(profile, dict):
        raise ValueError("ordinal calibration must be an object")
    if not profile:
        return profile
    models = profile.get("models")
    if not isinstance(models, dict):
        raise ValueError("ordinal calibration requires models")
    for key, model in models.items():
        parts = key.split(":") if isinstance(key, str) else []
        if (len(parts) != 2 or parts[0] not in {"image", "video", "audio", "text"}
                or not parts[1].isdigit() or int(parts[1]) < 2 or str(int(parts[1])) != parts[1]):
            raise ValueError("invalid ordinal modality/options bucket")
        if not isinstance(model, dict):
            raise ValueError("ordinal channel must be an object")
        kind = model.get("kind")
        if not isinstance(kind, str):
            raise ValueError("ordinal channel kind must be a string")
        size = {"identity": 0, "temperature": 1, "mix": 5, "location": 5}.get(kind)
        parameters, prior = model.get("parameters"), model.get("prior")
        if (size is None or not isinstance(parameters, list) or len(parameters) != size
                or not isinstance(prior, list) or len(prior) != int(parts[1])):
            raise ValueError("invalid ordinal channel parameters/prior")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
               for x in parameters + prior):
            raise ValueError("ordinal channel values must be finite numbers")
        if min(prior) < 0 or not math.isclose(sum(prior), 1., abs_tol=1e-6):
            raise ValueError("ordinal prior must be a probability distribution")
    return profile


def _softmax(values):
    values = values - np.max(values, axis=-1, keepdims=True)
    result = np.exp(values)
    return result / result.sum(axis=-1, keepdims=True)


def ordinal_probabilities(logits, model=None):
    values = np.asarray(logits, dtype=np.float64)
    if model is None or model["kind"] == "identity":
        return _softmax(values)
    parameters = model["parameters"]
    q = _softmax(values / np.exp(np.clip(parameters[0], -5, 5)))
    if model["kind"] == "temperature":
        return q
    levels = np.arange(values.shape[-1], dtype=np.float64)
    offset, gain = parameters[1], np.exp(np.clip(parameters[2], -5, 5))
    concentration = np.exp(np.clip(parameters[3], -5, 7))
    strength = float(_softmax(np.array([0., parameters[4]]))[1])
    centers = gain * levels + offset if model["kind"] == "mix" else gain * (q @ levels) + offset
    profile = _softmax(-concentration * (levels - centers[..., None]) ** 2)
    prediction = q @ profile if model["kind"] == "mix" else profile
    return strength * prediction + (1 - strength) * np.asarray(model["prior"])


def _channel_probability(logits, parameters, prior, kind):
    if kind == "identity":
        return logits.softmax(-1)
    q = (logits / parameters[0].clamp(-5, 5).exp()).softmax(-1)
    if kind == "temperature":
        return q
    levels = torch.arange(logits.shape[-1], dtype=logits.dtype, device=logits.device)
    gain, offset = parameters[2].clamp(-5, 5).exp(), parameters[1]
    centers = gain * levels + offset if kind == "mix" else gain * (q @ levels) + offset
    profile = (-parameters[3].clamp(-5, 7).exp() * (levels - centers[..., None]).square()).softmax(-1)
    prediction = q @ profile if kind == "mix" else profile
    return parameters[4].sigmoid() * prediction + (1 - parameters[4].sigmoid()) * prior


def _fit_channel(records, kind):
    logits = torch.tensor(np.asarray([r["logits"] for r in records]), dtype=torch.float64)
    target = torch.tensor(np.asarray([r["target"] for r in records]), dtype=torch.float64)
    prior = (target.sum(0) + 1) / (len(records) + target.shape[1])
    if kind == "identity":
        return dict(kind=kind, parameters=[], prior=prior.tolist())
    center = torch.tensor([math.log(2.), 0., 0., math.log(4.), math.log(19.)], dtype=torch.float64)
    deviations = torch.tensor([1., 1., .5, 1.5, 2.], dtype=torch.float64)
    parameters = (torch.zeros(1, dtype=torch.float64) if kind == "temperature" else center.clone()).requires_grad_()
    optimizer = torch.optim.LBFGS([parameters], lr=.5, max_iter=120, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        p = _channel_probability(logits, parameters, prior, kind)
        loss = -(target * p.clamp_min(1e-12).log()).sum(-1).mean()
        if kind != "temperature":
            loss = loss + .5 * ((parameters - center) / deviations).square().sum() / len(records)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite ordinal calibration objective")
        loss.backward()
        return loss

    optimizer.step(closure)
    return dict(kind=kind, parameters=parameters.detach().tolist(), prior=prior.tolist())


def fit_ordinal_calibration(records):
    from ..evaluation.native_metrics import metrics
    groups = defaultdict(lambda: [[], []])
    for record in records:
        if record["qtype"] != 1:
            continue
        k = record["k"]
        logits, target = np.asarray(record["logits"]), np.asarray(record["target"])
        if (logits.shape != (k,) or target.shape != (k,) or k < 2 or not np.isfinite(logits).all()
                or not np.isfinite(target).all() or target.min() < 0 or not np.isclose(target.sum(), 1.)):
            raise ValueError("invalid canonical ordinal calibration record")
        identity = record.get("calibration_group") or record.get("group")
        if not isinstance(identity, str) or not identity:
            raise ValueError("ordinal calibration requires explicit independent calibration groups")
        key = record["modality"] + ":" + str(k)
        checking = int(hashlib.sha256((identity + ":ordinal-channel-v2").encode()).hexdigest()[:8], 16) % 10 >= 7
        groups[key][int(checking)].append(record)
    models, evidence = {}, {}
    for key, (fitting, checking) in sorted(groups.items()):
        if len(fitting) < 40 or len(checking) < 15:
            evidence[key] = dict(fit_n=len(fitting), check_n=len(checking), selected="identity")
            continue
        candidates = []
        for kind in ("identity", "temperature", "mix", "location"):
            model = _fit_channel(fitting, kind)
            changed = [dict(r, logits=np.log(np.maximum(ordinal_probabilities(r["logits"], model), 1e-12)))
                       for r in checking]
            candidates.append(dict(model=model, metrics=metrics(changed)["overall"]))
        baseline = candidates[0]["metrics"]
        eligible = [item for item in candidates if item["metrics"]["accuracy"] >= baseline["accuracy"]
                    and item["metrics"]["score_mae"] <= baseline["score_mae"] + 1e-6
                    and item["metrics"]["nll"] <= baseline["nll"] + 1e-6]
        selected = min(eligible, key=lambda item: (item["metrics"]["nll"], item["metrics"]["score_mae"]))
        models[key] = selected["model"]
        evidence[key] = dict(fit_n=len(fitting), check_n=len(checking), candidates=candidates,
                             selected=selected["model"]["kind"])
    return validate_ordinal_calibration(dict(models=models, evidence=evidence,
        protocol="MAP ordinal channels; calibration groups split 70/30 by sha256(group + :ordinal-channel-v2)."))


def calibrate_multimodal(agent, records, *, ordinal_guard=False, ordinal_channels=False):
    """Fit temperatures on calibration data, optionally validating ordinal changes on separate groups.

    The ordinal guard uses only this calibration split. Its fit and check groups never consult
    development checkpoint-selection outcomes or test labels.
    """
    from ..evaluation.native_metrics import metrics

    pairs = lambda values: [(r["qtype"], r["logits"], r["target"], r["k"]) for r in values]
    calibration = fit_temperature_map(pairs(records))
    agent.temperature = calibration["temperature"]
    agent.temperature_by_options = calibration["temperature_by_options"]
    counts = Counter(r["qtype"] for r in records)
    agent.calibrated = bool(counts and all(n >= MIN_TYPE_N for n in counts.values()))
    agent.modality_calibration = {}
    agent.ordinal_calibration = {}
    for kind in sorted({r["modality"] for r in records}):
        subset = [r for r in records if r["modality"] == kind]
        fitted = fit_temperature_map(pairs(subset))
        fitted["fitted_types"] = [qt for qt, count in Counter(r["qtype"] for r in subset).items() if count >= MIN_TYPE_N]
        if ordinal_guard:
            score = [r for r in subset if r["qtype"] == QTYPES["score"]]
            folds = [[], []]
            for record in score:
                identity = str(record.get("calibration_group") or record.get("group") or record.get("example_hash") or record.get("id"))
                fold = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 2
                folds[fold].append(record)
            accepted, candidate, evidence = False, None, {"fit_n": len(folds[0]), "check_n": len(folds[1])}
            if min(map(len, folds)) >= MIN_TYPE_N:
                candidate = fit_temperature_map(pairs(folds[0]))
                plain = metrics(folds[1])["overall"]
                changed = metrics(folds[1], candidate["temperature"], candidate["temperature_by_options"])["overall"]
                accepted = changed["nll"] <= plain["nll"] + 1e-6 and changed["score_mae"] <= plain["score_mae"] + 1e-6
                evidence.update(raw=plain, candidate=changed)
            fitted["temperature_by_options"] = {key: value for key, value in fitted["temperature_by_options"].items()
                                                 if not key.startswith("score:")}
            fitted["temperature"][QTYPES["score"]] = candidate["temperature"][QTYPES["score"]] if accepted else 1.0
            if accepted:
                fitted["temperature_by_options"].update({key: value for key, value in candidate["temperature_by_options"].items()
                                                       if key.startswith("score:")})
            else:
                fitted["fitted_types"] = [qt for qt in fitted["fitted_types"] if qt != QTYPES["score"]]
            fitted["ordinal_guard"] = dict(evidence, accepted=accepted)
        agent.modality_calibration[kind] = fitted
    if ordinal_guard:
        # Unseen modalities do not inherit an unchecked global ordinal temperature.
        agent.temperature[QTYPES["score"]] = 1.0
        agent.temperature_by_options = {key: value for key, value in agent.temperature_by_options.items() if not key.startswith("score:")}
        calibration["temperature_by_options"] = agent.temperature_by_options
    calibration["by_modality"] = agent.modality_calibration
    if ordinal_channels:
        agent.ordinal_calibration = fit_ordinal_calibration(records)
        calibration["ordinal_calibration"] = agent.ordinal_calibration
    return calibration
