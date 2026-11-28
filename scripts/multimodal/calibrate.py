"""Fit local calibration temperatures and retain them only after held-out checks."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from laya_multimodal.models.multimodal import MultimodalAgent
from laya_multimodal.models.modality_calibration import calibrate_multimodal
from laya_multimodal.data import check_splits, load_rows
from laya_multimodal.evaluation.native_metrics import collect_records, metrics


def calibration_passes(raw, changed):
    failures = []
    for name, left, right in [('overall', raw['overall'], changed['overall'])] + [
            (mode, values, changed['by_modality'][mode]) for mode, values in raw['by_modality'].items()]:
        if not left['n']:
            continue
        if right['nll'] > left['nll'] + .02:
            failures.append(name + ':NLL')
        if right['ece'] > left['ece'] + .02:
            failures.append(name + ':ECE')
        if left.get('score_mae') is not None and right['score_mae'] > left['score_mae'] + .02:
            failures.append(name + ':ordinal_MAE')
    if changed['overall']['ece'] > raw['overall']['ece']:
        failures.append('overall:ECE_did_not_improve')
    return not failures, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device')
    parser.add_argument('--video-cache')
    args = parser.parse_args()
    rows, development = load_rows(args.calibration), load_rows(args.validation)
    check_splits(calibration=rows, development=development)
    agent = MultimodalAgent(args.model, device=args.device)
    if args.video_cache:
        agent._video_cache_dir = args.video_cache
    names = ('temperature', 'temperature_by_options', 'modality_calibration', 'ordinal_calibration', 'calibrated')
    original = {name: copy.deepcopy(getattr(agent, name)) for name in names}
    calibration = collect_records(agent, rows, 1)
    records = collect_records(agent, development, 1)
    raw = metrics(records, agent.temperature, agent.temperature_by_options,
        agent.modality_calibration, agent.ordinal_calibration)
    agent.ordinal_calibration = {}
    calibrate_multimodal(agent, calibration, ordinal_guard=True)
    candidate = metrics(records, agent.temperature, agent.temperature_by_options, agent.modality_calibration)
    accepted, failures = calibration_passes(raw, candidate)
    if not accepted:
        for name, value in original.items():
            setattr(agent, name, value)
    agent.save(args.out)
    report = dict(accepted=accepted, failures=failures, calibration_records=len(calibration),
        validation_records=len(records), before=raw, candidate=candidate)
    (args.out / 'calibration_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
