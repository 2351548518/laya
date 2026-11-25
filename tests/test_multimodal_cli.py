"""Portable training tools must work without local collection or experiment code."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from laya.grounding_train import GroundingTrainConfig
from scripts.multimodal import evaluate
from scripts.multimodal.calibrate import calibration_passes
from scripts.multimodal.train import validate_visual_training


ROOT = Path(__file__).resolve().parents[1]


def test_core_training_and_cli_have_no_research_dependencies():
    paths = list((ROOT / 'scripts/multimodal').glob('*.py'))
    paths += [ROOT / 'laya' / name for name in ['multimodal.py', 'multimodal_train.py',
        'grounding.py', 'grounding_train.py', 'multimodal_distributed.py', 'multimodal_objectives.py',
        'multimodal_readout.py', 'multimodal_calibration.py']]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [node.module or ''] if isinstance(node, ast.ImportFrom) else [
                name.name for name in node.names] if isinstance(node, ast.Import) else []
            assert not any(name.startswith(('research', '_local')) for name in modules), str(path)


@pytest.mark.parametrize('script', ['train.py', 'calibrate.py', 'evaluate.py'])
def test_cli_help_needs_no_local_datasets_or_weights(script):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/multimodal' / script), '--help'],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--model' in result.stdout and '--out' in result.stdout


def test_prediction_cli_loads_the_native_lane_for_text_and_media(tmp_path, monkeypatch, capsys):
    from PIL import Image
    from laya import cli, multimodal
    from test_multimodal import Q, tiny_agent

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "multimodal_config.json").write_text("{}", encoding="utf-8")
    image = tmp_path / "scene.png"
    Image.new("RGB", (12, 12), "red").save(image)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(Q), encoding="utf-8")
    sources = []
    agent = tiny_agent()

    def native(source, **kwargs):
        sources.append(source)
        return agent

    monkeypatch.setattr(multimodal, "MultimodalAgent", native)
    args = ["--multimodal", str(checkpoint), "--questions", str(questions), "--json"]
    for inputs in (["Current scene"], ["--image", str(image), "Current scene"]):
        assert cli.main([*args, *inputs]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["routing"]["model"] == "multimodal"
        assert result["usage"]["forward_passes"] == len(Q)
    batch = tmp_path / "requests.txt"
    batch.write_text("First scene\nSecond scene\n", encoding="utf-8")
    assert cli.main([*args, "--batch", str(batch)]) == 0
    results = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(results) == 2 and all(result["routing"]["model"] == "multimodal" for result in results)
    assert sources == [str(checkpoint)] * 3
    assert any(call.get("images") for call in agent.processor.calls)


def test_prediction_cli_reports_an_unavailable_native_checkpoint(tmp_path, capsys):
    from laya import cli

    assert cli.main(["--multimodal", str(tmp_path / "missing"), "Text"]) == 2
    assert "multimodal_config.json" in capsys.readouterr().err


def test_example_verifier_and_benchmark_execute_the_native_contract(tmp_path, monkeypatch, capsys):
    from PIL import Image
    import torch
    from benchmarks import bench_multimodal
    from examples import native_multimodal
    from verify import multimodal_smoke_test
    from test_multimodal import Q, tiny_agent

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "multimodal_config.json").write_text("{}", encoding="utf-8")
    image = tmp_path / "scene.png"
    Image.new("RGB", (112, 112), "red").save(image)
    agent = tiny_agent()
    agent.modality_calibration, agent.ordinal_calibration = {}, {}
    agent.grounding_trained = True

    def grounding_outputs(inputs, *, multiple=False):
        assert not multiple
        return {"boxes": torch.tensor([[[0.2, 0.2, 0.8, 0.8]]]), "quality_logits": torch.zeros(1, 1)}

    agent._grounding_outputs = grounding_outputs

    for module in (native_multimodal, multimodal_smoke_test, bench_multimodal, evaluate):
        monkeypatch.setattr(module, "MultimodalAgent", lambda *args, **kwargs: agent)
    native_multimodal.main([str(checkpoint), "--image", str(image)])
    example = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [result["modality"] for result in example] == ["text", "image"]
    assert example[1]["answers"]["target"]["type"] == "box"

    multimodal_smoke_test.main([str(checkpoint)])
    check = json.loads(capsys.readouterr().out)
    assert check["status"] == "passed"
    assert check["checks"]["text"]["types"] == ["choice", "noul", "score"]
    assert check["checks"]["image"]["types"] == ["noul", "box"]
    with pytest.raises(ValueError, match="token accounting"):
        multimodal_smoke_test.validate_prediction(example[0], "image", {"route": Q["category"]})

    workload = tmp_path / "requests.json"
    workload.write_text(json.dumps([{"id": "test", "state": "Current scene", "questions": Q}]), encoding="utf-8")
    output = tmp_path / "reports/latency.json"
    bench_multimodal.main(["--model", str(checkpoint), "--requests", str(workload),
                           "--repeats", "2", "--warmup", "0", "--out", str(output)])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == report
    measured = report["cases"][0]
    assert measured["usage"]["forward_passes"] == len(Q)
    assert measured["usage"]["output_tokens"] == 0
    assert measured["median_ms"] <= measured["p95_ms"] == max(measured["samples_ms"])

    decisions, boxes = tmp_path / "decisions.jsonl", tmp_path / "boxes.jsonl"
    decisions.write_text(json.dumps({"state": "Current scene", "questions": {"present": Q["present"]},
                                     "expected": {"present": True}}) + "\n", encoding="utf-8")
    boxes.write_text(json.dumps({
        "state": {"image": str(image)}, "questions": {"target": {"type": "box", "instructions": "the square"}},
        "grounding": {"task": "single", "boxes": [[0.2, 0.2, 0.8, 0.8]]},
    }) + "\n", encoding="utf-8")
    evaluation_output = tmp_path / "reports/accuracy.json"
    evaluate.main(["--model", str(checkpoint), "--decisions", str(decisions), "--grounding", str(boxes),
                   "--out", str(evaluation_output)])
    report = json.loads(evaluation_output.read_text(encoding="utf-8"))
    assert report["decisions"]["overall"]["n"] == report["grounding"]["single_n"] == 1
    assert report["grounding"]["acc_iou50"] == 1
    assert "harmonic_accuracy" in report and len(report["datasets"]["grounding"]["sha256"]) == 64

    from scripts.multimodal.prepare_dataset import main as prepare_dataset
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_bytes(decisions.read_bytes() + boxes.read_bytes())
    suite = tmp_path / "suite"
    prepare_dataset(["--split", "mixed=" + str(mixed), "--out", str(suite)])
    manifest = suite / "manifest.json"
    args = ["--model", str(checkpoint), "--suite-manifest", str(manifest), "--suite", "mixed",
            "--out", str(evaluation_output)]
    evaluate.main(args)
    frozen = json.loads(evaluation_output.read_text(encoding="utf-8"))
    assert frozen["decisions"] == report["decisions"] and frozen["grounding"] == report["grounding"]
    assert frozen["datasets"]["suite"]["name"] == "mixed"
    (suite / "mixed.jsonl").write_bytes(b"changed\n")
    with pytest.raises(SystemExit) as exc:
        evaluate.main(args)
    assert exc.value.code == 2


def test_harmonic_accuracy_requires_both_metrics_with_samples():
    decisions = {"overall": {"n": 10, "accuracy": 0.9}}
    assert evaluate.harmonic_accuracy(decisions, {"single_n": 8, "acc_iou50": 0.6}) == pytest.approx(0.72)
    assert evaluate.harmonic_accuracy(decisions, {"single_n": 0, "acc_iou50": 0.0}) is None
    assert evaluate.harmonic_accuracy({"overall": {"n": 1, "accuracy": 0}},
                                      {"single_n": 1, "acc_iou50": 0}) == 0.0


@pytest.mark.parametrize("dataset_flag", ["--decisions", "--grounding"])
def test_evaluation_output_cannot_replace_an_input_dataset(tmp_path, capsys, dataset_flag):
    dataset = tmp_path / "input.jsonl"
    dataset.write_bytes(b'{"state":"sample"}\n')
    output = tmp_path / "report.json"
    output.hardlink_to(dataset)
    for path in (dataset, output):
        with pytest.raises(SystemExit) as exc:
            evaluate.main(["--model", "unused", dataset_flag, str(dataset), "--out", str(path)])
        assert exc.value.code == 2 and "--out must refer to a separate file" in capsys.readouterr().err
    assert dataset.read_bytes() == b'{"state":"sample"}\n'


def test_portable_recipes_preserve_the_measured_optimizer_settings():
    head = json.loads((ROOT / 'configs/multimodal/head.json').read_text(encoding='utf-8'))
    fit = json.loads((ROOT / 'configs/multimodal/sft.json').read_text(encoding='utf-8'))
    for values in [head, fit]:
        GroundingTrainConfig(**values).validate()
        assert values['global_batch_size'] == 64 and values['micro_batch'] == 4
        assert values['slots'] == 1 and values['max_len'] == 16384
        assert not Path(values['video_cache_dir']).is_absolute()
    assert fit['learning_rate'] == 5e-5 and fit['head_learning_rate'] == 2e-4
    assert fit['decision_brier_weight'] == fit['decision_ordinal_weight'] == 0


def test_allowing_unadmitted_inputs_never_discards_audio_or_wrong_processor(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    questions = {'check': {'type': 'noul', 'instructions': 'Is the context relevant?'}}
    for name in ['train', 'dev', 'decision_dev']:
        (tmp_path / (name + '.jsonl')).write_text(json.dumps(dict(state='context', questions=questions)) + '\n')
    with pytest.raises(ValueError, match='processor admission'):
        validate_visual_training(tmp_path, model, 16384)
    validate_visual_training(tmp_path, model, 16384, require_admission=False)
    (tmp_path / 'train.jsonl').write_text(json.dumps(dict(state={'audio': 'voice.wav'}, questions=questions)) + '\n')
    with pytest.raises(ValueError, match='audio cannot be discarded'):
        validate_visual_training(tmp_path, model, 16384, require_admission=False)


def test_calibration_rejects_an_aggregate_gain_hiding_a_modality_regression():
    baseline = dict(n=10, nll=.3, ece=.05, score_mae=.4)
    raw = dict(overall=copy.deepcopy(baseline), by_modality={'image': copy.deepcopy(baseline)})
    changed = copy.deepcopy(raw)
    changed['overall']['ece'] = .04
    changed['by_modality']['image']['nll'] = .34
    assert not calibration_passes(raw, changed)[0]
    changed['by_modality']['image']['nll'] = .3
    assert calibration_passes(raw, changed) == (True, [])


def test_request_shaped_dataset_preserves_media_labels_and_option_order(tmp_path):
    from laya.grounding_train import load_grounding_rows
    from laya.multimodal_data import can_shuffle_choice, record_input
    from scripts.multimodal.prepare_dataset import main
    from test_multimodal import tiny_agent

    image = tmp_path / "photo.png"
    image.touch()
    source = tmp_path / "source.jsonl"
    box = [.1, .2, .5, .6]
    box_row = {"state": {"text": "Tabletop", "image": str(image)}, "id": "box-1",
               "questions": {"cup": {"type": "box", "instructions": "Find the cup"}},
               "grounding": {"task": "single", "boxes": [box], "ignored": []}}
    choice = {"type": "choice", "instructions": "Pick a response",
              "criteria": {"A": "accept", "B": "reject", "C": "none of the above"}, "option_order": [1, 0, 2]}
    choice_row = {"state": "Refund request", "id": "pick-1", "questions": {"action": choice},
                  "gold": {"action": {"probabilities": {"A": .75, "B": .25, "C": 0.}}}}
    source.write_text("\n".join(map(json.dumps, [box_row, choice_row])) + "\n")
    output = tmp_path / "dataset"
    main(["--training", "--split", "train=" + str(source), "--out", str(output)])
    saved, saved_choice = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert saved["expected"]["cup"] == {"type": "box", "confidence": None,
                                         "box": dict(left=.1, top=.2, right=.5, bottom=.6)}
    assert saved["request"]["assets"]
    wire_choice = saved_choice["request"]["questions"]["action"]
    assert "option_order" not in wire_choice and list(wire_choice["criteria"]) == ["B", "A", "C"]
    assert not Path(next(iter(saved["media_files"].values()))).is_absolute()
    box_record, choice_record = load_grounding_rows(output / "train.jsonl")
    assert box_record["grounding"] == box_row["grounding"]
    request, _ = record_input(tiny_agent(), box_record, "cup", box_row["questions"]["cup"])
    assert request["text"] == "Tabletop" and Path(request["image"][0]).resolve() == image
    assert choice_record["gold"] == choice_row["gold"]
    assert choice_record["questions"]["action"] == choice
    assert not can_shuffle_choice(choice)
    assert not can_shuffle_choice({key: value for key, value in choice.items() if key != "option_order"})
    assert can_shuffle_choice({"type": "choice", "instructions": "Pick a response", "criteria": ["accept", "reject"]})
