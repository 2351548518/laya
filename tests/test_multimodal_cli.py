"""Portable training tools must work without local collection or experiment code."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from laya_multimodal.train import GroundingTrainConfig
from scripts.multimodal import evaluate
from scripts.multimodal.calibrate import calibration_passes
from scripts.multimodal.train import validate_visual_training


ROOT = Path(__file__).resolve().parents[1]


def test_core_training_and_cli_have_no_research_dependencies():
    paths = list((ROOT / 'scripts/multimodal').glob('*.py'))
    paths += [ROOT / 'laya_multimodal' / name for name in ['multimodal.py', 'multimodal_train.py',
        'grounding.py', 'train.py', 'train_cli.py', 'multimodal_distributed.py', 'multimodal_objectives.py',
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


def test_checkout_script_uses_the_packaged_entrypoint():
    from laya_multimodal import train_cli
    from scripts.multimodal import train

    assert train.main is train_cli.main
    assert validate_visual_training is train_cli.validate_visual_training


def test_train_command_dispatches_to_joint_trainer(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from laya_multimodal import cli, train_cli

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model_type": "gemma4"}')
    config = tmp_path / "config.json"
    config.write_text('{"mode": "head", "train_readout_in_warmup": true}')
    data, out = tmp_path / "data", tmp_path / "head"
    agent = SimpleNamespace(
        model=SimpleNamespace(grounding_head=SimpleNamespace(architecture="single-v1")),
        readout="candidate", prompt_version="v2",
    )
    run = Mock()
    monkeypatch.setattr(train_cli, "MultimodalAgent", Mock(return_value=agent))
    monkeypatch.setattr(train_cli, "distributed_context",
                        lambda: SimpleNamespace(device="cpu", main=True, world_size=1))
    monkeypatch.setattr(train_cli, "finetune_grounding", run)

    cli.main(["train", "--model", str(base), "--data", str(data), "--out", str(out),
              "--config", str(config), "--train-split", "warmup", "--require-original-base"])

    run.assert_called_once()
    assert run.call_args.args == (data / "warmup.jsonl", data / "dev.jsonl", str(out))
    assert run.call_args.kwargs["decision_dev_data"] == data / "decision_dev.jsonl"
    assert run.call_args.kwargs["config"].train_readout_in_warmup
    assert run.call_args.kwargs["agent"] is agent
    assert (out / "initialization.json").is_file()


def test_prediction_cli_loads_the_native_lane_for_text_and_media(tmp_path, monkeypatch, capsys):
    from PIL import Image
    from laya_multimodal import cli, multimodal
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
    from laya_multimodal import cli

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
    from laya_multimodal.train import load_grounding_rows
    from laya_multimodal.multimodal_data import can_shuffle_choice, record_input
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


@pytest.mark.parametrize(("assets", "files", "error"), [
    ({"scene": {"type": "audio", "part": "file"}}, {"file": "scene.jpg"}, "declare type image"),
    ({"scene": {"type": "image", "part": "file"}, "view": {"type": "image", "part": "file"}},
     {"file": "scene.jpg"}, "distinct file part"),
    ({"scene": {"type": "image", "part": "file"}}, {"file": "scene.jpg", "orphan": "other.jpg"},
     "matching asset"),
])
def test_request_shaped_dataset_checks_file_parts(tmp_path, assets, files, error):
    from laya_multimodal.multimodal_data import normalize_record

    row = {"request": {"state": "scene", "assets": assets,
                       "questions": {"x": {"type": "noul", "instructions": "Is the scene clear?"}}},
           "media_files": files, "expected": {"x": True}}
    with pytest.raises(ValueError, match=error):
        normalize_record(row, tmp_path, verify_media=False)


def test_swanlab_mirror_preserves_existing_steps_and_separates_resumes(tmp_path):
    from types import SimpleNamespace
    from scripts.multimodal.monitor_swanlab import mirror

    (tmp_path / "sft").mkdir()
    (tmp_path / "launch.json").write_text(json.dumps({"schedule": {"sft": 20}}))
    (tmp_path / "sft-config.json").write_text(json.dumps({"global_batch_size": 64, "video_cache_dir": "/cache"}))
    dev = {"step": 10, "single_n": 20, "acc_iou50": .5,
           "decisions": {"overall": {"n": 12, "accuracy": .8, "ece": None},
                         "by_modality": {"text": {"n": 6, "accuracy": .9}, "image": {"n": 6, "accuracy": .7}}}}
    evaluation = "GROUNDING DEV " + json.dumps(dev)
    (tmp_path / "sft.log").write_text("\n".join([
        "grounding step=1 loss=1.00000 elapsed=20.0s ranks=4",
        "grounding step=10 loss=0.75000 elapsed=200.0s ranks=4",
        "grounding step=10 loss=0.76000 elapsed=201.0s ranks=4",
        evaluation, "GROUNDING DEV {", "Resumed joint training at optimizer step 5",
        "grounding step=6 loss=0.81000 elapsed=20.0s ranks=4", evaluation,
    ]) + "\n")
    runs = []

    def init(**options):
        points = []
        runs.append((options, points))
        return SimpleNamespace(log=lambda data, step: points.append((step, data)), finish=lambda **kwargs: None)

    fake = SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs)
    mirror(fake, tmp_path, "sft", tmp_path / "swanlog", False, None, mode="online", project="tracking")
    (options, points), (resumed_options, resumed_points) = runs
    assert options["mode"] == "online" and options["public"] is False
    assert "video_cache_dir" not in options["config"]
    assert [step for step, _ in points] == [1, 10, 10]
    assert points[1][1]["train/progress_pct"] == 50.
    assert points[1][1]["train/examples_per_second"] == pytest.approx(3.2)
    assert points[2][1]["dev/joint/macro_harmonic_accuracy"] == pytest.approx(2 * .8 * .5 / 1.3)
    assert "dev/decision/ece" not in points[2][1]
    assert resumed_options["name"].endswith("-resume-5")
    assert [step for step, _ in resumed_points] == [6, 10]
    assert "train/seconds_per_step_500" not in resumed_points[0][1]


def test_swanlab_follow_waits_for_a_complete_training_line(tmp_path, monkeypatch):
    from scripts.multimodal import monitor_swanlab

    log = tmp_path / "sft.log"
    report = tmp_path / "report.json"
    log.write_bytes(b"grounding step=10 loss=0.7")

    def append(_seconds):
        with log.open("ab") as stream:
            stream.write(b"5 elapsed=200.0s ranks=4\n")
        report.write_text('{"status":"complete"}')

    monkeypatch.setattr(monitor_swanlab.time, "sleep", append)
    assert list(monitor_swanlab.log_lines(log, True, report)) == ["grounding step=10 loss=0.75 elapsed=200.0s ranks=4"]


def test_native_swanlab_reporter_logs_metrics_and_isolates_sdk_failure(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace
    from laya_multimodal._swanlab import SwanLabReporter

    runs = []

    def init(**options):
        points, states = [], []
        runs.append((options, points, states))

        def log(data, step):
            if step == 12:
                raise OSError("simulated transport failure")
            points.append((step, data))

        return SimpleNamespace(log=log, finish=lambda state: states.append(state))

    monkeypatch.setitem(sys.modules, "swanlab", SimpleNamespace(init=init, Settings=lambda **values: values))
    config = dict(starting_step=0, planned_steps=12, world_size=4, global_batch_size=64)
    baseline = dict(single_n=2, acc_iou50=.5)
    output = tmp_path / "training/sft"
    with SwanLabReporter(output, mode="online", project="training") as reporter:
        reporter.start(config, baseline)
        reporter.training_step(1, 1.5, 20., .8, 64, {"head": .0002})
        reporter.evaluation(1, baseline, score=.55, selected=True, final=True)
    options, points, states = runs[0]
    assert options["log_dir"] == str(output / "swanlog")
    assert options["on_online_error"] == "offline" and options["public"] is False
    assert points[0] == (0, {"dev/box/single_n": 2, "dev/box/acc_iou50": .5})
    assert points[1][1]["lr/head"] == .0002 and points[1][1]["train/gradient_norm"] == .8
    assert points[2][1]["final/joint/selection_score"] == .55 and states == ["success"]

    with SwanLabReporter(output, mode="online", project="training") as reporter:
        reporter.start(dict(config, starting_step=10), baseline)
        reporter.training_step(11, .8, 20., .5, 64, {"head": .0001})
        reporter.training_step(12, .7, 40., .4, 64, {"head": .0001})
        reporter.evaluation(12, baseline, score=.55, selected=False, final=True)
    options, points, states = runs[1]
    assert options["group"] == "training" and options["name"].endswith("-resume-10")
    assert [step for step, _ in points] == [11] and states == ["crashed"]
    assert "SwanLab logging failed (OSError)" in caplog.text
