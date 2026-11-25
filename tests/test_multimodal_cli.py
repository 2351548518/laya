"""Portable training tools must work without local collection or experiment code."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from laya.grounding_train import GroundingTrainConfig
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
    for name in ['train', 'dev', 'decision_dev']:
        (tmp_path / (name + '.jsonl')).write_text(json.dumps(dict(state='context')) + '\n')
    with pytest.raises(ValueError, match='processor admission'):
        validate_visual_training(tmp_path, model, 16384)
    validate_visual_training(tmp_path, model, 16384, require_admission=False)
    (tmp_path / 'train.jsonl').write_text(json.dumps(dict(state={'audio': 'voice.wav'})) + '\n')
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
