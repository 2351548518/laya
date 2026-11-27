"""Exercise each joint training entry point without loading backbone weights."""

from types import SimpleNamespace
from unittest.mock import Mock

from laya import cli, train_cli
from scripts.multimodal import train


def test_checkout_script_uses_the_packaged_entrypoint():
    assert train.main is train_cli.main
    assert train.validate_visual_training is train_cli.validate_visual_training


def test_train_command_dispatches_to_joint_trainer(tmp_path, monkeypatch):
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
