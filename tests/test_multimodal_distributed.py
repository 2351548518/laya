"""Real two-process gradient synchronization, uneven tails and optimizer/data-cursor resume."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("PIL")
pytest.importorskip("scipy")
pytestmark = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="needs PyTorch Gloo")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from laya.multimodal_distributed import _atomic_torch_save, distributed_context  # noqa: E402
from laya.multimodal_train import MultimodalTrainConfig, finetune_multimodal, load_rows  # noqa: E402
from test_multimodal import Q, tiny_agent  # noqa: E402


def _save_head(agent, directory, mode="head"):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().clone() for k, v in agent.model.lm_head.state_dict().items()},
              str(directory / "decision_head.safetensors"))


def _train_cpu(train, output, *, held_out=None, **settings):
    agent = tiny_agent()
    agent.save = lambda path, mode="head": _save_head(agent, path, mode)
    config = MultimodalTrainConfig(mode="head", epochs=2, micro_batch=1, global_batch_size=4,
        grad_accum=4, learning_rate=.01, gradient_checkpointing=False, log_every=1, **settings)
    report = finetune_multimodal(train, output, agent=agent, config=config, **(held_out or {}))
    return agent.model.lm_head.weight.detach().clone(), report


def _distributed_worker(rank, world, directory):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=(directory / "rendezvous").as_uri(), rank=rank, world_size=world)
    try:
        context = distributed_context()
        # One conditional parameter per rank and a globally unused parameter.
        parameters = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
        parameters[rank].grad = torch.full((2,), float(rank + 1))
        context.sum_gradients(iter(parameters))
        torch.testing.assert_close(parameters[0].grad, torch.ones(2))
        torch.testing.assert_close(parameters[1].grad, torch.full((2,), 2.))
        assert parameters[2].grad is None
        # A noncontiguous BF16 matrix larger than one FP32 reduction bucket.
        large = torch.nn.Parameter(torch.zeros((4097, 2048), dtype=torch.bfloat16).T)
        large.grad = torch.full_like(large, float(rank + 1))
        context.sum_gradients([large])
        assert torch.all(large.grad == 3)
        del large
        held_out = {name + "_data": directory / (name + ".jsonl") for name in ("validation", "test", "selection")}
        with patch("laya.multimodal_train.load_rows", wraps=load_rows) as loaded:
            weight, report = _train_cpu(directory / "train.jsonl", directory / "parallel", held_out=held_out)
        requested = [call.args[0] for call in loaded.call_args_list]
        assert requested == [directory / "train.jsonl"] + (list(held_out.values()) if context.main else [])
        assert report["world_size"] == world and report["trained_decisions"] == 10
        assert report["validation_rows"] == report["test_rows"] == report["selection_rows"] == 5
        assert len(report["selection_history"]) == 2 and report["after"]["overall"]["n"] == 5
        torch.save(weight, directory / ("rank%d.pt" % rank))
    finally:
        dist.destroy_process_group()


def _dataset(path):
    rows = [{"id": str(i), "state": "%s context %d" % (path.stem, i),
             "questions": {"category": Q["category"]}, "expected": {"category": "bright" if i % 2 else "dark"}}
            for i in range(5)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_two_process_training_matches_single_process_with_uneven_tail(tmp_path):
    _dataset(tmp_path / "train.jsonl")
    held_out = {name + "_data": tmp_path / (name + ".jsonl") for name in ("validation", "test", "selection")}
    for path in held_out.values():
        _dataset(path)
    expected, _ = _train_cpu(tmp_path / "train.jsonl", tmp_path / "single", held_out=held_out)
    mp.spawn(_distributed_worker, args=(2, str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        actual = torch.load(tmp_path / ("rank%d.pt" % rank), weights_only=True)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)


def test_resume_restores_adam_state_and_replays_the_data_cursor(tmp_path):
    _dataset(tmp_path / "train.jsonl")
    expected, _ = _train_cpu(tmp_path / "train.jsonl", tmp_path / "complete")
    _, partial = _train_cpu(tmp_path / "train.jsonl", tmp_path / "partial", max_steps=2)
    assert partial["trained_decisions"] == 5
    actual, resumed = _train_cpu(tmp_path / "train.jsonl", tmp_path / "continued",
        resume_from=str(tmp_path / "partial/checkpoint_latest"))
    assert resumed["resume"]["optimizer_restored"] and resumed["resume"]["starting_decisions"] == 5
    assert resumed["trained_decisions"] == 10 and resumed["optimizer_steps"] == 4
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_cosine_schedule_horizon_survives_an_early_stop(tmp_path):
    _dataset(tmp_path / "train.jsonl")
    expected, _ = _train_cpu(tmp_path / "train.jsonl", tmp_path / "complete", lr_schedule="cosine")
    _train_cpu(tmp_path / "train.jsonl", tmp_path / "partial", max_steps=2, lr_schedule="cosine")
    checkpoint = str(tmp_path / "partial/checkpoint_latest")
    actual, _ = _train_cpu(tmp_path / "train.jsonl", tmp_path / "continued", resume_from=checkpoint, lr_schedule="cosine")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="lr_schedule"):
        _train_cpu(tmp_path / "train.jsonl", tmp_path / "changed", resume_from=checkpoint)


def test_optimizer_restore_rejects_missing_rank_rng_before_mutating_optimizer(tmp_path):
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    parameter.sum().backward()
    optimizer.step()
    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    fresh = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    with pytest.raises(ValueError, match="RNG state"):
        distributed_context().restore_training_state(tmp_path, fresh)
    assert not fresh.state


def test_failed_checkpoint_write_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "optimizer.pt"
    torch.save({"step": 1}, path)

    def fail_during_save(value, stream):
        stream.write(b"partial checkpoint")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_during_save)
    with pytest.raises(OSError, match="disk full"):
        _atomic_torch_save({"step": 2}, path)
    assert torch.load(path, weights_only=True) == {"step": 1}
    assert list(tmp_path.iterdir()) == [path]
