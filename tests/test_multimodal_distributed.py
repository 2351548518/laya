"""Real two-process gradient synchronization, uneven tails and optimizer/data-cursor resume."""
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
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
from laya_multimodal.training.distributed import DistributedContext, _atomic_torch_save, distributed_context  # noqa: E402
from laya_multimodal.train import (  # noqa: E402
    GroundingTrainConfig, _evaluate_development, _shared_indexed_rows, finetune_grounding,
)
from laya_multimodal.models.grounding import GroundingHead  # noqa: E402
from laya_multimodal.data import load_rows  # noqa: E402
from test_multimodal import Q, tiny_agent  # noqa: E402


def _save_head(agent, directory):
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().clone() for k, v in agent.model.lm_head.state_dict().items()},
              str(directory / "decision_head.safetensors"))


def _joint_agent():
    agent = tiny_agent()
    agent.model.grounding_head = GroundingHead(8, 16, 1, "single-v1")
    agent.grounding_trained = agent.detection_trained = False
    agent.save = lambda path: _save_head(agent, path)
    return agent


def _train_cpu(directory, output):
    agent = _joint_agent()
    config = GroundingTrainConfig(epochs=2, micro_batch=1, global_batch_size=4,
        eval_every=2, head_learning_rate=.01, train_readout_in_warmup=True, batch_decisions=True,
        distributed_decision_eval=True)
    finetune_grounding(directory / "train.jsonl", directory / "dev.jsonl", output, agent=agent, config=config,
                       decision_dev_data=directory / "decision_dev.jsonl")
    report = json.loads((output / "training_report.json").read_text())
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
        weight, report = _train_cpu(directory, directory / "parallel")
        assert report["world_size"] == world and report["steps"] == 4
        assert report["forward_passes"] == {"geometry": 0, "decisions": 10}
        assert len(report["selection_history"]) == 2 and report["after"]["decisions"]["overall"]["n"] == 5
        torch.save(weight, directory / ("rank%d.pt" % rank))
    finally:
        dist.destroy_process_group()


def _dataset(path):
    rows = [{"id": str(i), "state": "%s context %d" % (path.stem, i),
             "questions": {"category": Q["category"]}, "expected": {"category": "bright" if i % 2 else "dark"}}
            for i in range(5)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_two_process_training_matches_single_process_with_uneven_tail(tmp_path):
    for name in ("train", "dev", "decision_dev"):
        _dataset(tmp_path / (name + ".jsonl"))
    expected, _ = _train_cpu(tmp_path, tmp_path / "single")
    mp.spawn(_distributed_worker, args=(2, str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        actual = torch.load(tmp_path / ("rank%d.pt" % rank), weights_only=True)
        torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)


def _joint_development_worker(rank, world, directory):
    torch.set_num_threads(1)
    path = Path(directory)
    dist.init_process_group("gloo", init_method=(path / "joint-rendezvous").as_uri(), rank=rank, world_size=world)
    try:
        ctx = distributed_context()
        validated = []
        rows = _shared_indexed_rows(path / "train.jsonl", lambda row: validated.append(row["id"]), ctx)
        expected = load_rows(path / "train.jsonl")
        assert list(rows) == expected and rows[-1] == expected[-1]
        assert validated == ([row["id"] for row in expected] if ctx.main else [])

        agent = tiny_agent()
        agent._video_cache_size = 2
        def prepare(request, question):
            value = int(request["text"][-1])
            inputs = {"input_ids": torch.tensor([[1, value + 2]])}
            return inputs, agent._to_internal(question), {}
        agent._prepare = prepare
        decision_rows = [{**row, "state": {"text": row["state"], "video": "clip.mp4"}}
                         if i < 2 else row for i, row in enumerate(expected)]
        cfg = GroundingTrainConfig(distributed_decision_eval=True, retention_tagged_only=True)
        decision_rows[-1]["retention_reference"] = True
        actual = _evaluate_development(agent, [], decision_rows, cfg, ctx)
        if ctx.main:
            original = _evaluate_development(agent, [], decision_rows, cfg, DistributedContext())
            assert actual == original and actual["decisions"]["overall"]["n"] == len(decision_rows)
        else:
            assert actual is None
    finally:
        dist.destroy_process_group()


def test_joint_development_and_startup_share_work_without_changing_results(tmp_path):
    _dataset(tmp_path / "train.jsonl")
    mp.spawn(_joint_development_worker, args=(2, str(tmp_path)), nprocs=2, join=True)


def _joint_restart_worker(rank, world, directory):
    from laya_multimodal.training.checkpoint import resolve_checkpoint, save_checkpoint

    torch.set_num_threads(1)
    path = Path(directory)
    dist.init_process_group("gloo", init_method=(path / "restart-rendezvous").as_uri(), rank=rank, world_size=world)
    ctx = distributed_context()

    class Recorder:
        def __init__(self):
            self.events = []

        def start(self, config, baseline):
            self.events.append(("start", config["starting_step"]))
            assert config["world_size"] == world and config["planned_steps"] == 6

        def training_step(self, step, loss, elapsed, grad_norm, window_size, learning_rates):
            self.events.append(("train", step))
            assert loss >= 0 and elapsed >= 0 and grad_norm >= 0 and window_size <= 2
            assert 0 < learning_rates["head"] <= .01

        def evaluation(self, step, report, *, score, selected, final=False):
            self.events.append(("final" if final else "dev", step))
            assert report["grounding_retention_passed"] and isinstance(selected, bool)

    def train(output, resume_from=None, reporter=None):
        agent = _joint_agent()
        cfg = GroundingTrainConfig(epochs=2, global_batch_size=2, micro_batch=1, eval_every=2,
            learning_rate=.01, head_learning_rate=.01, train_readout_in_warmup=True,
            batch_decisions=True, shuffle_options=True, checkpoint_every=1, replay_fraction=.5)
        finetune_grounding(path / "train.jsonl", path / "dev.jsonl", output, agent=agent, config=cfg,
                           replay_data=path / "replay.jsonl", resume_from=resume_from, reporter=reporter)
        return agent.model.lm_head.weight.detach().clone()

    def interrupted(*args):
        save_checkpoint(*args)
        if args[4]["steps"] == 2:
            raise RuntimeError("simulated interruption after publishing the checkpoint")

    try:
        expected = train(path / "full")
        with patch("laya_multimodal.train.save_checkpoint", side_effect=interrupted):
            with pytest.raises(RuntimeError, match="simulated interruption"):
                train(path / "interrupted")
        saved = resolve_checkpoint(path / "interrupted")
        assert json.loads((saved / "progress.json").read_text())["steps"] == 2
        assert (saved / ("option_rng_rank%d.pt" % rank)).is_file()
        reporter = Recorder()
        log = StringIO()
        with redirect_stdout(log):
            actual = train(path / "continued", resume_from=path / "interrupted", reporter=reporter)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if ctx.main:
            events = [line.split(" ", 2) for line in log.getvalue().splitlines()]
            assert [kind for _, kind, _ in events] == ["RESUME", "TRAIN", "DEV", "DEV"]
            assert all(datetime.fromisoformat(stamp).utcoffset() is not None for stamp, _, _ in events)
            assert events[0][2] == "step=2"
            assert events[1][2].startswith("step=3 loss=") and events[1][2].endswith(" ranks=2")
            assert reporter.events == [("start", 2), ("train", 3), ("dev", 4), ("train", 6), ("dev", 6), ("final", 6)]
            reference = json.loads((path / "full/training_report.json").read_text())
            resumed = json.loads((path / "continued/training_report.json").read_text())
            assert resumed["steps"] == reference["steps"] == 6
            assert resumed["resume"]["starting_step"] == 2
            assert resumed["selection_history"] == reference["selection_history"]
            assert [json.loads(payload) for _, kind, payload in events if kind == "DEV"] == resumed["selection_history"][-2:]
            assert len(list((path / "continued/optimizer-checkpoints").glob("step-*"))) == 2
        else:
            assert not reporter.events and not log.getvalue()
    finally:
        dist.destroy_process_group()


def test_joint_training_restores_optimizer_and_order_across_ranks(tmp_path):
    for name in ("train", "dev", "replay"):
        _dataset(tmp_path / (name + ".jsonl"))
    mp.spawn(_joint_restart_worker, args=(2, str(tmp_path)), nprocs=2, join=True)


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
