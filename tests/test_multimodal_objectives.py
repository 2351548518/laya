"""Proper objectives, optimizer schedules, and native microbatch gradient contracts."""
import json

import pytest
import torch

from laya_multimodal.training.objectives import decision_loss, grow_lora, learning_rate_factor


def test_ranked_probability_loss_penalizes_farther_ordinal_errors_and_has_gradients():
    target = torch.tensor([[1., 0., 0.]])
    mask, qt = torch.ones(1, 3, dtype=torch.bool), torch.tensor([1])
    near = torch.tensor([[0.2, 0.5, 0.3]]).log().requires_grad_()
    far = torch.tensor([[0.2, 0.3, 0.5]]).log().requires_grad_()
    a, _ = decision_loss(near, target, mask, qt, ordinal_weight=1)
    b, _ = decision_loss(far, target, mask, qt, ordinal_weight=1)
    assert b > a
    b.backward()
    assert torch.isfinite(far.grad).all() and far.grad.abs().sum() > 0


def test_reference_kl_has_finite_masked_gradients():
    logits = torch.tensor([[0.5, -0.2, 1.], [0., 0., -10000.]], requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    target = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    reference = target.clone()
    reference[1, 0] = float("nan")
    accepted = torch.tensor([True, False])
    loss, parts = decision_loss(logits, target, mask, torch.tensor([0, 2]),
        reference=reference, reference_mask=accepted, kl_weight=0.01)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and logits.grad[1, 2] == 0
    assert parts["reference_kl"] > 0


def test_lr_warmup_and_cosine_reach_requested_floor():
    assert learning_rate_factor(1, 100, 0.1) == pytest.approx(0.1)
    assert learning_rate_factor(10, 100, 0.1) == 1
    assert learning_rate_factor(100, 100, 0.1) == pytest.approx(0.1)


def test_variable_micro_batches_preserve_supervised_ordinal_and_reference_gradients():
    logits = torch.randn(5, 3, requires_grad=True)
    y = torch.eye(3)[torch.tensor([0, 1, 2, 1, 0])]
    mask, qt = torch.ones_like(logits, dtype=torch.bool), torch.tensor([0, 1, 2, 1, 0])
    reference = y * .8 + .2 / 3
    kwargs = dict(ordinal_weight=.25, brier_weight=.025, kl_weight=.2)
    complete, _ = decision_loss(logits, y, mask, qt, reference=reference, **kwargs)
    expected = torch.autograd.grad(complete, logits)[0]
    pieces = []
    for start, end in ((0, 1), (1, 4), (4, 5)):
        loss, parts = decision_loss(logits[start:end], y[start:end], mask[start:end], qt[start:end],
                                    reference=reference[start:end], collect_components=False, **kwargs)
        assert parts == {}
        pieces.append(loss * (end - start) / 5)
    actual = torch.autograd.grad(sum(pieces), logits)[0]
    torch.testing.assert_close(expected, actual)


def test_growing_lora_preserves_predictions_and_saved_shapes(tmp_path):
    peft = pytest.importorskip("peft")
    base = torch.nn.Sequential(torch.nn.Linear(8, 6, bias=False))
    model = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4, target_modules=["0"]))
    with torch.no_grad():
        model.base_model.model[0].lora_B["default"].weight.normal_()
    x = torch.randn(4, 8)
    before = model(x).detach()
    grow_lora(model, 4, 8)
    torch.testing.assert_close(model(x), before, atol=1e-6, rtol=1e-6)
    model.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "adapter_config.json").read_text())["r"] == 4


def test_media_hash_cache_revalidates_same_size_file_changes(tmp_path):
    import os
    from laya_multimodal.data import _media_hash
    path=tmp_path/'media';path.write_bytes(b'first')
    before=path.stat();one=_media_hash(str(path),before.st_mtime_ns,before.st_size)
    path.write_bytes(b'other');os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns+1000000))
    after=path.stat();two=_media_hash(str(path),after.st_mtime_ns,after.st_size)
    assert before.st_size==after.st_size and one!=two
