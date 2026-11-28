"""Exercise v6's readouts and losses on CPU, with optional checks of the sample index."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from laya_multimodal.decisions.questions import QTYPES
from laya_multimodal.models.grounding import GroundingHead, grounding_loss, validate_boxes
from laya_multimodal.models.multimodal import MultimodalAgent
from laya_multimodal.data import normalize_record, record_modality
from laya_multimodal.training.objectives import decision_loss
from laya_multimodal.models.readout import CandidateScorer
from laya_multimodal.train import decision_target


def check_examples(directory):
    index = json.loads(Path(__file__).with_name("examples-index.json").read_text())
    manifest = json.loads((directory.parent / "manifest.json").read_text())
    assert index["train_sha256"] == manifest["native_splits"]["train"]["sha256"]
    counts = {}
    with (directory / "train.jsonl").open("rb") as stream:
        for section, expected in (("io", 30), ("tasks", 192)):
            entries = index[section]
            assert len(entries) == expected
            for entry in entries:
                stream.seek(entry["byte_offset"])
                raw = json.loads(stream.readline())
                assert raw["id"] == entry["id"]
                row = normalize_record(raw, directory)
                assert record_modality(row) == entry["modality"]
                qid = entry["question_id"]
                question = row["questions"][qid]
                assert question["type"] == entry["type"]
                if entry["type"] == "box":
                    assert row["grounding"]["task"] == "single"
                    assert len(validate_boxes(row["grounding"]["boxes"])) == 1
                else:
                    target = decision_target(row, qid, MultimodalAgent._to_internal(question))
                    assert math.isclose(sum(target), 1.0, abs_tol=1e-6)
            counts[section] = len(entries)
    return counts


def run_heads():
    torch.manual_seed(105)
    hidden = torch.randn(3, 8, 2560, requires_grad=True)
    positions = torch.tensor([[1, 3, 5], [2, 4, -1], [1, 4, 6]])
    mask = positions >= 0
    scorer = CandidateScorer(2560, 256)
    logits = scorer(hidden, positions)
    targets = torch.tensor([[0., 1., 0.], [.25, .75, 0.], [0., .6, .4]])
    types = torch.tensor([QTYPES[kind] for kind in ("choice", "noul", "score")])
    decision, decision_parts = decision_loss(logits, targets, mask, types, ordinal_weight=.1, brier_weight=.5)
    decision.backward()
    assert logits.shape == (3, 3) and float(logits[1, 2].detach()) == -1e4
    assert torch.isfinite(scorer.query.weight.grad).all()

    head = GroundingHead(hidden_size=2560, dimension=256, slots=1)
    box_hidden = torch.randn(1, 8, 2560, requires_grad=True)
    predicted = head(box_hidden, attention_mask=torch.ones(1, 8, dtype=torch.long))
    truth = [[[.1, .2, .6, .8]]]
    box, box_parts = grounding_loss(predicted, truth)
    box.backward()
    assert predicted["boxes"].shape == (1, 1, 4)
    assert predicted["quality_logits"].shape == (1, 1)
    assert torch.isfinite(head.single[-1].weight.grad).all()
    return {"logits_shape": list(logits.shape), "box_shape": list(predicted["boxes"].shape),
            "decision_loss_parts": decision_parts, "grounding_loss_parts": box_parts}


def check_calculations():
    mask = torch.ones(1, 3, dtype=torch.bool)
    choice = torch.log(torch.tensor([[.1, .7, .2]]))
    loss, _ = decision_loss(choice, torch.tensor([[0., 1., 0.]]), mask,
                            torch.tensor([QTYPES["choice"]]), ordinal_weight=.1, brier_weight=.5)
    assert math.isclose(float(loss), -math.log(.7) + .5*.14, abs_tol=1e-6)

    score = torch.log(torch.tensor([[.1, .5, .4]]))
    _, score_parts = decision_loss(score, torch.tensor([[0., .6, .4]]), mask,
                                   torch.tensor([QTYPES["score"]]), ordinal_weight=.1, brier_weight=.5)
    assert math.isclose(score_parts["ce"], -.6*math.log(.5) - .4*math.log(.4), abs_tol=1e-6)
    assert math.isclose(score_parts["ordinal_rps"], .005, abs_tol=1e-6)
    assert math.isclose(score_parts["brier"], .02, abs_tol=1e-6)

    box, _ = grounding_loss({"boxes": torch.tensor([[[.2, .2, .7, .8]]]),
                             "quality_logits": torch.zeros(1, 1)}, [[[.1, .2, .6, .8]]])
    assert math.isclose(float(box), 5*.2 + 2*(1-2/3) + .5*math.log(2), abs_tol=1e-6)
    return {"choice_warmup_loss": float(loss), "score_parts": score_parts, "single_box_loss": float(box)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Path to the local v6 native/ directory")
    args = parser.parse_args()
    result = {"scope": "CPU examples with synthetic hidden states", "heads": run_heads(),
              "worked_calculations": check_calculations()}
    if args.data is not None:
        result["verified_v6_examples"] = check_examples(args.data.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
