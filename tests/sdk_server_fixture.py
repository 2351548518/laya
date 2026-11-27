"""Serve a tiny offline checkpoint for SDK integration tests, never for production."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import uvicorn  # noqa: E402
from laya import Router  # noqa: E402
from laya.serve import _predict_with_tournament, create_app  # noqa: E402
from test_multimodal import TinyProcessor, tiny_agent  # noqa: E402


class FixtureProcessor(TinyProcessor):
    """Give distinct prompts distinct token lengths so the batch test can detect reordering."""

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"][-1]["text"]

    def __call__(self, **kwargs):
        length = 1 + len(kwargs["text"]) // 8
        return {"input_ids": torch.full((1, length), 2),
                "attention_mask": torch.ones((1, length), dtype=torch.long)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--tournament", action="store_true")
    parser.add_argument("--batch", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(1)
    agent = tiny_agent()
    agent.processor = FixtureProcessor()
    questions = {
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": {"meaning": "money"}, "technical": None}},
        "urgency": {"type": "score", "instructions": "Urgency?", "criteria": ["low", "medium", "high"]},
        "refund": {"type": "noul", "instructions": "Refund?", "criteria": {"true": "yes", "false": "no"}},
        "single": {"type": "choice", "instructions": "Pick", "criteria": ["only"]},
    }
    state = {"text": "hello", "unicode": "नमस्ते", "metadata": {"count": 0, "flag": False}}
    expected = agent.predict(state, questions)
    router = Router(multimodal=agent)
    batch_cases = []
    if args.batch:
        states = ["hello " * 9, "hello", {"text": "hello " * 4}, ["hello " * 11, "hello"]]
        for name, inputs, controls in (
            ("default", states, {}),
            ("singleton", states[:1], {}),
            ("serial", states, {"batch_size": 1, "sort_by_length": True}),
            ("sorted", states, {"batch_size": 2, "sort_by_length": True}),
            ("controls", states, {"batch_size": 2, "sort_by_length": True, "task": "multimodal",
                                  "lang": "en", "lang_guess": "fr", "max_len": 128,
                                  "min_confidence": {"default": 0.0}}),
        ):
            item_controls = {key: value for key, value in controls.items()
                             if key not in ("batch_size", "sort_by_length", "min_confidence")}
            call_controls = {key: value for key, value in controls.items()
                             if key in ("batch_size", "sort_by_length", "min_confidence")}
            results = router.predict_batch(
                [dict(state=s, questions=questions, model="multimodal", **item_controls) for s in inputs],
                **call_controls,
            )
            batch_cases.append({"name": name, "states": inputs, "controls": controls, "expected": results})
    os.environ["LAYA_API_KEY"] = "integration-test"
    # This fixture exercises the default HTTP contract, including its confidence extension.
    os.environ["LAYA_JEV_STRICT"] = "1" if args.strict else "0"
    os.environ["LAYA_TOURNAMENT_GROUP"] = "2" if args.tournament else "0"
    tournament = None
    if args.batch and args.tournament:
        large = {"large": {"type": "choice", "instructions": "Pick one",
                           "criteria": {"option%d" % i: "Option %d" % i for i in range(5)}}}
        tournament = {"questions": large, "expected": [
            _predict_with_tournament(router, s, large, 2, model="multimodal") for s in states
        ]}
    app = create_app(router)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        print("LAYA_TEST_SERVER=" + json.dumps({
            "baseURL": "http://127.0.0.1:%d" % listener.getsockname()[1],
            "questions": questions, "state": state, "expected": expected,
            "batch_cases": batch_cases, "tournament": tournament,
        }), flush=True)
        uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[listener])


if __name__ == "__main__":
    main()
