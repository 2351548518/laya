"""Exercise the hook lifecycle on the native decision runtime."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya_multimodal import BaseHook, Router  # noqa: E402
from test_multimodal import Q, tiny_agent  # noqa: E402


class Observer(BaseHook):
    def __init__(self):
        self.events = []

    def on_predict_start(self, ctx):
        self.events.append("start")
        ctx.states[0] = "updated context"

    def on_predict_end(self, ctx):
        self.events.append("end")
        if ctx.error is None:
            assert ctx.usage["output_tokens"] == 0

    def on_error(self, ctx):
        self.events.append("error")


class HooksTests(unittest.TestCase):
    def test_direct_and_routed_calls_observe_the_same_lifecycle(self):
        agent = tiny_agent()
        expected = agent.predict("updated context", Q)["answers"]
        for predictor in (agent, Router(multimodal=agent)):
            observer = Observer()
            result = predictor.predict("original", Q, hooks=[observer])
            self.assertEqual(result["answers"], expected)
            self.assertEqual(observer.events, ["start", "end"])

    def test_invalid_question_dispatches_error_then_end(self):
        agent = tiny_agent()
        observer = Observer()
        with self.assertRaises(ValueError):
            agent.predict("context", {"q": {"type": "unknown"}}, hooks=[observer])
        self.assertEqual(observer.events, ["start", "error", "end"])


if __name__ == "__main__":
    unittest.main()
