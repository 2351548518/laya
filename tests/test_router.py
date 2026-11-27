"""Native routing, batch order and bounded checkpoint reuse."""

from pathlib import Path
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya import Router  # noqa: E402
from test_multimodal import Q, tiny_agent  # noqa: E402


class RouterTests(unittest.TestCase):
    def test_default_routes_text_and_media_to_the_same_checkpoint(self):
        agent = tiny_agent()
        router = Router(multimodal=agent)
        for state in ("request a refund", "请退款", {"image": "scene.png"}):
            self.assertEqual(router.route(state).model, "multimodal")
        self.assertEqual(router.predict("text", Q)["answers"], agent.predict("text", Q)["answers"])

    def test_batch_keeps_request_order_across_checkpoint_groups(self):
        alpha, beta = tiny_agent(), tiny_agent()
        alpha.model_id, beta.model_id = "alpha", "beta"
        router = Router(multimodal=alpha)
        router.attach("other", beta)
        results = router.predict_batch([
            {"state": "one", "questions": Q},
            {"state": "two", "questions": Q, "model": "other"},
            {"state": "three", "questions": Q},
        ])
        self.assertEqual([result["model"] for result in results], ["alpha", "beta", "alpha"])

    def test_concurrent_loads_share_one_build(self):
        router = Router(multimodal="/checkpoints/native")
        agent = tiny_agent()
        with patch.object(router, "_build", return_value=agent) as build:
            with ThreadPoolExecutor(max_workers=4) as pool:
                loaded = list(pool.map(router.load, ["multimodal"] * 8))
        self.assertEqual(loaded, [agent] * 8)
        build.assert_called_once_with("multimodal")

    def test_repointing_a_native_registration_evicts_the_old_agent(self):
        router = Router(multimodal=tiny_agent())
        router.register("multimodal", "/checkpoints/replacement")
        self.assertEqual(router.route("text")["repo"], "/checkpoints/replacement")
        agent = tiny_agent()
        with patch("laya.agent.load", return_value=agent) as load:
            self.assertIs(router.load("multimodal"), agent)
        self.assertEqual(load.call_args.args, ("/checkpoints/replacement",))


if __name__ == "__main__":
    unittest.main()
