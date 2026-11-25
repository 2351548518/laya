import base64
import copy
import json
import unittest

from laya_client import LayaAPIError, LayaClient, LayaResponseError, LayaValidationError

from _support import PREDICTION, QUESTIONS, Server


BOX_QUESTIONS = {"target": {"type": "box", "instructions": "the red cup"}}
BOX_RESULT = {
    "model": "native-qwen",
    "answers": {"target": {"type": "box", "box": {"left": .1, "top": .2, "right": .6, "bottom": .9},
                            "confidence": .82}},
    "usage": {"input_tokens": 96, "output_tokens": 0, "forward_passes": 1, "truncated": False},
    "routing": {"model": "multimodal", "repo": None, "reason": "explicit model", "detection": None, "workflow": None},
}


class MultimodalTests(unittest.TestCase):
    def test_systemone_image_assets_send_the_header_and_allow_null_box_confidence(self):
        result = copy.deepcopy(BOX_RESULT)
        result.pop("routing")
        result["usage"] = {"input_tokens": 96, "output_tokens": 0}
        result["answers"]["target"]["confidence"] = None
        assets = {"photo": {"type": "image", "format": "png", "data_base64": "aW1hZ2U="}}
        with Server(result) as server:
            client = LayaClient(base_url=server.url)
            self.assertEqual(client.predict_image_assets("photo", BOX_QUESTIONS, assets), result)
            method, path, headers, wire = server.requests[0]
            self.assertEqual((method, path), ("POST", "/v1/systemone"))
            self.assertEqual({key.lower(): val for key, val in headers.items()}["typesafe-danger-zone-images"], "true")
            self.assertEqual(json.loads(wire), {"state": "photo", "model": "multimodal",
                                                "questions": BOX_QUESTIONS, "assets": assets})

    def test_sends_readable_box_and_exact_media_bytes(self):
        with Server(BOX_RESULT) as server:
            client = LayaClient(base_url=server.url + "/api", api_key="test-key")
            image = bytes(range(256)) * 140
            reply = client.predict_multimodal(BOX_QUESTIONS, image=image, state="Inspect the tabletop", max_len=512)
            self.assertEqual(reply, BOX_RESULT)
            method, path, headers, wire = server.requests[0]
            self.assertEqual((method, path), ("POST", "/api/v1/multimodal"))
            self.assertEqual(headers["Authorization"], "Bearer test-key")
            body = json.loads(wire)
            self.assertEqual(base64.b64decode(body.pop("image")["base64"], validate=True), image)
            self.assertEqual(body, {"questions": BOX_QUESTIONS, "state": "Inspect the tabletop", "max_len": 512})

    def test_audio_and_video_return_normal_typed_answers(self):
        expected = copy.deepcopy(PREDICTION)
        expected["usage"].update(forward_passes=3, truncated=False)
        with Server(expected) as server:
            client = LayaClient(base_url=server.url)
            for kind in ("audio", "video"):
                self.assertEqual(client.predict_multimodal(QUESTIONS, **{kind: b"media"}), expected)
                body = json.loads(server.requests[-1][-1])
                self.assertEqual(body, {"questions": QUESTIONS, kind: {"base64": "bWVkaWE="}})

    def test_rejects_invalid_inputs_before_transport(self):
        with Server(BOX_RESULT) as server:
            client = LayaClient(base_url=server.url)
            for media in ({}, {"image": b""}, {"image": "scene.jpg"}, {"image": b"ab", "video": b"cd"}):
                with self.subTest(media=media), self.assertRaises(LayaValidationError):
                    client.predict_multimodal(BOX_QUESTIONS, **media)
            with self.assertRaises(LayaValidationError):
                client.predict_multimodal({"target": {**BOX_QUESTIONS["target"], "criteria": []}}, image=b"a")
            with self.assertRaises(LayaValidationError):
                client.predict("state", BOX_QUESTIONS)
            self.assertEqual(server.requests, [])

    def test_rejects_bad_box_and_surfaces_backend_errors(self):
        malformed = copy.deepcopy(BOX_RESULT)
        malformed["answers"]["target"]["box"]["right"] = .05
        with Server(malformed) as server, self.assertRaisesRegex(LayaResponseError, "box order"):
            LayaClient(base_url=server.url).predict_multimodal(BOX_QUESTIONS, image=b"media")
        with Server({"detail": "configure LAYA_MULTIMODAL_MODEL"}, status=503) as server:
            with self.assertRaises(LayaAPIError) as error:
                LayaClient(base_url=server.url).predict_multimodal(BOX_QUESTIONS, image=b"media")
            self.assertEqual(error.exception.status_code, 503)
