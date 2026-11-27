"""Check revision and digest forwarding through native checkpoint loads."""

from hashlib import sha256
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from laya_multimodal import Router  # noqa: E402
from laya_multimodal.multimodal import _checkpoint_directory  # noqa: E402
from laya_multimodal.revisions import DEFAULT_CHECKPOINT, PINNED_REVISIONS, resolve_revision  # noqa: E402


class CheckpointTests(unittest.TestCase):
    def test_reviewed_pin_resolves_for_the_native_release(self):
        self.assertEqual(resolve_revision(DEFAULT_CHECKPOINT.lower(), "reviewed"),
                         PINNED_REVISIONS[DEFAULT_CHECKPOINT])

    def test_registered_checkpoint_uses_its_own_revision_and_digests(self):
        digest = {"model.safetensors": "a" * 64}
        router = Router(models={"speech": "example/speech"}, revision="general",
                        revisions={"speech": "specific"}, sha256_digests={"speech": digest})
        with patch("laya_multimodal.agent.load") as load:
            router.load("speech")
        self.assertEqual(load.call_args.args, ("example/speech",))
        self.assertEqual(load.call_args.kwargs["revision"], "specific")
        self.assertEqual(load.call_args.kwargs["expected_sha256"], digest)

    def test_local_digest_is_checked_before_loading_a_model(self):
        with tempfile.TemporaryDirectory() as root:
            artifact = Path(root) / "decision_head.safetensors"
            artifact.write_bytes(b"fixture")
            pin = {artifact.name: sha256(b"fixture").hexdigest()}
            self.assertEqual(_checkpoint_directory(root, None, None, pin), Path(root))
            artifact.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                _checkpoint_directory(root, None, None, pin)

    def test_manifest_resolves_the_entire_checkpoint_at_one_commit(self):
        snapshot = Path("/models/repo/snapshots") / ("a" * 40)
        with patch("huggingface_hub.hf_hub_download", return_value=str(snapshot / "multimodal_config.json")), \
                patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as fetch:
            self.assertEqual(_checkpoint_directory(DEFAULT_CHECKPOINT, "branch", None, None), snapshot)
        self.assertEqual(fetch.call_args.kwargs["revision"], snapshot.name)
        self.assertIn("processor/*", fetch.call_args.kwargs["allow_patterns"])


if __name__ == "__main__":
    unittest.main()
