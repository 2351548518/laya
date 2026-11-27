"""Check that Compose overlays point the native service at the mounted checkpoint."""

import json
from pathlib import Path
from shutil import which
import os
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ComposeTests(unittest.TestCase):
    def config(self, *overlays, **variables):
        docker = which("docker")
        if not docker or subprocess.run([docker, "compose", "version"], capture_output=True).returncode:
            self.skipTest("Docker Compose is unavailable")
        args = [docker, "compose", "-f", "compose.yaml", "-f", "compose.http.yaml"]
        for overlay in overlays:
            args.extend(("-f", overlay))
        env = {name: value for name, value in os.environ.items()
               if not name.startswith(("LAYA_", "HF_", "TRANSFORMERS_"))}
        env.update(variables)
        result = subprocess.run([*args, "config", "--format", "json"], cwd=ROOT,
                                text=True, capture_output=True, check=True, env=env)
        return json.loads(result.stdout)["services"]

    def test_runtime_build_includes_the_native_stack(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn('pip install ".[serve,multimodal]"', dockerfile)
        self.assertIn("ffmpeg libsndfile1", dockerfile)
        self.assertIn("docker/entrypoint.py", dockerfile)

    def test_local_mount_disables_online_loading(self):
        services = self.config("compose.multimodal.yaml", "compose.example.yml",
                               LAYA_MULTIMODAL_MODEL_DIR=str(ROOT), LAYA_INPUT_DIR=str(ROOT / "examples/docker"))
        for service in services.values():
            env = service["environment"]
            self.assertEqual(env["LAYA_MULTIMODAL_MODEL"], "/opt/laya/checkpoint")
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
            mount = next(volume for volume in service["volumes"] if volume["target"] == "/opt/laya/checkpoint")
            self.assertEqual(Path(mount["source"]), ROOT)
            self.assertTrue(mount["read_only"])
            self.assertFalse(mount.get("bind", {}).get("create_host_path", False))
        self.assertEqual(services["laya-multimodal"]["environment"]["LAYA_REQUEST_FILE"], "/inputs/request.json")
        self.assertEqual(services["laya-multimodal-serve"]["command"], ["laya-multimodal-serve"])

    def test_gpu_overlays_select_matching_wheels_and_devices(self):
        for overlay, index in (("compose.cuda.yaml", "cu126"), ("compose.spark.yaml", "cu130")):
            with self.subTest(overlay=overlay):
                for service in self.config(overlay, LAYA_GPU_ID="3").values():
                    self.assertEqual(service["build"]["args"]["TORCH_INDEX"], index)
                    self.assertEqual(service["environment"]["LAYA_DEVICE"], "cuda")
                    devices = service["deploy"]["resources"]["reservations"]["devices"]
                    self.assertEqual(devices[0]["device_ids"], ["3"])


if __name__ == "__main__":
    unittest.main()
