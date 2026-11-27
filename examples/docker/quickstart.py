"""Run a native decision request with text or mounted media."""
import json
import os
from pathlib import Path

import torch

from laya import Router
from laya.mcp.device import env_device


def main():
    device = env_device() or "cpu"
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. "
            "Check the GPU override, host driver and NVIDIA Container Toolkit."
        )
    request_file = os.environ.get("LAYA_REQUEST_FILE") or Path(__file__).with_name("request.json")
    request = json.loads(Path(request_file).read_text(encoding="utf-8"))
    model = os.environ.get("LAYA_MODEL", "auto")
    result = Router(device=device).predict(
        request["state"], request["questions"], model=None if model == "auto" else model
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
