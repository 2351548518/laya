"""Make one decision with a classic Laya checkpoint.

Ask one typed question using a local classic checkpoint or the published English weights.
"""
import argparse
import os
from pathlib import Path

# Set these before importing laya: a stray TensorFlow install can deadlock model loading, and
# without the last one the fast tokenizer forks a thread per core.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import laya  # noqa: E402  -- must come after the environment variables above

local = Path(__file__).resolve().parents[2] / "models" / "laya"
default = str(local) if (local / "rl_agent_config.json").is_file() else "convaiinnovations/laya"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("checkpoint", nargs="?", default=default, help="classic checkpoint directory or model ID")
parser.add_argument("--device", help="inference device, such as cpu or cuda")
args = parser.parse_args()

agent = laya.load(args.checkpoint, device=args.device)

result = agent.predict(
    {"body": "Hi, we were billed twice for March. Please refund the duplicate today."},
    {"refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    }},
)

answer = result["answers"]["refund_requested"]
print("P(refund requested) = %.4f  (confidence %.4f)" % (answer["noul"], answer["confidence"]))
