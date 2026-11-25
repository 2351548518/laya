"""Check the typed prediction contract of a local native multimodal checkpoint."""

import argparse
import json
import math
from pathlib import Path

from laya import MultimodalAgent


def validate_prediction(result, modality, questions):
    usage = result["usage"]
    if (result["modality"] != modality or result["answers"].keys() != questions.keys()
            or usage["forward_passes"] != len(questions) or usage["output_tokens"] != 0
            or usage["input_tokens"] <= 0 or usage["truncated"]):
        raise ValueError(f"{modality} prediction has invalid answers or token accounting")

    for name, question in questions.items():
        answer = result["answers"][name]
        kind = question["type"]
        if answer["type"] != kind or not 0 <= answer["confidence"] <= 1:
            raise ValueError(f"{name} has an invalid answer type or confidence")
        if kind == "box":
            box = answer["box"]
            if not (0 <= box["left"] < box["right"] <= 1 and 0 <= box["top"] < box["bottom"] <= 1):
                raise ValueError(f"{name} has invalid normalized box coordinates")
        elif kind == "noul":
            if not 0 <= answer["noul"] <= 1:
                raise ValueError(f"{name} has an invalid yes probability")
        else:
            options = question["criteria"]
            labels = set(options) if kind == "choice" else {str(i) for i in range(len(options))}
            probabilities = answer["probabilities"]
            # Decoded probabilities round to four decimal places.
            if (probabilities.keys() != labels or not all(0 <= p <= 1 for p in probabilities.values())
                    or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.000051 * len(labels))):
                raise ValueError(f"{name} has an invalid option distribution")
            if kind == "choice" and answer["choice"] not in labels:
                raise ValueError(f"{name} returned a choice outside its criteria")
            if kind == "score" and not 0 <= answer["score"] <= len(labels) - 1:
                raise ValueError(f"{name} returned a score outside its levels")
    return usage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="trained native checkpoint directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video", help="short local video for a visual-only check")
    parser.add_argument("--audio", help="short local recording for an audio-capable checkpoint")
    args = parser.parse_args(argv)
    if not (args.checkpoint / "multimodal_config.json").is_file():
        parser.error("checkpoint must be a directory containing multimodal_config.json")

    from PIL import Image, ImageDraw

    agent = MultimodalAgent(args.checkpoint, device=args.device)
    image = Image.new("RGB", (112, 112), "white")
    ImageDraw.Draw(image).rectangle((24, 24, 88, 88), fill="red")
    requests = {
        "text": ("The customer asked for a refund after being charged twice.", {
            "route": {"type": "choice", "instructions": "Where should the request go?",
                      "criteria": {"billing": "payments and refunds", "technical": "software failures"}},
            "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
            "urgency": {"type": "score", "instructions": "How urgent is the request?",
                        "criteria": ["routine", "time-sensitive", "critical"]},
        }),
        "image": ({"image": image}, {
            "red": {"type": "noul", "instructions": "Is the square red?"},
        }),
    }
    if agent.grounding_trained:
        requests["image"][1]["target"] = {"type": "box", "instructions": "the red square"}
    if args.video:
        requests["video"] = ({"video": args.video, "video_audio": False}, {
            "motion": {"type": "noul", "instructions": "Is there visible motion?"},
        })
    if args.audio:
        requests["audio"] = ({"audio": args.audio}, {
            "speech": {"type": "noul", "instructions": "Is anyone speaking?"},
        })

    checks = {}
    for modality, (state, questions) in requests.items():
        result = agent.predict(state, questions)
        checks[modality] = {"types": [q["type"] for q in questions.values()],
                            "usage": validate_prediction(result, modality, questions)}
    print(json.dumps({"status": "passed", "device": str(agent.device), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
