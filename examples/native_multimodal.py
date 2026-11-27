"""Ask typed questions about local text and media using a trained checkpoint."""

import argparse
import json
from pathlib import Path

from laya_multimodal import MultimodalAgent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="local native checkpoint directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image", help="image containing a red cup")
    parser.add_argument("--video", help="short clip for a visual decision")
    parser.add_argument("--audio", help="short recording for an audio-capable checkpoint")
    args = parser.parse_args(argv)
    if not (args.checkpoint / "multimodal_config.json").is_file():
        parser.error("checkpoint must be a directory containing multimodal_config.json")

    agent = MultimodalAgent(args.checkpoint, device=args.device)
    text = agent.predict("I was charged twice for the same invoice.", {
        "route": {"type": "choice", "instructions": "Which team should handle the request?",
                  "criteria": {"billing": "invoices and refunds", "technical": "software issues"}},
    })
    print(json.dumps(text))

    if args.image:
        image = agent.predict(image=args.image, questions={
            "target": {"type": "box", "instructions": "the red cup"},
        })
        print(json.dumps(image))
    if args.video:
        video = agent.predict({"video": args.video, "video_audio": False}, {
            "motion": {"type": "score", "instructions": "How much visible motion is there?",
                       "criteria": ["little or none", "some motion", "continuous motion"]},
        })
        print(json.dumps(video))
    if args.audio:
        audio = agent.predict(audio=args.audio, questions={
            "speech": {"type": "noul", "instructions": "Is anyone speaking?"},
        })
        print(json.dumps(audio))


if __name__ == "__main__":
    main()
