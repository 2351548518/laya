"""Evaluate local typed decisions and single-box targets without dataset-specific tooling."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from laya.grounding_train import grounding_metrics, load_grounding_rows
from laya.multimodal import MultimodalAgent
from laya.multimodal_train import collect_records, load_rows, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--decisions', type=Path)
    parser.add_argument('--grounding', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device')
    parser.add_argument('--max-len', type=int)
    parser.add_argument('--micro-batch', type=int, default=1)
    parser.add_argument('--video-cache')
    args = parser.parse_args()
    if not args.decisions and not args.grounding:
        parser.error('Provide --decisions, --grounding, or both')
    if args.micro_batch < 1 or (args.max_len is not None and args.max_len < 1):
        parser.error('Batch size and context length must be positive')
    agent = MultimodalAgent(args.model, device=args.device)
    if args.max_len is not None:
        agent.max_len = args.max_len
    if args.video_cache:
        agent._video_cache_dir = args.video_cache
    result = dict(model=args.model, max_len=agent.max_len, output_tokens=0)
    if args.decisions:
        records = collect_records(agent, load_rows(args.decisions), args.micro_batch)
        result['decisions'] = metrics(records, agent.temperature, agent.temperature_by_options,
            agent.modality_calibration, agent.ordinal_calibration)
    if args.grounding:
        result['grounding'] = grounding_metrics(agent, load_grounding_rows(args.grounding), limit=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
