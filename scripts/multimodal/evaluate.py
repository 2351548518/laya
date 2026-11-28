"""Evaluate local typed decisions and single-box targets without dataset-specific tooling."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from laya_multimodal.evaluation.evaluator import file_fingerprint
from laya_multimodal.train import grounding_metrics, load_grounding_rows
from laya_multimodal.models.multimodal import MultimodalAgent
from laya_multimodal.data import load_rows, read_records
from laya_multimodal.evaluation.native_metrics import collect_records, metrics


def harmonic_accuracy(decisions, grounding):
    if not decisions or not grounding or not decisions['overall']['n'] or not grounding['single_n']:
        return None
    accuracy, box_accuracy = decisions['overall']['accuracy'], grounding['acc_iou50']
    total = accuracy + box_accuracy
    return 2 * accuracy * box_accuracy / total if total else 0.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--decisions', type=Path)
    parser.add_argument('--grounding', type=Path)
    parser.add_argument('--suite-manifest', type=Path, help='manifest from prepare_dataset.py')
    parser.add_argument('--suite', help='named split in --suite-manifest')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device')
    parser.add_argument('--max-len', type=int)
    parser.add_argument('--micro-batch', type=int, default=1)
    parser.add_argument('--grounding-micro-batch', type=int, default=1)
    parser.add_argument('--max-micro-tokens', type=int, default=4096)
    parser.add_argument('--video-cache')
    parser.add_argument('--video-cache-size', type=int, default=0, help='decoded clips to retain in memory')
    args = parser.parse_args(argv)
    suite_path = None
    if args.suite_manifest or args.suite:
        if not args.suite_manifest or not args.suite or args.decisions or args.grounding:
            parser.error('Use --suite and --suite-manifest together, or pass individual files')
        manifest = json.loads(args.suite_manifest.read_text(encoding='utf-8'))
        entry = manifest.get('splits', {}).get(args.suite)
        if manifest.get('status') != 'complete' or not entry:
            parser.error('Suite is missing from the completed manifest')
        suite_path = args.suite_manifest.parent / entry['file']
        if file_fingerprint(suite_path) != entry['sha256']:
            parser.error('Suite contents differ from the manifest')
    elif not args.decisions and not args.grounding:
        parser.error('Provide --decisions, --grounding, or a named suite')
    if min(args.micro_batch, args.grounding_micro_batch, args.max_micro_tokens) < 1 or (
            args.max_len is not None and args.max_len < 1):
        parser.error('Batch size, token budget and context length must be positive')
    if args.video_cache_size < 0:
        parser.error('Video cache size must be nonnegative')
    output = args.out.resolve()
    for source in (args.decisions, args.grounding, suite_path, args.suite_manifest):
        if source and (output == source.resolve() or
                       (output.is_file() and source.is_file() and output.samefile(source))):
            parser.error('--out must refer to a separate file from the input datasets')
    agent = MultimodalAgent(args.model, device=args.device)
    if args.max_len is not None:
        agent.max_len = args.max_len
    if args.video_cache:
        agent._video_cache_dir = args.video_cache
    agent._video_cache_size = args.video_cache_size
    result = dict(model=args.model, backbone=agent.model_type, revision=agent.revision,
        device=str(agent.device), max_len=agent.max_len, default_video_frames=agent.frames,
        video_audio=agent.use_audio_in_video, micro_batch=args.micro_batch,
        grounding_micro_batch=args.grounding_micro_batch, max_micro_tokens=args.max_micro_tokens,
        video_cache_size=args.video_cache_size, output_tokens=0, datasets={})
    decision_rows = load_rows(args.decisions) if args.decisions else []
    box_rows = load_grounding_rows(args.grounding) if args.grounding else []
    if suite_path:
        for row in read_records(suite_path):
            (box_rows if 'grounding' in row else decision_rows).append(row)
        result['datasets']['suite'] = dict(name=args.suite, path=str(suite_path.resolve()), sha256=entry['sha256'])
    if decision_rows:
        records = collect_records(agent, decision_rows, args.micro_batch,
                                  max_micro_tokens=args.max_micro_tokens)
        result['decisions'] = metrics(records, agent.temperature, agent.temperature_by_options,
                                      agent.modality_calibration, agent.ordinal_calibration, slices=True)
        if args.decisions:
            result['datasets']['decisions'] = dict(path=str(args.decisions.resolve()), sha256=file_fingerprint(args.decisions))
    if box_rows:
        result['grounding'] = grounding_metrics(agent, box_rows, limit=0,
                                                micro_batch=args.grounding_micro_batch, max_micro_tokens=args.max_micro_tokens)
        if args.grounding:
            result['datasets']['grounding'] = dict(path=str(args.grounding.resolve()), sha256=file_fingerprint(args.grounding))
    combined = harmonic_accuracy(result.get('decisions'), result.get('grounding'))
    if combined is not None:
        result['harmonic_accuracy'] = combined
    report = json.dumps(result, ensure_ascii=False, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
