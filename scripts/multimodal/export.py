"""Merge a native Laya LoRA checkpoint into a self-contained safetensors checkpoint."""
import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from laya.evals import file_fingerprint
from laya.multimodal import MultimodalAgent, _is_adapter_model


def fingerprints(root):
    return {str(path.relative_to(root)): (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
            for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts}


def compare_answers(before, after, probability_tolerance=.03, box_tolerance=.02):
    """Check typed answers without treating quantization drift as a new accuracy measurement."""
    if before.keys() != after.keys():
        raise ValueError('answer question IDs changed during export')
    measurements = {}
    for qid, left in before.items():
        right = after[qid]
        if left['type'] != right['type']:
            raise ValueError('answer type changed during export: ' + qid)
        values = {}
        if left['type'] == 'box':
            delta = max(abs(left['box'][key] - right['box'][key]) for key in ('left', 'top', 'right', 'bottom'))
            values['max_coordinate_delta'] = delta
            if delta > box_tolerance:
                raise ValueError('box coordinates changed beyond tolerance: ' + qid)
            if 'confidence' in left and 'confidence' in right:
                values['quality_delta'] = abs(left['confidence'] - right['confidence'])
                if values['quality_delta'] > probability_tolerance:
                    raise ValueError('box quality changed beyond tolerance: ' + qid)
        elif left['type'] == 'noul':
            values['max_probability_delta'] = abs(left['noul'] - right['noul'])
        else:
            if left['probabilities'].keys() != right['probabilities'].keys():
                raise ValueError('answer options changed during export: ' + qid)
            values['max_probability_delta'] = max(abs(p - right['probabilities'][key])
                                                   for key, p in left['probabilities'].items())
            if left['type'] == 'choice':
                values['same_choice'] = left['choice'] == right['choice']
                if not values['same_choice']:
                    raise ValueError('selected choice changed during export: ' + qid)
            else:
                values['score_delta'] = abs(left['score'] - right['score'])
        if values.get('max_probability_delta', 0) > probability_tolerance:
            raise ValueError('probabilities changed beyond tolerance: ' + qid)
        measurements[qid] = values
    return measurements


def predict_fixtures(agent, fixtures):
    return [agent.predict(case['state'], case['questions']) for case in fixtures]


def main():
    try:
        from transformers.initialization import no_init_weights
    except ImportError:
        from transformers.modeling_utils import no_init_weights
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base-id', required=True, help='Portable upstream model ID recorded in the full checkpoint')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--fixtures', type=Path, help='Local JSON list of synthetic state/question cases for reload parity')
    args = parser.parse_args()
    source = args.checkpoint.resolve(strict=True)
    destination = args.out.resolve()
    if destination == source or source in destination.parents or destination.exists():
        raise ValueError('output must be a new directory outside the source checkpoint')
    saved = json.loads((source / 'multimodal_config.json').read_text())
    if saved['mode'] != 'lora':
        raise ValueError('expected a native LoRA checkpoint')
    fixtures = json.loads(args.fixtures.read_text()) if args.fixtures else []
    original = fingerprints(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.device.startswith('cuda') or not fixtures else torch.float32
    print('LOAD SOURCE', source, 'device', args.device, 'dtype', dtype, flush=True)
    agent = MultimodalAgent(str(source), device=args.device, dtype=dtype)
    before = predict_fixtures(agent, fixtures)
    print('BASELINE FIXTURES', len(before), flush=True)
    agent.model = agent.model.merge_and_unload(safe_merge=True)
    if _is_adapter_model(agent.model) or any('lora_' in name for name, _ in agent.model.named_parameters()):
        raise RuntimeError('LoRA parameters remain after merge')
    agent.model_id = agent.model_id_or_path = args.base_id
    agent._checkpoint_mode = 'full'
    agent.model.to('cpu').eval().requires_grad_(False)
    print('MERGED; SAVE FULL CHECKPOINT', flush=True)
    with tempfile.TemporaryDirectory(prefix='.' + destination.name + '-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'checkpoint'
        agent.save(staged, mode='full')
        del agent
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
        print('SAVED; RELOAD WITHOUT ORIGINAL BASE', flush=True)
        with no_init_weights():
            restored = MultimodalAgent(str(staged), device=args.device, dtype=dtype)
        after = predict_fixtures(restored, fixtures)
        results = [{'case': case['id'], 'deltas': compare_answers(left['answers'], right['answers']),
                    'before': left['answers'], 'after': right['answers'],
                    'input_tokens_before': left['usage']['input_tokens'],
                    'input_tokens_after': right['usage']['input_tokens']}
                   for case, left, right in zip(fixtures, before, after)]
        for item in results:
            if item['input_tokens_before'] != item['input_tokens_after']:
                raise ValueError('processor token count changed during export: ' + item['case'])
        if fingerprints(source) != original:
            raise RuntimeError('source checkpoint changed during export')
        report = {'status': 'passed' if fixtures else 'weights_only', 'source_mode': 'lora',
                  'export_mode': 'full', 'base_model': args.base_id, 'peft_required_at_inference': False,
                  'original_backbone_required_at_inference': False, 'source_checkpoint_unchanged': True,
                  'device': args.device, 'dtype': str(dtype), 'probability_tolerance': .03,
                  'box_coordinate_tolerance': .02, 'fixtures': results}
        (staged / 'merge-verification.json').write_text(json.dumps(report, indent=2) + '\n')
        checksums = {}
        for path in sorted(staged.rglob('*')):
            if path.is_file():
                checksums[str(path.relative_to(staged))] = file_fingerprint(str(path))
        (staged / 'SHA256SUMS').write_text(''.join(value + '  ' + name + '\n' for name, value in checksums.items()))
        staged.rename(destination)
        print('EXPORT VERIFIED', destination, 'files', len(checksums), flush=True)


if __name__ == '__main__':
    main()
