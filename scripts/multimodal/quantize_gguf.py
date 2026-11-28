"""Create mobile GGUF variants from floating-point exports, preserving decision heads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from laya_multimodal.evaluation.evaluator import file_fingerprint
from scripts.multimodal.export_gguf import inspect_gguf


def floating_source(reader, gguf):
    allowed = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
               gguf.GGMLQuantizationType.BF16}
    if any(tensor.tensor_type not in allowed for tensor in reader.tensors):
        raise ValueError('quantize from a floating-point original, not an already quantized model')


def head_fingerprints(reader):
    return {tensor.name: {'shape': tensor.shape.tolist(),
                         'sha256': hashlib.sha256(tensor.data.tobytes()).hexdigest()}
            for tensor in reader.tensors if tensor.name.startswith('decision.')}


def quantize(source, target, kind, binary, gguf, env, threads, require_heads=True, head_quantization='F32',
             imatrix=None, pure=False, token_embedding_type=None, tensor_types=()):
    original = inspect_gguf(source, gguf, require_heads)
    reader = gguf.GGUFReader(source)
    floating_source(reader, gguf)
    heads = head_fingerprints(reader)
    command = [str(binary)]
    if imatrix is not None:
        command += ['--imatrix', str(imatrix)]
    if pure:
        command += ['--pure']
    if token_embedding_type is not None:
        command += ['--token-embedding-type', token_embedding_type.lower()]
    if require_heads and head_quantization != 'F32':
        command += ['--tensor-type', r'decision\.(readout|box_hidden|box_output)\.weight='
                    + head_quantization.lower()]
    command += ['--tensor-type', r'decision\..*=f32']
    for rule in tensor_types:
        command += ['--tensor-type', rule]
    command += [str(source), str(target), kind, str(threads)]
    subprocess.run(command, env=env, check=True)
    converted = inspect_gguf(target, gguf, require_heads, head_quantization)
    if converted['decision_profile'] != original['decision_profile']:
        raise ValueError('quantization changed the decision metadata')
    after = head_fingerprints(gguf.GGUFReader(target))
    protected = set(heads)
    if head_quantization != 'F32' and require_heads:
        protected -= {'decision.readout.weight', 'decision.box_hidden.weight', 'decision.box_output.weight'}
    if any(after.get(name) != heads[name] for name in protected):
        raise ValueError('quantization changed a protected decision/localization tensor')
    converted.update(quantization=kind, head_quantization=head_quantization,
                     decision_tensors_bitwise_unchanged=after == heads, pure=pure,
                     token_embedding_type=token_embedding_type,
                     tensor_type_overrides=list(tensor_types),
                     importance_matrix_sha256=None if imatrix is None else file_fingerprint(str(imatrix)))
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--mmproj', type=Path, required=True)
    parser.add_argument('--llama-cpp', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--name', default='jev-multimodal-qwen3.5-4b')
    parser.add_argument('--variants', nargs='+', choices=('Q8_0', 'Q4_K_M', 'Q4_0', 'Q3_K_M', 'Q3_K_S', 'Q2_K',
                                                        'Q2_0', 'Q1_0', 'IQ3_XXS', 'IQ2_XXS', 'IQ2_XS', 'IQ1_S'),
                        default=['Q4_K_M', 'Q3_K_M'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--head-quantization', choices=('F32', 'Q8_0', 'Q4_0', 'Q2_0', 'Q1_0'), default='F32',
                        help='Head matrix quantization; normalization weights and biases remain FP32')
    parser.add_argument('--mmproj-quantization', choices=('Q8_0', 'Q4_0', 'Q3_K_M', 'Q2_0', 'Q1_0'), default='Q8_0')
    parser.add_argument('--imatrix', type=Path, help='Locally collected language-model importance matrix')
    parser.add_argument('--pure', action='store_true', help='Disable mixed-precision quantization heuristics')
    parser.add_argument('--token-embedding-type', choices=('Q1_0', 'Q2_0', 'Q4_0', 'Q8_0'),
                        help='Explicit embedding precision, needed for pure IQ with text-only imatrix')
    parser.add_argument('--tensor-type', action='append', default=[],
                        help='Extra backbone tensor regex=type overrides; head protection takes precedence')
    args = parser.parse_args()
    model, projector = args.model.resolve(strict=True), args.mmproj.resolve(strict=True)
    repo, output = args.llama_cpp.resolve(strict=True), args.out.resolve()
    imatrix = args.imatrix.resolve(strict=True) if args.imatrix is not None else None
    if any(kind.startswith('IQ') for kind in args.variants) and imatrix is None:
        raise ValueError('the selected IQ quantizations require a measured importance matrix')
    if args.pure and any(kind.startswith('IQ') for kind in args.variants) and args.token_embedding_type is None:
        raise ValueError('pure IQ needs an explicit token embedding type; text imatrix omits embedding lookup data')
    if output.exists() or '/' in args.name or '\\' in args.name or args.threads < 1:
        raise ValueError('use a new output directory, a basename and a positive thread count')
    binary = repo / 'build-laya/bin/llama-quantize'
    if not binary.is_file():
        raise FileNotFoundError(binary)
    sys.path.insert(0, str(repo / 'gguf-py'))
    import gguf
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + output.name + '-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'mobile'
        stage.mkdir()
        records = {}
        for kind in dict.fromkeys(args.variants):
            suffix = '-pure' if args.pure else ''
            if args.token_embedding_type is not None:
                suffix += '-emb-' + args.token_embedding_type
            if args.head_quantization != 'F32':
                suffix += '-heads-' + args.head_quantization
            target = stage / (args.name + '-' + kind + suffix + '.gguf')
            records[target.name] = quantize(model, target, kind, binary, gguf, env, args.threads,
                                            head_quantization=args.head_quantization, imatrix=imatrix,
                                            pure=args.pure, token_embedding_type=args.token_embedding_type,
                                            tensor_types=args.tensor_type)
        target = stage / ('mmproj-' + args.mmproj_quantization + '.gguf')
        records[target.name] = quantize(projector, target, args.mmproj_quantization, binary, gguf, env,
                                        args.threads, require_heads=False)
        report = {'status': 'structure_verified', 'files': records,
                  'language_source': str(model), 'vision_source': str(projector),
                  'decision_heads_preserved_fp32': args.head_quantization == 'F32',
                  'head_quantization': args.head_quantization,
                  'behavioral_quality': 'Requires a separate inference evaluation',
                  'phone_validation': 'Not performed by quantization'}
        (stage / 'quantization-verification.json').write_text(json.dumps(report, indent=2) + '\n')
        checksums = {}
        for path in sorted(stage.iterdir()):
            checksums[path.name] = file_fingerprint(str(path))
        (stage / 'SHA256SUMS').write_text(''.join(sha + '  ' + name + '\n' for name, sha in checksums.items()))
        stage.rename(output)
    print('MOBILE GGUF STRUCTURE VERIFIED', output, flush=True)


if __name__ == '__main__':
    main()
