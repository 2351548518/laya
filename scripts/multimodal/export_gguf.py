"""Export a merged native checkpoint with a compatible local llama.cpp-jev converter."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from laya.evals import file_fingerprint


def inspect_gguf(path, gguf, require_heads=True, head_quantization='F32'):
    reader = gguf.GGUFReader(path)
    architecture = reader.fields['general.architecture'].contents()
    profile = None
    for key, field in reader.fields.items():
        if key.endswith('.decision.laya_config'):
            profile = json.loads(field.contents())
    heads = {tensor.name: str(tensor.tensor_type) for tensor in reader.tensors
             if tensor.name.startswith('decision.')}
    if require_heads:
        if not profile or profile.get('readout') != 'candidate' or profile.get('box_hidden') != 256:
            raise ValueError('GGUF is missing the candidate/single-box decision metadata')
        expected = {'decision.readout.weight', 'decision.box_norm.weight', 'decision.box_norm.bias',
                    'decision.box_hidden.weight', 'decision.box_hidden.bias',
                    'decision.box_output.weight', 'decision.box_output.bias'}
        if set(heads) != expected:
            raise ValueError('GGUF decision tensors are missing or unexpected')
        matrices = {'decision.readout.weight', 'decision.box_hidden.weight', 'decision.box_output.weight'}
        if head_quantization not in ('F32', 'Q8_0', 'Q4_0', 'Q2_0', 'Q1_0'):
            raise ValueError('unsupported decision head quantization')
        for tensor in reader.tensors:
            wanted = (getattr(gguf.GGMLQuantizationType, head_quantization)
                      if tensor.name in matrices else gguf.GGMLQuantizationType.F32)
            if tensor.name in expected and tensor.tensor_type != wanted:
                raise ValueError('unexpected decision/localization tensor type: ' + tensor.name)
    return {'architecture': architecture, 'tensor_count': len(reader.tensors),
            'decision_profile': profile, 'decision_tensors': heads, 'bytes': path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--llama-cpp', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--name', default='jev-multimodal-qwen3.5-4b')
    parser.add_argument('--quantize', choices=('none', 'Q8_0', 'Q4_K_M'), default='Q8_0')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    repo = args.llama_cpp.resolve(strict=True)
    output = args.out.resolve()
    if output.exists() or output == checkpoint or checkpoint in output.parents:
        raise ValueError('output must be a new directory outside the checkpoint')
    if '/' in args.name or '\\' in args.name or args.threads < 1:
        raise ValueError('invalid output name or CPU thread count')
    settings = json.loads((checkpoint / 'multimodal_config.json').read_text())
    if settings['mode'] != 'full':
        raise ValueError('merge and validate the LoRA checkpoint first')
    converter = repo / 'convert_laya_to_gguf.py'
    if not converter.is_file():
        raise FileNotFoundError('a converter supporting native candidate and single-box heads is required')
    quantizer = repo / 'build-laya/bin/llama-quantize'
    if args.quantize != 'none' and not quantizer.is_file():
        raise FileNotFoundError(quantizer)
    sys.path.insert(0, str(repo / 'gguf-py'))
    import gguf
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=str(args.threads),
               OPENBLAS_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
               CUDA_VISIBLE_DEVICES='')
    output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = file_fingerprint(str(converter))
    with tempfile.TemporaryDirectory(prefix='.' + output.name + '-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'gguf'
        subprocess.run([sys.executable, str(converter), str(checkpoint), '--out-dir', str(stage),
                        '--outtype', 'bf16'], env=env, check=True)
        model = stage / (args.name + '-BF16.gguf')
        (stage / 'model.gguf').rename(model)
        (stage / 'mmproj.gguf').rename(stage / 'mmproj-BF16.gguf')
        records = {model.name: inspect_gguf(model, gguf),
                   'mmproj-BF16.gguf': inspect_gguf(stage / 'mmproj-BF16.gguf', gguf, False)}
        if args.quantize != 'none':
            target = stage / (args.name + '-' + args.quantize + '.gguf')
            subprocess.run([str(quantizer), '--tensor-type', r'decision\..*=f32',
                            str(model), str(target), args.quantize, str(args.threads)], env=env, check=True)
            records[target.name] = inspect_gguf(target, gguf)
        if file_fingerprint(str(converter)) != source_hash:
            raise RuntimeError('converter changed during conversion')
        report = {'status': 'structure_verified', 'source_format': 'merged_native_full',
                  'runtime_required': 'llama.cpp-jev with laya-multimodal decision support',
                  'converter_sha256': source_hash, 'decision_heads_preserved_fp32': True,
                  'quantization': args.quantize, 'files': records,
                  'behavioral_parity': 'Run a separate inference comparison before publication'}
        (stage / 'conversion-verification.json').write_text(json.dumps(report, indent=2) + '\n')
        checksums = {}
        for path in sorted(stage.iterdir()):
            if path.is_file():
                checksums[path.name] = file_fingerprint(str(path))
        (stage / 'SHA256SUMS').write_text(''.join(sha + '  ' + name + '\n' for name, sha in checksums.items()))
        stage.rename(output)
        print('GGUF STRUCTURE VERIFIED', output, flush=True)


if __name__ == '__main__':
    main()
