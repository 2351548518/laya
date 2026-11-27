"""Run the frozen scenario set with Laya or an explicitly configured Jev API."""
import argparse
import datetime
import getpass
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from audit import load_cases
from prompts import digest, interpret
from prompts import requests_for as frozen_request


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['laya', 'jev'], required=True)
    parser.add_argument('--checkpoint', type=Path, help='Local checkpoint; no automatic weight download')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--model', default=None, help='Jev model ID, or descriptive local Laya model ID')
    parser.add_argument('--checkpoint-revision', help='Optional user-supplied Hub revision; file hash is always recorded')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--limit', type=int, help='Smoke test only: evaluate first N cases')
    parser.add_argument('--modes', nargs='+', choices=['choice', 'four_noul'], default=['choice', 'four_noul'])
    args = parser.parse_args()
    if args.repeats < 1 or args.limit is not None and not 1 <= args.limit <= 64 or len(set(args.modes)) != len(args.modes):
        parser.error('Invalid repeat count, limit or duplicate mode')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new output directory; archived results must not be overwritten')
    cases, manifest = load_cases()
    cases = cases[:args.limit] if args.limit else cases
    metadata = dict(backend=args.backend, started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    cases_sha256=manifest['cases_sha256'], requests_sha256=manifest['requests_sha256'],
                    evaluated_ids=[c['id'] for c in cases], modes=args.modes, repeats=args.repeats,
                    smoke_test=len(cases) != 64 or args.repeats != 3 or len(args.modes) != 2,
                    python=platform.python_version(), os=platform.system(), architecture=platform.machine(),
                    seed=20260921, warmups=[], runner_sha256=file_sha(__file__))
    client = None
    if args.backend == 'jev':
        import httpx
        key = os.environ.get('TYPESAFE_API_KEY') or getpass.getpass('Jev API key (hidden): ')
        if not key.strip():
            parser.error('A Jev API key is required')
        client = httpx.Client(timeout=60, headers={'Authorization': 'Bearer ' + key.strip()})
        key = None
        model_id = args.model or 'jev-1.13.0'
        metadata.update(model=model_id, httpx=httpx.__version__)

        def call(req):
            start = time.perf_counter()
            response = client.post('https://api.typesafe.ai/v1/systemone', json={**req, 'model': model_id})
            if response.status_code != 200:
                raise RuntimeError('HTTP ' + str(response.status_code))
            result = response.json()
            elapsed = (time.perf_counter() - start) * 1000
            return {k: result[k] for k in ['model', 'answers', 'usage'] if k in result}, elapsed, {}
    else:
        if not args.checkpoint or not (args.checkpoint / 'multimodal_config.json').is_file():
            parser.error('--checkpoint must be a local native checkpoint with multimodal_config.json')
        files = sorted(args.checkpoint.glob('*.safetensors'))
        adapter = args.checkpoint / 'adapter'
        files.extend(sorted(adapter.glob('*.safetensors')))
        files.append(args.checkpoint / 'multimodal_config.json')
        if (adapter / 'adapter_config.json').is_file():
            files.append(adapter / 'adapter_config.json')
        os.environ.update(HF_HUB_OFFLINE='1', USE_TF='0', USE_TORCH='1', TOKENIZERS_PARALLELISM='false')
        import torch
        import transformers
        import laya_multimodal
        torch.set_num_threads(2)
        start = time.perf_counter()
        agent = laya_multimodal.load(str(args.checkpoint.resolve()), device=args.device)
        load_seconds = time.perf_counter() - start
        source = Path(laya_multimodal.__file__).resolve().parent
        try:
            commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True,
                                             stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        metadata.update(model=args.model or args.checkpoint.name,
                        checkpoint_revision=agent.checkpoint_revision or args.checkpoint_revision,
                        checkpoint_config=agent.cfg,
                        checkpoint_artifacts={file.relative_to(args.checkpoint).as_posix(): file_sha(file)
                                              for file in files},
                        load_seconds=load_seconds, torch=torch.__version__,
                        transformers=transformers.__version__, device=str(agent.device), cpu_threads=2,
                        laya_source_commit=commit,
                        laya_source_files={name: file_sha(source / name) for name in ['multimodal.py', 'common.py']})

        def call(req):
            if agent.device.type == 'mps':
                torch.mps.synchronize()
            elif agent.device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            response = agent.predict(req['state'], req['questions'])
            if agent.device.type == 'mps':
                torch.mps.synchronize()
            elif agent.device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            usage = response.get("usage", {})
            return response, elapsed, dict(device=str(agent.device),
                                           input_tokens=usage.get("input_tokens"),
                                           state_truncated=bool(usage.get("truncated")),
                                           instructions_truncated=False)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for mode in args.modes:
            try:
                result, elapsed, _ = call(frozen_request(cases[0])[mode])
                metadata['warmups'].append(dict(mode=mode, elapsed_ms=elapsed, usage=result.get('usage')))
            except Exception as e:
                metadata['warmups'].append(dict(mode=mode, error_type=type(e).__name__))
        (args.output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                                   encoding='utf-8', newline='\n')
        jobs = [(repeat, case, mode) for repeat in range(args.repeats) for case in cases for mode in args.modes]
        random.Random(20260921).shuffle(jobs)
        with (args.output / 'raw.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for i, (repeat, case, mode) in enumerate(jobs):
                req = frozen_request(case)[mode]
                row = dict(id=case['id'], family=case['family'], expected=case['expected'], mode=mode,
                           repeat=repeat, request_sha256=digest(req))
                try:
                    response, elapsed, diagnostics = call(req)
                    row.update(status='ok', elapsed_ms=elapsed, response=response, diagnostics=diagnostics,
                               **interpret(mode, response['answers']))
                except Exception as e:
                    row.update(status='error', error_type=type(e).__name__)
                    if isinstance(e, RuntimeError) and str(e).startswith('HTTP '):
                        row['http_status'] = str(e)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                if (i + 1) % 32 == 0:
                    print(f'{i + 1}/{len(jobs)} requests completed', flush=True)
        metadata['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (args.output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                                   encoding='utf-8', newline='\n')
    finally:
        if client:
            client.close()


if __name__ == '__main__':
    main()
