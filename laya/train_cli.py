"""Train native decisions and single-box grounding with local Gemma 4 or Qwen3.5."""
import argparse
from contextlib import nullcontext
from dataclasses import fields
from importlib.util import find_spec
import json
import os
from pathlib import Path

import torch

from .train import GroundingTrainConfig, finetune_grounding
from .multimodal import MultimodalAgent
from .multimodal_data import read_records
from .multimodal_distributed import distributed_context
from ._qwen_training import qwen_training_delta


def _visual_row_validator(model, max_len, require_admission):
    # Never silently drop the audio part of a trajectory to make a VLM fit.
    expected_processor = Path(model).resolve()
    verified_processors = set()

    def validate(split, row):
        state = row['state']
        if isinstance(state, dict) and (state.get('audio') is not None or
                (state.get('video') is not None and state.get('video_audio') is not False)):
            raise ValueError('Qwen3.5 requires an admitted visual-only view; audio cannot be discarded: ' + str(row.get('id')))
        admission = row.get('training_admission', {})
        if not isinstance(admission, dict):
            raise ValueError('training_admission must be an object')
        admitted = admission.get('processor')
        if split == 'train' and not admitted and require_admission:
            raise ValueError('Qwen3.5 training requires native processor admission for every training row')
        if admitted and admitted not in verified_processors:
            if Path(admitted).resolve() != expected_processor:
                raise ValueError('Training view was admitted using a different backbone processor: ' + str(admitted))
            verified_processors.add(admitted)
        if max_len is not None and admitted and admission.get('max_len') != max_len:
            raise ValueError('Native admission and training context budgets must match')

    return validate


def validate_visual_training(root, model, max_len=None, require_admission=True, *, train_split='train'):
    validate = _visual_row_validator(model, max_len, require_admission)
    for split in (train_split, 'dev', 'decision_dev'):
        for row in read_records(root / (split + '.jsonl'), verify_media=False):
            validate('train' if split == train_split else split, row)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="laya-train", description=__doc__)
    parser.add_argument('--model', required=True, help='Local backbone or native checkpoint directory')
    parser.add_argument('--data', required=True, help='Directory with the prepared JSONL splits')
    parser.add_argument('--out', required=True, help='Run and checkpoint directory')
    parser.add_argument('--config', required=True, help='Joint training settings in JSON')
    parser.add_argument('--train-split', choices=['train', 'warmup'], default='train')
    parser.add_argument('--require-original-base', action='store_true')
    parser.add_argument('--visual-only', action='store_true')
    parser.add_argument('--allow-unadmitted', action='store_true',
        help='Allow local JSONL without stored processor receipts; native media and context checks still apply')
    parser.add_argument('--qwen-training-delta', action='store_true',
        help='Use the local PyTorch chunk scan for Qwen3.5 gradient computation')
    parser.add_argument('--resume', help='Run directory or optimizer snapshot to resume with the original model and data')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic PyTorch training operations')
    parser.add_argument('--text-attention', choices=['sdpa', 'flex_attention'], default='sdpa')
    parser.add_argument('--swanlab', choices=('local', 'online'), help='Record training metrics with SwanLab on rank zero')
    parser.add_argument('--swanlab-project', default='laya-multimodal', help='SwanLab project name')
    parser.add_argument('--swanlab-name', help='Override the SwanLab experiment name derived from --out')
    args = parser.parse_args(argv)
    if args.swanlab and (find_spec('swanlab') is None or (args.swanlab == 'local' and find_spec('swanboard') is None)):
        parser.error("--swanlab requires installing the SwanLab extra: pip install -e '.[multimodal,swanlab]'")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    values = json.loads(Path(args.config).read_text())
    if set(values) - {field.name for field in fields(GroundingTrainConfig)}:
        raise ValueError('unknown training fields')
    config = GroundingTrainConfig(**values)
    config.validate()
    ctx = distributed_context()

    model_dir = Path(args.model)
    checkpoint = model_dir / 'multimodal_config.json'
    saved = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    source = saved.get('backbone', args.model)
    model_type = json.loads((Path(source) / 'config.json').read_text())['model_type']
    if model_type not in {'gemma4', 'qwen3_5'}:
        raise ValueError('joint training requires Gemma 4 or Qwen3.5')
    if args.visual_only != (model_type == 'qwen3_5'):
        raise ValueError('Qwen3.5 requires --visual-only; Gemma4 keeps the full-omni recipe')
    if args.qwen_training_delta and model_type != 'qwen3_5':
        raise ValueError('--qwen-training-delta requires Qwen3.5')
    root = Path(args.data)
    row_validator = (_visual_row_validator(source, config.max_len, not args.allow_unadmitted)
                     if args.visual_only else None)
    if args.require_original_base and (checkpoint.exists() or (model_dir / 'adapter_config.json').exists()):
        raise ValueError('fresh-base training must not load a decision checkpoint or adapter')
    # Model construction precedes finetune_grounding's own seed reset.
    torch.manual_seed(config.seed)
    agent = MultimodalAgent(
        args.model, device=ctx.device, max_len=config.max_len, frames=4, max_audio_seconds=30.,
        use_audio_in_video=not args.visual_only, readout='candidate', prompt_version='v2', grounding_slots=1,
    )
    agent._training_variant = {
        "text_attention": args.text_attention,
        "qwen_training_delta": args.qwen_training_delta,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }
    if args.text_attention != 'sdpa':
        if model_type != 'gemma4':
            raise ValueError('Experimental text attention currently validated for Gemma only')
        agent.model.set_attn_implementation({
            '': 'sdpa', 'text_config': args.text_attention, 'vision_config': 'sdpa', 'audio_config': 'eager',
        })
    inherited_adapters = hasattr(agent.model, 'peft_config')
    base = agent.model.get_base_model() if inherited_adapters else agent.model
    if base.grounding_head.architecture != 'single-v1':
        raise ValueError('expected compact single-box head')
    if args.require_original_base and inherited_adapters:
        raise ValueError('inherited adapters forbidden')
    if ctx.main:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        initialization = {
            'model': str(model_dir.resolve()),
            'seed_before_head_construction': config.seed,
            'original_base': args.require_original_base,
            'readout': agent.readout,
            'prompt_version': agent.prompt_version,
            'grounding_architecture': 'single-v1',
            'inherited_adapters': inherited_adapters,
            'fresh_optimizer': args.resume is None,
            'world_size': ctx.world_size,
            **agent._training_variant,
        }
        (out / 'initialization.json').write_text(json.dumps(initialization, indent=2) + '\n')
    reporter = None
    if ctx.main and args.swanlab:
        from ._swanlab import SwanLabReporter

        reporter = SwanLabReporter(args.out, mode=args.swanlab, project=args.swanlab_project, name=args.swanlab_name)
    with (
        reporter if reporter is not None else nullcontext(),
        qwen_training_delta() if args.qwen_training_delta else nullcontext(),
    ):
        finetune_grounding(
            root / (args.train_split + '.jsonl'), root / 'dev.jsonl', args.out, model=args.model,
            decision_dev_data=root / 'decision_dev.jsonl', config=config, agent=agent,
            row_validator=row_validator, resume_from=args.resume, reporter=reporter,
        )


if __name__ == '__main__':
    main()
