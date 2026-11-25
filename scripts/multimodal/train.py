"""Seed before fresh heads; full-omni Gemma and explicit visual-only Qwen training."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from laya.grounding_train import GroundingTrainConfig,finetune_grounding
from laya.multimodal import MultimodalAgent
from laya.multimodal_data import read_records
from laya.multimodal_distributed import distributed_context


def validate_visual_training(root, model, max_len=None, require_admission=True, *, train_split='train'):
    # Never silently drop the audio part of a trajectory to make a VLM fit.
    expected_processor=Path(model).resolve();verified_processors=set()
    for split in (train_split, 'dev', 'decision_dev'):
        name = split + '.jsonl'
        for row in read_records(root/name, verify_media=False):
            state=row['state']
            if isinstance(state,dict) and (state.get('audio') is not None or
                    (state.get('video') is not None and state.get('video_audio') is not False)):
                raise ValueError('Qwen3.5 requires an admitted visual-only view; audio cannot be discarded: '+str(row.get('id')))
            admitted=row.get('training_admission',{}).get('processor')
            if split == train_split and not admitted and require_admission:
                raise ValueError('Qwen3.5 training requires native processor admission for every training row')
            if admitted and admitted not in verified_processors:
                if Path(admitted).resolve()!=expected_processor:
                    raise ValueError('Training view was admitted using a different backbone processor: '+str(admitted))
                verified_processors.add(admitted)
            if max_len is not None and admitted and row['training_admission'].get('max_len')!=max_len:
                raise ValueError('Native admission and training context budgets must match')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True);parser.add_argument('--data',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--config',required=True)
    parser.add_argument('--train-split', choices=['train', 'warmup'], default='train')
    parser.add_argument('--require-original-base',action='store_true')
    parser.add_argument('--visual-only',action='store_true')
    parser.add_argument('--allow-unadmitted',action='store_true',
        help='Allow local JSONL without stored processor receipts; native media and context checks still apply')
    parser.add_argument('--text-attention',choices=['sdpa','flex_attention'],default='sdpa');args=parser.parse_args()
    values=json.loads(Path(args.config).read_text())
    if set(values)-{field.name for field in fields(GroundingTrainConfig)}:raise ValueError('unknown training fields')
    config=GroundingTrainConfig(**values);config.validate();ctx=distributed_context()
    checkpoint=Path(args.model)/'multimodal_config.json'
    saved=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    source=saved.get('backbone',args.model)
    model_type=json.loads((Path(source)/'config.json').read_text())['model_type']
    if model_type not in {'gemma4','qwen3_5'}:raise ValueError('v5 supports Gemma4 or Qwen3.5')
    if args.visual_only!=(model_type=='qwen3_5'):
        raise ValueError('Qwen3.5 requires --visual-only; Gemma4 keeps the full-omni recipe')
    root=Path(args.data)
    if args.visual_only:
        validate_visual_training(root, source, config.max_len, not args.allow_unadmitted,
                                 train_split=args.train_split)
    if args.require_original_base:
        if (Path(args.model)/'multimodal_config.json').exists() or (Path(args.model)/'adapter_config.json').exists():
            raise ValueError('fresh-base training must not load a decision checkpoint or adapter')
    # Model construction precedes finetune_grounding's own seed reset.
    torch.manual_seed(config.seed)
    agent=MultimodalAgent(args.model,device=ctx.device,max_len=config.max_len,frames=4,
        max_audio_seconds=30.,use_audio_in_video=not args.visual_only,readout='candidate',prompt_version='v2',grounding_slots=1)
    if args.text_attention!='sdpa':
        if model_type!='gemma4':raise ValueError('Experimental text attention currently validated for Gemma only')
        agent.model.set_attn_implementation({'':'sdpa','text_config':args.text_attention,'vision_config':'sdpa','audio_config':'eager'})
    base=agent.model.get_base_model() if hasattr(agent.model,'peft_config') else agent.model
    if base.grounding_head.architecture!='single-v1':raise ValueError('expected compact single-box head')
    if args.require_original_base and hasattr(agent.model,'peft_config'):raise ValueError('inherited adapters forbidden')
    if ctx.main:
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        (out/'initialization.json').write_text(json.dumps({'model':str(Path(args.model).resolve()),
            'seed_before_head_construction':config.seed,'original_base':args.require_original_base,
            'readout':agent.readout,'prompt_version':agent.prompt_version,'grounding_architecture':'single-v1',
            'inherited_adapters':hasattr(agent.model,'peft_config'),
            'fresh_optimizer':True,'world_size':ctx.world_size,'text_attention':args.text_attention},indent=2)+'\n')
    finetune_grounding(root/(args.train_split+'.jsonl'),root/'dev.jsonl',args.out,model=args.model,
        decision_dev_data=root/'decision_dev.jsonl',config=config,agent=agent)


if __name__=='__main__':main()
