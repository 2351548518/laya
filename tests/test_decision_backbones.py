"""Backbone-specific adapters, native video, and shared optimizer policy."""
import json
import re

import pytest
import torch

from laya.grounding_train import GroundingTrainConfig
from laya.multimodal_objectives import lora_target_pattern, optimizer_groups


def tiny_qwen_config():
    from transformers import Qwen3_5Config
    return Qwen3_5Config(text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=128,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
        linear_value_head_dim=8, layer_types=['linear_attention'] * 3 + ['full_attention'],
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': 1.,
                         'mrope_section': [1, 1, 2]}),
        vision_config=dict(hidden_size=32, intermediate_size=64, depth=1, num_heads=4,
                           out_hidden_size=32), image_token_id=120, video_token_id=121)


def test_qwen_delta_adapters_receive_gradients_and_exclude_vision():
    Qwen3_5ForConditionalGeneration = pytest.importorskip(
        "transformers.models.qwen3_5.modeling_qwen3_5").Qwen3_5ForConditionalGeneration
    peft = pytest.importorskip("peft")
    LoraConfig, get_peft_model = peft.LoraConfig, peft.get_peft_model
    torch.manual_seed(105); torch.set_num_threads(1)
    model = Qwen3_5ForConditionalGeneration(tiny_qwen_config())
    pattern = lora_target_pattern('all-linear')
    names = [name for name, module in model.named_modules()
             if isinstance(module, torch.nn.Linear) and re.fullmatch(pattern, name)]
    assert len(names) == 31 and not any('visual' in name for name in names)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=pattern))
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    torch.nn.functional.cross_entropy(out.logits[0, -1:].float(), torch.tensor([10])).backward()
    for name, module in model.get_base_model().named_modules():
        if name.endswith('linear_attn.in_proj_qkv'):
            assert module.lora_B['default'].weight.grad.abs().sum() > 0
    assert sum(name.endswith('linear_attn.in_proj_qkv') for name in names) == 3


def test_native_qwen_candidate_scores_survive_left_padding_and_unequal_options():
    Qwen3_5ForConditionalGeneration = pytest.importorskip(
        "transformers.models.qwen3_5.modeling_qwen3_5").Qwen3_5ForConditionalGeneration
    from laya.multimodal import MultimodalAgent
    from laya.multimodal_readout import CandidateScorer
    from laya.multimodal_train import collate_multimodal_inputs
    torch.manual_seed(106); torch.set_num_threads(1)
    agent = MultimodalAgent.__new__(MultimodalAgent)
    agent.model = Qwen3_5ForConditionalGeneration(tiny_qwen_config()).eval()
    agent.model.decision_scorer = CandidateScorer(32)
    agent.device = torch.device('cpu'); agent.dtype = torch.float32; agent.readout = 'candidate'
    examples = []
    for ids, positions in [([2, 3, 4, 5], [1, 3]), ([2, 4, 6, 8, 10, 12, 14], [1, 4, 6])]:
        examples.append({'input_ids': torch.tensor([ids]), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long),
            'mm_token_type_ids': torch.zeros(1, len(ids), dtype=torch.long), '_candidate_positions': torch.tensor([positions])})
    with torch.no_grad():
        original = [agent._logits(example, example['_candidate_positions'].shape[1]) for example in examples]
        batched = agent._logits(collate_multimodal_inputs(examples, 0), 3)
    for index, expected in enumerate(original):
        torch.testing.assert_close(batched[index, :expected.shape[1]], expected[0], rtol=1e-4, atol=1e-4)
    assert batched[0, 2] < -1000


def test_merged_qwen_keeps_decision_and_box_inference_with_stale_peft_metadata():
    Qwen3_5ForConditionalGeneration = pytest.importorskip(
        "transformers.models.qwen3_5.modeling_qwen3_5").Qwen3_5ForConditionalGeneration
    peft = pytest.importorskip("peft")
    LoraConfig, get_peft_model = peft.LoraConfig, peft.get_peft_model
    from laya.grounding import GroundingHead
    from laya.multimodal import MultimodalAgent, _is_adapter_model
    from laya.multimodal_readout import CandidateScorer
    from laya.multimodal_train import MultimodalTrainConfig, configure_training
    torch.manual_seed(108); torch.set_num_threads(1)
    model = Qwen3_5ForConditionalGeneration(tiny_qwen_config()).eval()
    model.decision_scorer = CandidateScorer(32)
    model.grounding_head = GroundingHead(32, 16, 1, 'single-v1')
    agent = MultimodalAgent.__new__(MultimodalAgent)
    agent.model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj']))
    agent.model.eval(); agent.device = torch.device('cpu'); agent.dtype = torch.float32; agent.readout = 'candidate'
    ids = torch.tensor([[2, 3, 4, 5]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), _candidate_positions=torch.tensor([[1, 3]]))
    assert _is_adapter_model(agent.model)
    with torch.no_grad():
        expected = agent._logits(inputs, 2)
        boxes = agent._grounding_outputs(inputs)
        agent.model = agent.model.merge_and_unload(safe_merge=True)
        assert hasattr(agent.model, 'peft_config') and not _is_adapter_model(agent.model)
        torch.testing.assert_close(agent._logits(inputs, 2), expected, rtol=0, atol=0)
        for name, value in boxes.items():
            torch.testing.assert_close(agent._grounding_outputs(inputs)[name], value, rtol=0, atol=0)
    trainable = configure_training(agent, MultimodalTrainConfig(mode="head"))
    expected = list(agent.model.get_output_embeddings().parameters()) + list(agent.model.decision_scorer.parameters())
    assert {id(p) for p in trainable} == {id(p) for p in expected}


def test_length_bucketing_preserves_global_loss_and_gradients():
    Qwen3_5ForConditionalGeneration = pytest.importorskip(
        "transformers.models.qwen3_5.modeling_qwen3_5").Qwen3_5ForConditionalGeneration
    from laya.agent import Agent
    from laya.multimodal import MultimodalAgent
    from laya.multimodal_readout import CandidateScorer
    from laya.grounding_train import _decision_microbatches, _decision_batch_loss
    torch.manual_seed(107); torch.set_num_threads(1)
    agent=MultimodalAgent.__new__(MultimodalAgent)
    agent.model=Qwen3_5ForConditionalGeneration(tiny_qwen_config()).eval()
    agent.model.decision_scorer=CandidateScorer(32)
    agent.device=torch.device('cpu');agent.dtype=torch.float32;agent.readout='candidate'
    agent.tok=type('Tokenizer',(),{'pad_token_id':0})()
    lengths=[4,13,5,12]
    rows=[dict(id=str(i),state=str(i),questions={'q':{'type':'choice','instructions':'Choose',
          'criteria':['left','right']}},expected={'q':'left' if i%2 else 'right'},
          training_admission={'tokens':length}) for i,length in enumerate(lengths)]
    def prepare(request,qdef):
        length=lengths[int(request['text'])];ids=torch.arange(2,length+2)[None]
        return {'input_ids':ids,'attention_mask':torch.ones_like(ids),
            '_candidate_positions':torch.tensor([[1,length-2]])},Agent._to_internal(qdef),{}
    agent._prepare=prepare
    results=[]
    for enabled in (False,True):
        agent.model.zero_grad(set_to_none=True)
        cfg=GroundingTrainConfig(length_bucketing=enabled,micro_batch=4,max_micro_tokens=24,decision_brier_weight=.5)
        loss_sum=0.;padded=0
        for chunk,inputs in _decision_microbatches(agent,rows,cfg):
            padded+=inputs['input_ids'].numel()
            loss=_decision_batch_loss(agent,chunk,inputs,cfg)*len(chunk)/len(rows)
            loss.backward();loss_sum+=float(loss.detach())
        gradients={name:parameter.grad.clone() for name,parameter in agent.model.named_parameters() if parameter.grad is not None}
        results.append((loss_sum,padded,gradients))
    assert results[1][1]<results[0][1]
    assert results[0][0]==pytest.approx(results[1][0],abs=1e-5)
    assert results[0][2].keys()==results[1][2].keys()
    for name in results[0][2]:
        torch.testing.assert_close(results[0][2][name],results[1][2][name],rtol=2e-4,atol=1e-4)


def test_joint_optimizer_applies_head_and_projector_rates_without_duplicates():
    model = torch.nn.Module()
    for name in ('decision_scorer', 'grounding_head', 'lm_head', 'embed_audio', 'language_model'):
        setattr(model, name, torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 4)))
    model.language_model[1].weight.requires_grad_(False)
    groups = optimizer_groups(model, 2e-5, 1e-4, 3e-5)
    assigned = {id(parameter): group for group in groups for parameter in group['params']}
    assert len(assigned) == sum(len(group['params']) for group in groups)
    assert set(assigned) == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        group = assigned[id(parameter)]
        expected = 1e-4 if name.startswith(('decision_scorer', 'grounding_head', 'lm_head')) else 3e-5 if name.startswith('embed_audio') else 2e-5
        assert group['lr'] == group['base_lr'] == expected
        assert group['weight_decay'] == (0. if parameter.ndim < 2 else .01)


def test_indexed_training_preserves_unicode_labels_random_access_and_validation(tmp_path):
    from laya.grounding_train import _IndexedGroundingRows, load_grounding_rows
    path = tmp_path / 'train.jsonl'
    rows = [{'id': str(i), 'group': str(i), 'state': '决策\u2028测试' + str(i),
        'questions': {'q': {'type': 'noul', 'instructions': '可以继续吗？'}}, 'expected': {'q': i % 2}}
        for i in range(5)]
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows))
    indexed = _IndexedGroundingRows(path)
    assert len(indexed) == 5 and list(indexed) == load_grounding_rows(path) == rows
    assert [indexed[index] for index in [3, 0, -1, 2]] == [rows[index] for index in [3, 0, -1, 2]]
    assert len(indexed.offsets) * indexed.offsets.itemsize == 48
    with pytest.raises(IndexError): indexed[5]
    path2 = tmp_path / 'invalid.jsonl'; path2.write_text(json.dumps({'questions': {}}) + '\n')
    with pytest.raises(ValueError, match='exactly one'): _IndexedGroundingRows(path2)


@pytest.mark.parametrize('values', [{'projector_learning_rate': 0}, {'projector_learning_rate': True},
    {'warmup_ratio': -1}, {'min_lr_ratio': 1.1}, {'warmup_ratio': float('nan')}])
def test_joint_schedule_rejects_invalid_parameters(values):
    with pytest.raises(ValueError): GroundingTrainConfig(**values).validate()




def test_visual_only_training_rejects_lost_audio_and_wrong_native_processor(tmp_path):
    from scripts.multimodal.train import validate_visual_training
    for name in ('train', 'dev', 'decision_dev'):
        (tmp_path / (name + '.jsonl')).write_text(json.dumps({'id': name, 'state': 'text',
            'training_admission': {'processor': 'models/Qwen3.5-4B', 'max_len': 16384}}) + '\n')
    validate_visual_training(tmp_path, 'models/Qwen3.5-4B')
    with pytest.raises(ValueError, match='context budgets'):
        validate_visual_training(tmp_path, 'models/Qwen3.5-4B', 8192)
    for state in ({'audio': 'clip.wav'}, {'video': 'clip.mp4'}, {'video': 'clip.mp4', 'video_audio': True}):
        (tmp_path / 'train.jsonl').write_text(json.dumps({'id': 'unsafe', 'state': state}) + '\n')
        with pytest.raises(ValueError, match='audio cannot be discarded'):
            validate_visual_training(tmp_path, 'models/Qwen3.5-4B')
    (tmp_path / 'train.jsonl').write_text(json.dumps({'id': 'wrong', 'state': 'text',
        'training_admission': {'processor': 'models/gemma-4-E2B-it'}}) + '\n')
    with pytest.raises(ValueError, match='different backbone processor'):
        validate_visual_training(tmp_path, 'models/Qwen3.5-4B')
    (tmp_path / 'train.jsonl').write_text(json.dumps({'id': 'unadmitted', 'state': 'text'}) + '\n')
    with pytest.raises(ValueError, match='every training row'):
        validate_visual_training(tmp_path, 'models/Qwen3.5-4B')
