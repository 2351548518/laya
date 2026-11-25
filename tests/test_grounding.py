"""Geometry, set assignment, standard Box schema and non-generative native integration."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from laya.agent import Agent
from laya.grounding import GroundingHead, box_answer, box_iou, grounding_loss, last_valid_state, validate_boxes
from laya.grounding_train import GroundingTrainConfig
from laya.multimodal import MultimodalAgent


def test_box_geometry_and_masked_last_state():
    a = torch.tensor([[.1, .2, .6, .7]])
    torch.testing.assert_close(box_iou(a, a), torch.ones(1, 1))
    torch.testing.assert_close(box_iou(a, a, True), torch.ones(1, 1))
    hidden = torch.arange(24.).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    torch.testing.assert_close(last_valid_state(hidden, mask), torch.stack([hidden[0, 1], hidden[1, 2]]))
    with pytest.raises(ValueError):validate_boxes([[.8, .1, .2, .7]])
    with pytest.raises(ValueError):validate_boxes([[0., 0., float('nan'), 1.]])
    answer = box_answer(a[0], .8)
    assert set(answer) == {'type', 'box', 'confidence'}
    assert set(answer['box']) == {'top', 'bottom', 'left', 'right'}


@pytest.mark.parametrize('architecture',['set-v1','set-v2'])
def test_single_head_migration_preserves_outputs_input_gradients_and_trained_weights(architecture):
    torch.manual_seed(19);torch.set_num_threads(1)
    legacy=GroundingHead(16,32,4,architecture).eval();compact=legacy.to_single()
    assert compact.architecture=='single-v1' and compact.slots==1
    assert all(k.startswith('single.') for k in compact.state_dict())
    assert not any(hasattr(compact,k) for k in ['queries','decoder','memory','objects','attention'])
    assert sum(p.numel() for p in compact.parameters())<sum(p.numel() for p in legacy.parameters())
    hidden=torch.randn(3,7,16,requires_grad=True);other=hidden.detach().clone().requires_grad_(True)
    mask=torch.tensor([[1,1,1,0,0,0,0],[0,0,1,1,1,1,1],[1,1,1,1,1,1,1]])
    original=legacy(hidden,mask);actual=compact(other,mask)
    for key in original:torch.testing.assert_close(actual[key],original[key],atol=0,rtol=0)
    sum(v.sum() for v in original.values()).backward();sum(v.sum() for v in actual.values()).backward()
    torch.testing.assert_close(hidden.grad,other.grad,atol=0,rtol=0)
    with pytest.raises(ValueError,match='multi-object'):compact(other,multiple=True)


def test_single_box_default_trains_without_importing_the_set_matcher(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'scipy.optimize',None)
    head=GroundingHead(16,32);assert head.architecture=='single-v1'
    output=head(torch.randn(2,4,16))
    loss,_=grounding_loss(output,[[[.1,.2,.8,.9]],[]]);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(head.single[-1].weight.grad).all()


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('padding',['none','left','right'])
def test_single_box_selects_before_float_cast_with_exact_outputs_and_gradients(dtype,padding):
    import copy
    from laya.grounding import ordered_boxes
    torch.manual_seed(27);torch.set_num_threads(1)
    original=GroundingHead(16,32);optimized=copy.deepcopy(original)
    hidden=torch.randn(3,29,16,dtype=dtype,requires_grad=True)
    other=hidden.detach().clone().requires_grad_(True)
    mask=torch.ones(3,29,dtype=torch.long)
    if padding=='left':mask[:,:7]=0
    elif padding=='right':mask[:,19:]=0
    else:mask=None
    raw=original.single(last_valid_state(hidden.float(),mask))[:,None]
    expected={'boxes':ordered_boxes(raw[...,:4]),'quality_logits':raw[...,4]}
    # Exercise the surrounding autocast used by native backbone inference too.
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):actual=optimized(other,mask)
    for key in expected:torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
    sum(value.sum() for value in expected.values()).backward()
    sum(value.sum() for value in actual.values()).backward()
    torch.testing.assert_close(other.grad,hidden.grad,atol=0,rtol=0)
    for before,after in zip(original.parameters(),optimized.parameters()):
        torch.testing.assert_close(after.grad,before.grad,atol=0,rtol=0)


def test_single_box_selection_does_not_average_in_an_untrained_set_task():
    from laya.grounding_train import checkpoint_score,grounding_retained
    metrics={'single_n':10,'set_n':0,'acc_iou50':.8,'set_f1_50':0.}
    assert checkpoint_score(metrics)==.8
    assert grounding_retained(dict(metrics,acc_iou50=.78),metrics,.03)
    assert not grounding_retained(dict(metrics,acc_iou50=.7),metrics,.03)


def test_single_training_rejects_multi_target_rows_before_loading_a_backbone(tmp_path):
    import json
    from laya.grounding_train import finetune_grounding
    row={'group':'a','state':{'image':'unused.jpg'},'questions':{'q':{'type':'box','instructions':'the mug'}},
        'grounding':{'task':'multiple','boxes':[[.1,.1,.2,.3],[.5,.5,.7,.8]]}}
    (tmp_path / 'unused.jpg').touch()
    train=tmp_path/'train.jsonl';dev=tmp_path/'dev.jsonl'
    train.write_text(json.dumps(row)+'\n');dev.write_text(json.dumps(dict(row,group='b'))+'\n')
    with pytest.raises(ValueError,match='never relabel'):
        finetune_grounding(train,dev,tmp_path/'out',model='unused-backbone')


def test_head_learns_geometry_and_set_matching_is_permutation_invariant():
    torch.manual_seed(9);torch.set_num_threads(1)
    head = GroundingHead(16, 32, 4)
    hidden = torch.randn(2, 5, 16);mask = torch.ones(2, 5, dtype=torch.long)
    truth = [[[.05, .1, .25, .5]], [[.5, .3, .9, .8]]]
    optimizer = torch.optim.Adam(head.parameters(), lr=.015)
    first = float(grounding_loss(head(hidden, mask), truth)[0].detach())
    for _ in range(60):
        optimizer.zero_grad();loss, _ = grounding_loss(head(hidden, mask), truth);loss.backward();optimizer.step()
    assert float(grounding_loss(head(hidden, mask), truth)[0].detach()) < first * .3
    multi = head(hidden, mask, multiple=True)
    target = [[[.1, .1, .2, .2], [.5, .5, .8, .9]], []]
    forward, _ = grounding_loss(multi, target, multiple=True)
    reverse, _ = grounding_loss(multi, [target[0][::-1], []], multiple=True)
    torch.testing.assert_close(forward, reverse)
    forward.backward()
    assert head.queries.grad is not None and torch.isfinite(head.queries.grad).all()
    with pytest.raises(ValueError, match='slots'):
        grounding_loss(multi, [target[0] * 3, []], multiple=True)


def test_empty_targets_have_no_fabricated_regression_loss():
    head = GroundingHead(16, 32, 4);out = head(torch.randn(1, 3, 16), multiple=True)
    loss, parts = grounding_loss(out, [[]], multiple=True)
    assert parts['l1'] == parts['giou'] == 0 and torch.isfinite(loss)
    loss.backward()
    assert head.objects[-1].weight.grad is not None


def test_small_prediction_inside_crowd_is_ignored():
    boxes=torch.tensor([[[.1,.1,.2,.2]]],requires_grad=True)
    out={'boxes':boxes,'object_logits':torch.zeros(1,1,requires_grad=True),
         'quality_logits':torch.zeros(1,1,requires_grad=True)}
    loss,parts=grounding_loss(out,[[]],multiple=True,ignored=[[[0.,0.,1.,1.]]])
    assert all(value==0. for value in parts.values())
    loss.backward()
    assert out['object_logits'].grad.abs().sum()==0


def test_box_contract_is_multimodal_only_and_requires_no_criteria():
    q = {'type': 'box', 'instructions': 'the red mug'}
    MultimodalAgent._check_question('mug', q)
    with pytest.raises(ValueError):Agent._check_question('mug', q)
    with pytest.raises(ValueError, match='criteria'):MultimodalAgent._check_question('mug', dict(q, criteria=[]))
    with pytest.raises(ValueError):MultimodalAgent._check_question('mug', dict(q, instructions=''))
    with pytest.raises(ValueError):MultimodalAgent._check_question('mug', dict(q, instructions={'find':'mug'}))


def test_mixed_box_and_noul_share_one_local_backbone(monkeypatch):
    from test_multimodal import tiny_agent
    agent = tiny_agent();backbone = agent.model
    class Features(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            backbone.calls += 1
            return SimpleNamespace(last_hidden_state=backbone.embeddings(input_ids))
    agent.model.model = Features();agent.model.grounding_head = GroundingHead(8, 32, 3)
    agent.grounding_trained = agent.detection_trained = True
    image = np.full((8, 8, 3), 127, np.uint8)
    questions = {'mug': {'type': 'box', 'instructions': 'the mug'},
                 'visible': {'type': 'noul', 'instructions': 'Is a mug visible?'}}
    result = agent.predict(image=image, questions=questions)
    assert result['answers']['mug']['type'] == 'box'
    assert set(result['answers']['mug']['box']) == {'top', 'bottom', 'left', 'right'}
    assert result['usage']['output_tokens'] == 0 and result['usage']['forward_passes'] == 2
    detected = agent.detect(image=image, labels=['mug', 'person'], threshold=0.)
    assert len(detected['objects']) == 6 and detected['usage']['forward_passes'] == 2
    assert {x['label'] for x in detected['objects']} == {'mug', 'person'}
    agent.detection_trained = False
    with pytest.raises(ValueError, match='multi-object'):agent.detect(image=image, labels=['mug'])
    agent.grounding_trained = False
    with pytest.raises(ValueError, match='not been trained'):agent.predict(image=image, questions={'mug': questions['mug']})


@pytest.mark.parametrize('architecture', ['set-v1', 'set-v2'])
def test_head_safetensor_roundtrip(tmp_path, architecture):
    from safetensors.torch import save_file, load_file
    torch.manual_seed(0);head = GroundingHead(16, 32, 4, architecture)
    hidden = torch.randn(2, 5, 16);expected = head(hidden, multiple=True)
    p = tmp_path/'grounding_head.safetensors';save_file(head.state_dict(), p)
    restored = GroundingHead(16, 32, 4, architecture);restored.load_state_dict(load_file(p))
    actual = restored(hidden, multiple=True)
    for key in expected:torch.testing.assert_close(actual[key], expected[key])


def test_grounding_training_configuration_rejects_invalid_modes():
    GroundingTrainConfig().validate()
    with pytest.raises(ValueError):GroundingTrainConfig(mode='generate').validate()
    with pytest.raises(ValueError):GroundingTrainConfig(slots=0).validate()
    with pytest.raises(ValueError):GroundingTrainConfig(max_steps=1.5).validate()
    with pytest.raises(ValueError):GroundingTrainConfig(learning_rate=True).validate()
    with pytest.raises(ValueError):GroundingTrainConfig(lora_targets='vision').validate()
    with pytest.raises(ValueError):GroundingTrainConfig(micro_batch=0).validate()
    with pytest.raises(ValueError):GroundingTrainConfig(max_micro_tokens=True).validate()


@pytest.mark.parametrize('multiple',[False,True])
def test_batched_geometry_preserves_per_example_gradients(multiple):
    import copy
    torch.manual_seed(41);torch.set_num_threads(1)
    batched=GroundingHead(16,32,4);sequential=copy.deepcopy(batched)
    hidden=torch.randn(3,6,16);mask=torch.tensor([[0,0,1,1,1,1],[1,1,1,1,1,1],[0,1,1,1,1,1]])
    targets=[[[.1,.1,.3,.7]],[],[[.6,.2,.9,.8]]]
    if multiple:targets[0].append([.5,.1,.9,.9])
    joint,_=grounding_loss(batched(hidden,mask,multiple=multiple),targets,multiple=multiple)
    (joint*3/7).backward()
    for i in range(3):
        individual,_=grounding_loss(sequential(hidden[i:i+1],mask[i:i+1],multiple=multiple),[targets[i]],multiple=multiple)
        (individual/7).backward()
    for name,parameter in batched.named_parameters():
        other=dict(sequential.named_parameters())[name]
        if parameter.grad is None or other.grad is None:assert parameter.grad is other.grad is None
        else:torch.testing.assert_close(parameter.grad,other.grad,atol=2e-6,rtol=1e-4)


def test_geometry_microbatch_respects_head_and_token_limits():
    from laya.grounding_train import _geometry_microbatches
    def prepare(state,question):
        length=state['length']
        return {'input_ids':torch.ones(1,length,dtype=torch.long),'attention_mask':torch.ones(1,length,dtype=torch.long)},None,None
    agent=SimpleNamespace(tok=SimpleNamespace(pad_token_id=0),_input=lambda state:state,_prepare=prepare)
    rows=[{'id':str(i),'state':{'length':length},'questions':{'x':{'type':'box','instructions':'target'}},
        'grounding':{'task':task,'boxes':[]}} for i,(length,task) in enumerate([(3,'single'),(4,'single'),(6,'single'),(7,'multiple')])]
    chunks=list(_geometry_microbatches(agent,rows,GroundingTrainConfig(micro_batch=3,max_micro_tokens=8)))
    assert sorted(row['id'] for chunk,_,_ in chunks for row in chunk)==['0','1','2','3']
    assert sorted(len(chunk) for chunk,_,_ in chunks)==[1,1,2]
    for chunk,inputs,multiple in chunks:
        assert all((row['grounding']['task']=='multiple')==multiple for row in chunk)
        assert inputs['input_ids'].numel()<=8 or len(chunk)==1


def test_lora_mode_activates_checkpointed_language_and_freezes_tower_modes():
    from laya.grounding_train import _grounding_train_mode
    model=torch.nn.Module();model.model=torch.nn.Module()
    model.model.vision_tower=torch.nn.Dropout(.5);model.model.audio_tower=torch.nn.Dropout(.5)
    model.model.language_model=torch.nn.Linear(4,4);model.grounding_head=GroundingHead(4,32,4)
    agent=SimpleNamespace(model=model)
    _grounding_train_mode(agent,GroundingTrainConfig(mode='lora'))
    assert model.training and model.model.language_model.training
    assert not model.model.vision_tower.training and not model.model.audio_tower.training
    _grounding_train_mode(agent,GroundingTrainConfig(mode='head'))
    assert not model.training and not model.model.language_model.training and model.grounding_head.training




def test_grounding_rows_cannot_silently_discard_extra_questions(tmp_path):
    import json
    from laya.grounding_train import load_grounding_rows
    path=tmp_path/'rows.jsonl'
    q={'type':'box','instructions':'the mug'}
    path.write_text(json.dumps({'state': '', 'questions':{'a':q,'b':q}})+'\n')
    with pytest.raises(ValueError,match='exactly one'):load_grounding_rows(path)


def test_retention_guard_checks_every_modality():
    from laya.grounding_train import decision_retained
    before={'by_modality':{'text':{'accuracy':.7},'audio':{'accuracy':.7}}}
    after={'by_modality':{'text':{'accuracy':.9},'audio':{'accuracy':.5}}}
    assert not decision_retained(after,before,.05)


def test_joint_validation_accepts_hard_and_soft_decision_target_formats():
    from laya.grounding_train import grounding_metrics
    from test_multimodal import tiny_agent

    agent = tiny_agent()
    row = {"state": "A scene with a mug", "questions": {"visible": {
        "type": "noul", "instructions": "Is the mug visible?"}}}
    hard = {**row, "expected": {"visible": True}}
    soft = {**row, "gold": {"visible": {"probabilities": {"false": 0., "true": 1.}}}}
    hard_metrics = grounding_metrics(agent, [hard], limit=0)
    assert hard_metrics["presence_n"] == 1
    assert hard_metrics == grounding_metrics(agent, [soft], limit=0)


def test_permuted_decision_replay_preserves_training_epoch_order(tmp_path):
    import json
    from test_multimodal import tiny_agent
    from laya.grounding_train import finetune_grounding
    agent=tiny_agent();agent.model.grounding_head=GroundingHead(8,32,4)
    agent.model.peft_config = {}  # Metadata retained by merged base models.
    agent.grounding_trained=agent.detection_trained=False
    agent.save=lambda path:None
    original=agent.model.get_output_embeddings().weight.detach().clone()
    question={'type':'choice','instructions':'Choose the category','criteria':['dark','bright'],'option_order':[1,0]}
    row={'group':'train','state':'training','questions':{'x':question},
         'gold':{'x':{'probabilities':{'dark':0.,'bright':1.}}}}
    train=tmp_path/'train.jsonl';train.write_text(json.dumps(row)+'\n')
    dev=tmp_path/'dev.jsonl';dev.write_text(json.dumps({'group':'dev','state':'held out',
        'questions':{'x':{'type':'noul','instructions':'Is it visible?'}},
        'gold':{'x':{'probabilities':{'false':0.,'true':1.}}}})+'\n')
    output=tmp_path/'out'
    finetune_grounding(train,dev,output,agent=agent,config=GroundingTrainConfig(epochs=3,max_steps=3,
        global_batch_size=1,learning_rate=.01,train_readout_in_warmup=True,eval_every=0,
        video_cache_dir=None,video_cache_size=5))
    report=json.loads((output/'training_report.json').read_text())
    assert report['steps']==3
    assert report['forward_passes']=={'geometry':0,'decisions':3}
    assert agent._video_cache_dir is None and agent._video_cache_size == 5
    assert {group['base_lr'] for group in report['optimizer_groups'] if group['name']=='head'}=={.0002}
    assert not torch.equal(original,agent.model.get_output_embeddings().weight)
