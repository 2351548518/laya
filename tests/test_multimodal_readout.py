"""Candidate scoring positions, padding and benchmark conversion contracts."""
import torch
import pytest

from laya.multimodal_readout import CandidateScorer,candidate_positions
from laya.multimodal_train import collate_multimodal_inputs


def test_candidate_positions_follow_media_expansion():
    offsets=torch.tensor([[0,0],[40,45],[45,48],[48,49],[49,55]])
    result=candidate_positions(offsets,35,[(25,28)])
    assert result.tolist()==[[2]]
    assert candidate_positions(offsets,35,[(25,28)]*8).tolist()==[[2]*8]


def legacy_positions(offsets, original_text_length, spans):
    shift = int(offsets[:, 1].max()) - original_text_length
    result = []
    for start, end in spans:
        selected = ((offsets[:, 0] >= start + shift) & (offsets[:, 1] <= end + shift)
                    & (offsets[:, 1] > offsets[:, 0])).nonzero().flatten()
        if not len(selected):
            raise ValueError('candidate marker has no matching tokenizer offsets')
        result.append(int(selected[-1]))
    return torch.tensor([result], dtype=torch.long)


@pytest.mark.parametrize('count', [0, 2, 8, 77, 256])
def test_candidate_positions_match_legacy_with_padding_repeated_and_overlapping_offsets(count):
    generator = torch.Generator().manual_seed(105)
    for _ in range(20):
        starts = torch.arange(384) * 3 + 50
        ends = starts + torch.randint(1, 4, (384,), generator=generator)
        offsets = torch.stack([starts, ends], dim=1).repeat_interleave(2, dim=0)
        offsets[::13] = 0
        available = (offsets[:, 1] > offsets[:, 0]).nonzero().flatten()
        selected = available[torch.randint(len(available), (count,), generator=generator)]
        spans = offsets[selected].clone()
        spans[:, 0] -= torch.randint(0, 9, (count,), generator=generator)
        shift = 29
        spans = (spans - shift).tolist()
        length = int(offsets[:, 1].max()) - shift
        expected = legacy_positions(offsets, length, spans)
        actual = candidate_positions(offsets, length, iter(spans))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(candidate_positions(offsets, length, torch.tensor(spans)), expected, rtol=0, atol=0)
        assert actual.dtype == torch.long and actual.device.type == 'cpu'


@pytest.mark.parametrize('nonmonotonic', [False, True])
def test_candidate_positions_preserve_nonmonotonic_fallback_and_missing_marker_errors(nonmonotonic):
    offsets = torch.tensor([[0,0], [20,22], [22,25], [25,28], [28,29], [29,30]])
    if nonmonotonic:
        offsets[[2, 4]] = offsets[[4, 2]]
    spans = [(20, 29)] * 8
    torch.testing.assert_close(candidate_positions(offsets, 30, spans), legacy_positions(offsets, 30, spans))
    with pytest.raises(ValueError, match='candidate marker has no matching tokenizer offsets'):
        candidate_positions(offsets, 30, [*spans, (10, 11)])


def test_candidate_positions_do_not_round_fractional_bounds():
    offsets = torch.stack([torch.arange(10), torch.arange(10)+1], dim=1)
    with pytest.raises(ValueError, match='candidate marker has no matching tokenizer offsets'):
        candidate_positions(offsets, 10, [(1.9, 2.1)]*8)
    torch.testing.assert_close(candidate_positions(offsets,10,[(1.1,3.1)]*8),
                               legacy_positions(offsets,10,[(1.1,3.1)]*8), rtol=0, atol=0)


def test_candidate_positions_left_padding_preserves_invalid_sentinel():
    batch=collate_multimodal_inputs([
        {'input_ids':torch.tensor([[4,5]]),'_candidate_positions':torch.tensor([[0,1]])},
        {'input_ids':torch.tensor([[6,7,8,9]]),'_candidate_positions':torch.tensor([[0,2,3]])}],0)
    assert batch['_candidate_positions'].tolist()==[[2,3,-1],[0,2,3]]


def test_shared_scorer_supports_77_candidates_and_masks_padding():
    scorer=CandidateScorer(12,8)
    hidden=torch.randn(2,100,12,requires_grad=True)
    positions=torch.arange(77).repeat(2,1);positions[0,-1]=-1
    result=scorer(hidden,positions)
    assert result.shape==(2,77) and result[0,-1]<-1000
    result[:,1].sum().backward()
    assert hidden.grad.abs().sum()>0 and scorer.query.weight.grad.abs().sum()>0


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('gradients',[False,True])
def test_native_candidate_logits_preserve_scores_and_training_gradients(dtype,gradients):
    import copy
    from types import SimpleNamespace
    from laya.multimodal import MultimodalAgent
    torch.manual_seed(31);torch.set_num_threads(1)
    original=CandidateScorer(16,8);actual_head=copy.deepcopy(original)
    hidden=torch.randn(2,7,16,dtype=dtype,requires_grad=gradients)
    other=hidden.detach().clone().requires_grad_(gradients)
    # Include repeated candidate indices and a candidate at the query position:
    # their gradients must add before BF16/FP16 rounding during training.
    positions=torch.tensor([[2,5,6,-1],[6,6,0,5]])
    agent=MultimodalAgent.__new__(MultimodalAgent)
    agent.device=torch.device('cpu');agent.dtype=dtype;agent.readout='candidate'
    agent.model=SimpleNamespace(model=lambda **kwargs:SimpleNamespace(last_hidden_state=other),decision_scorer=actual_head)
    with torch.set_grad_enabled(gradients):
        expected=original(hidden.float(),positions)
        actual=agent._logits({'_candidate_positions':positions},4)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        if gradients:
            actual.sum().backward();expected.sum().backward()
            torch.testing.assert_close(other.grad,hidden.grad,rtol=0,atol=0)
            for before,after in zip(original.parameters(),actual_head.parameters()):
                torch.testing.assert_close(after.grad,before.grad,rtol=0,atol=0)




def test_predecoded_video_preserves_official_sample_timestamps(tmp_path):
    import json
    Image = pytest.importorskip('PIL.Image')
    from laya.multimodal import MultimodalAgent
    path=tmp_path/'one.png';Image.new('RGB',(32,32),(12,34,56)).save(path)
    sequence=tmp_path/'clip.frames.json'
    sequence.write_text(json.dumps({'format':'laya_video_frames_v1','paths':[str(path),str(path)],
                                   'timestamps':[2.25,3.75],'duration_seconds':5.0}))
    agent=MultimodalAgent.__new__(MultimodalAgent);agent.frames=2;agent.max_audio_seconds=30.;agent.use_audio_in_video=False
    info,images,times,audio=agent._read_video(str(sequence))
    assert times==[2.25,3.75] and len(images)==2 and info['duration_s']==5.0 and audio is None


def test_real_processor_pads_multiple_audio_segments_when_cached(tmp_path):
    import numpy as np
    from pathlib import Path
    from transformers import AutoProcessor
    from laya.multimodal import MultimodalAgent
    cache=Path('models/gemma-4-E2B-it')
    if not cache.is_dir():pytest.skip('optional local Gemma processor is absent')
    sf = pytest.importorskip('soundfile')
    paths=[]
    for i,n in enumerate([16000,12000]):
        path=tmp_path/f'{i}.wav';sf.write(path,np.sin(np.arange(n)*.05)*.1,16000);paths.append(str(path))
    agent=MultimodalAgent.__new__(MultimodalAgent);agent.processor=AutoProcessor.from_pretrained(cache,local_files_only=True)
    agent.symbols=list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789');agent.device=torch.device('cpu')
    agent.dtype=torch.float32;agent.max_len=4096;agent.max_audio_seconds=30.;agent.model_type='gemma4'
    inputs,_,metadata=agent._prepare({'text':'Consecutive audio segments.','audio':paths},{'type':'noul','instructions':'Is there speech?'})
    assert inputs['input_features'].shape[0]==2
    mask=inputs['input_features_mask'];assert mask[0].sum()>mask[1].sum()>0
    assert metadata['audio_seconds']==[1.,.75]
