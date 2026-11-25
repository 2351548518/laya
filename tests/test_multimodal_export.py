"""Behavioral guards for native checkpoint conversion; no model downloads."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location('multimodal_export',
    Path(__file__).resolve().parents[1] / 'scripts/multimodal/export.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

GGUF_SPEC = importlib.util.spec_from_file_location('multimodal_export_gguf',
    Path(__file__).resolve().parents[1] / 'scripts/multimodal/export_gguf.py')
GGUF_MODULE = importlib.util.module_from_spec(GGUF_SPEC)
GGUF_SPEC.loader.exec_module(GGUF_MODULE)


def test_conversion_rejects_choice_changes_even_when_probabilities_are_close():
    before = {'q': {'type': 'choice', 'choice': 'a', 'probabilities': {'a': .501, 'b': .499}}}
    after = {'q': {'type': 'choice', 'choice': 'b', 'probabilities': {'a': .499, 'b': .501}}}
    with pytest.raises(ValueError, match='selected choice'):
        MODULE.compare_answers(before, after)


def test_conversion_checks_score_distributions_even_with_the_same_expectation():
    before = {'q': {'type': 'score', 'score': 1., 'probabilities': {'0': .1, '1': .8, '2': .1}}}
    after = {'q': {'type': 'score', 'score': 1., 'probabilities': {'0': .4, '1': .2, '2': .4}}}
    with pytest.raises(ValueError, match='probabilities'):
        MODULE.compare_answers(before, after)


def test_conversion_checks_box_quality_as_well_as_coordinates():
    before = {'q': {'type': 'box', 'box': {'left': .1, 'top': .2, 'right': .7, 'bottom': .8}, 'confidence': .8}}
    after = deepcopy(before)
    after['q']['confidence'] = .3
    with pytest.raises(ValueError, match='box quality'):
        MODULE.compare_answers(before, after)
    after['q']['confidence'] = .79
    after['q']['box']['left'] = .101
    report = MODULE.compare_answers(before, after)
    assert report['q']['max_coordinate_delta'] < .002
    assert before['q']['box']['left'] == .1


def test_conversion_rejects_missing_questions_and_changed_option_ids():
    before = {'q': {'type': 'choice', 'choice': 'a', 'probabilities': {'a': .9, 'b': .1}}}
    with pytest.raises(ValueError, match='question IDs'):
        MODULE.compare_answers(before, {})
    after = deepcopy(before)
    after['q']['probabilities'] = {'a': .9, 'c': .1}
    with pytest.raises(ValueError, match='options'):
        MODULE.compare_answers(before, after)


def test_gguf_rejects_quantized_heads_unless_explicitly_requested(tmp_path):
    path = tmp_path / 'model.gguf'
    path.write_bytes(b'fixture')
    names = ('readout.weight', 'box_norm.weight', 'box_norm.bias', 'box_hidden.weight',
             'box_hidden.bias', 'box_output.weight', 'box_output.bias')
    matrices = {'readout.weight', 'box_hidden.weight', 'box_output.weight'}
    tensors = [SimpleNamespace(name='decision.' + name, tensor_type=8 if name in matrices else 0)
               for name in names]
    reader = SimpleNamespace(tensors=tensors, fields={
        'general.architecture': SimpleNamespace(contents=lambda: 'qwen35'),
        'qwen35.decision.laya_config': SimpleNamespace(contents=lambda: json.dumps(
            {'readout': 'candidate', 'box_hidden': 256})),
    })
    gguf = SimpleNamespace(GGUFReader=lambda _: reader, GGMLQuantizationType=SimpleNamespace(F32=0, Q8_0=8))
    with pytest.raises(ValueError, match='tensor type'):
        GGUF_MODULE.inspect_gguf(path, gguf)
    result = GGUF_MODULE.inspect_gguf(path, gguf, head_quantization='Q8_0')
    assert len(result['decision_tensors']) == 7
    tensors[1].tensor_type = 8
    with pytest.raises(ValueError, match='box_norm.weight'):
        GGUF_MODULE.inspect_gguf(path, gguf, head_quantization='Q8_0')


def test_gguf_requires_all_localization_tensors(tmp_path):
    path = tmp_path / 'model.gguf'
    path.write_bytes(b'fixture')
    reader = SimpleNamespace(tensors=[SimpleNamespace(name='decision.readout.weight', tensor_type=0)], fields={
        'general.architecture': SimpleNamespace(contents=lambda: 'qwen35'),
        'qwen35.decision.laya_config': SimpleNamespace(contents=lambda: json.dumps(
            {'readout': 'candidate', 'box_hidden': 256})),
    })
    gguf = SimpleNamespace(GGUFReader=lambda _: reader, GGMLQuantizationType=SimpleNamespace(F32=0, Q8_0=8))
    with pytest.raises(ValueError, match='missing or unexpected'):
        GGUF_MODULE.inspect_gguf(path, gguf)


@pytest.mark.parametrize('head_type,code', [('Q4_0', 2), ('Q2_0', 41), ('Q1_0', 40)])
def test_gguf_low_bit_heads_keep_normalization_and_biases_float(tmp_path, head_type, code):
    path = tmp_path / 'model.gguf'
    path.write_bytes(b'fixture')
    matrices = {'readout.weight', 'box_hidden.weight', 'box_output.weight'}
    names = matrices | {'box_norm.weight', 'box_norm.bias', 'box_hidden.bias', 'box_output.bias'}
    tensors = [SimpleNamespace(name='decision.' + name, tensor_type=code if name in matrices else 0)
               for name in sorted(names)]
    reader = SimpleNamespace(tensors=tensors, fields={
        'general.architecture': SimpleNamespace(contents=lambda: 'qwen35'),
        'qwen35.decision.laya_config': SimpleNamespace(contents=lambda: json.dumps(
            {'readout': 'candidate', 'box_hidden': 256})),
    })
    types = SimpleNamespace(F32=0, **{head_type: code})
    gguf = SimpleNamespace(GGUFReader=lambda _: reader, GGMLQuantizationType=types)
    assert GGUF_MODULE.inspect_gguf(path, gguf, head_quantization=head_type)['tensor_count'] == 7
    next(t for t in tensors if t.name == 'decision.box_hidden.bias').tensor_type = code
    with pytest.raises(ValueError, match='box_hidden.bias'):
        GGUF_MODULE.inspect_gguf(path, gguf, head_quantization=head_type)


def test_pure_iq_rejects_missing_embedding_override_before_running_quantizer(tmp_path):
    for name in ('model.gguf', 'vision.gguf', 'imatrix.gguf'):
        (tmp_path / name).write_bytes(b'fixture')
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/multimodal/quantize_gguf.py'),
               '--model', str(tmp_path / 'model.gguf'), '--mmproj', str(tmp_path / 'vision.gguf'),
               '--imatrix', str(tmp_path / 'imatrix.gguf'), '--llama-cpp', str(tmp_path),
               '--out', str(tmp_path / 'output'), '--variants', 'IQ1_S', '--pure']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'explicit token embedding type' in result.stderr
    assert not (tmp_path / 'output').exists()
