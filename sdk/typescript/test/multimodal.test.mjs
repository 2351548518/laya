import assert from 'node:assert/strict';
import test from 'node:test';

import { Laya, LayaAPIError, LayaResponseError, LayaValidationError } from '../dist/esm/index.js';

const questions = { target: { type: 'box', instructions: 'the red cup' } };
const prediction = {
  model: 'native-qwen',
  answers: { target: { type: 'box', box: { left: 0.1, top: 0.2, right: 0.6, bottom: 0.9 }, confidence: 0.82 } },
  usage: { input_tokens: 96, output_tokens: 0, forward_passes: 1, truncated: false },
  routing: { model: 'multimodal', repo: null, reason: 'explicit model', detection: null, workflow: null },
};

test('System One image assets use the image header and accept nullable box confidence', async () => {
  const result = { model: 'native-qwen', usage: { input_tokens: 96, output_tokens: 0 },
    answers: { target: { ...prediction.answers.target, confidence: null } } };
  const assets = { photo: { type: 'image', format: 'png', data_base64: 'aW1hZ2U=' } };
  const state = ['photo', 'Locate the cup.'];
  const client = new Laya({ fetch: async (url, init) => {
    assert.match(url, /\/v1\/systemone$/);
    assert.equal(init.headers.get('typesafe-danger-zone-images'), 'true');
    assert.deepEqual(JSON.parse(init.body), { model: 'multimodal', state, questions, assets });
    return Response.json(result);
  } });
  assert.deepEqual(await client.predictImageAssets(state, questions, assets), result);
});

test('native image request preserves subarray bytes and returns normalized box', async () => {
  const bytes = Buffer.alloc(20006);
  for (let i = 0; i < bytes.length; i++) bytes[i] = i % 256;
  const image = bytes.subarray(3, 20004);
  const laya = new Laya({ baseURL: 'http://127.0.0.1:8000/local', model: 'english', apiKey: 'test-key',
    fetch: async (url, init) => {
      assert.equal(url, 'http://127.0.0.1:8000/local/v1/multimodal');
      assert.equal(init.headers.get('Authorization'), 'Bearer test-key');
      const { image: { base64 }, ...rest } = JSON.parse(init.body);
      assert.deepEqual(Buffer.from(base64, 'base64'), image);
      assert.deepEqual(rest, { questions, state: 'Inspect the tabletop', max_len: 512 });
      return Response.json(prediction);
    },
  });
  assert.deepEqual(await laya.predictMultimodal({ image }, questions, { state: 'Inspect the tabletop', maxLen: 512 }), prediction);
});

test('native audio and video keep decision questions and normal errors', async () => {
  for (const kind of ['audio', 'video']) {
    const q = { answer: { type: 'noul', instructions: 'Is the event present?' } };
    const result = { model: 'native', answers: { answer: { type: 'noul', noul: 0.8 } }, usage: prediction.usage };
    const laya = new Laya({ fetch: async (url, init) => {
      assert.match(url, /\/v1\/multimodal$/);
      assert.deepEqual(JSON.parse(init.body), { questions: q, [kind]: { base64: 'AAH/' } });
      return Response.json(result);
    } });
    assert.deepEqual(await laya.predictMultimodal({ [kind]: Uint8Array.of(0, 1, 255) }, q), result);
  }
  const unavailable = new Laya({ fetch: async () => Response.json({ detail: 'configure native model' }, { status: 503 }) });
  await assert.rejects(unavailable.predictMultimodal({ image: Uint8Array.of(1) }, questions), LayaAPIError);
});

test('native media reports an HTTP redirect as a request error', async () => {
  const laya = new Laya({ fetch: async (_url, init) => {
    assert.equal(init.redirect, 'manual');
    return new Response(null, { status: 307, headers: { Location: '/new-endpoint' } });
  } });
  await assert.rejects(
    laya.predictMultimodal({ image: Uint8Array.of(1) }, questions),
    error => error instanceof LayaAPIError && error.status === 307,
  );
});

test('native request validates media, box questions and response geometry', async () => {
  const image = Uint8Array.of(1);
  const laya = new Laya({ fetch: async () => assert.fail('invalid requests must not reach transport') });
  for (const media of [{}, { image, audio: image }, { image: new Uint8Array() }, { image: 'scene.jpg' }]) {
    await assert.rejects(laya.predictMultimodal(media, questions), LayaValidationError);
  }
  await assert.rejects(laya.predictMultimodal({ image }, { target: { ...questions.target, criteria: [] } }), LayaValidationError);
  await assert.rejects(laya.predictMultimodal({ image }, questions, { model: 'english' }), LayaValidationError);
  await assert.rejects(laya.predict('', questions), LayaValidationError);
  const wrong = structuredClone(prediction);
  wrong.answers.target.box.right = 0;
  const malformed = new Laya({ fetch: async () => Response.json(wrong) });
  await assert.rejects(malformed.predictMultimodal({ image }, questions), LayaResponseError);
});

test('health accepts the configured native checkpoint', async () => {
  const laya = new Laya({ fetch: async () => Response.json({ status: 'ok', loaded: ['multimodal'] }) });
  assert.deepEqual((await laya.health()).loaded, ['multimodal']);
});
