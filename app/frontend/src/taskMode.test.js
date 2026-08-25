import assert from 'node:assert/strict';
import test from 'node:test';

import { resolveRequestTask } from './taskMode.js';

test('preserves the selected text-search task mode', () => {
  assert.equal(resolveRequestTask('KIS', false), 'KIS');
  assert.equal(resolveRequestTask('VQA', false), 'VQA');
  assert.equal(resolveRequestTask('TRAKE', false), 'TRAKE');
});

test('uses VKIS for an image-reference search', () => {
  assert.equal(resolveRequestTask('TRAKE', true), 'VKIS');
});

test('falls back to KIS for an unknown task mode', () => {
  assert.equal(resolveRequestTask('unexpected', false), 'KIS');
});
