import assert from 'node:assert/strict';
import test from 'node:test';

import { apiOrigin, staticMediaUrl } from './staticMedia.js';


test('apiOrigin removes a trailing API prefix', () => {
  assert.equal(apiOrigin('https://search.example/api'), 'https://search.example');
});


test('staticMediaUrl preserves development-relative paths and uses a production origin', () => {
  assert.equal(staticMediaUrl('/keyframes/L21_V001/001.jpg', ''), '/keyframes/L21_V001/001.jpg');
  assert.equal(
    staticMediaUrl('/keyframes/L21_V001/001.jpg', 'https://media.example/'),
    'https://media.example/keyframes/L21_V001/001.jpg',
  );
});
