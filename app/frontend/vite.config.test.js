import assert from 'node:assert/strict';
import test from 'node:test';

import viteConfig from './vite.config.js';


test('Vite proxy target removes a trailing API prefix', () => {
  const originalApiUrl = process.env.VITE_API_URL;
  process.env.VITE_API_URL = 'https://search.example/api';

  try {
    const config = viteConfig({ mode: 'test' });
    assert.equal(config.server.proxy['/api'].target, 'https://search.example');
    assert.equal(config.server.proxy['/keyframes'].target, 'https://search.example');
  } finally {
    if (originalApiUrl === undefined) {
      delete process.env.VITE_API_URL;
    } else {
      process.env.VITE_API_URL = originalApiUrl;
    }
  }
});
