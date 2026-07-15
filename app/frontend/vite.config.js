import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// The proxy target must track the backend's PORT (app/backend/.env). Hardcoding
// it here meant changing the backend port broke the UI with a silent 502.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_');

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': {
          target: env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
  };
});
