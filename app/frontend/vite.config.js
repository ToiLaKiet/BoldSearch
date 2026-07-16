import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// Target tracks the backend's PORT; hardcoding it broke the UI on a port change.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_');

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/api': {
          target: env.VITE_API_URL || 'http://localhost:8000',
          changeOrigin: true,
        },
      },
    },
  };
});
