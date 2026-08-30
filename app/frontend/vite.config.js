import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// Target tracks the backend's PORT; hardcoding it broke the UI on a port change.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_');
  const backendTarget = (env.VITE_API_URL || 'http://localhost:8000')
    .replace(/\/api\/?$/, '')
    .replace(/\/$/, '');

  return {
    plugins: [react()],
    server: {
      // Allow ngrok tunnel hosts (Vite blocks unknown Host headers by default).
      host: true,
      allowedHosts: true,
      proxy: {
        '/api': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/keyframes': {
          target: backendTarget,
          changeOrigin: true,
        },
        '/map-keyframes': {
          target: backendTarget,
          changeOrigin: true,
        },
      },
    },
  };
});
