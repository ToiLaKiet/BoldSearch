import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const frontendRoot = resolve(
  process.env.BOLDSEARCH_FRONTEND_ROOT || resolve(process.cwd(), 'app/frontend'),
);
const outputRoot = resolve(
  process.env.BOLDSEARCH_FRONTEND_DIST || resolve(process.cwd(), 'runtime/frontend-dist'),
);
const reactPluginPath = resolve(
  frontendRoot,
  'node_modules/@vitejs/plugin-react/index.js',
);

function runtimeTransform() {
  return {
    name: 'boldsearch-runtime-transform',
    enforce: 'pre',
    transform(code, id) {
      if (!id.endsWith('/src/App.jsx')) return null;
      let transformed = code;
      const apiBase = "const API_BASE = 'http://0.0.0.0:8000/api';";
      if (transformed.includes(apiBase)) {
        transformed = transformed.replace(apiBase, "const API_BASE = '/api';");
      }
      const replacements = [
        [
          '<img src={thumbSrc} alt={`Frame ${f.frame_id}`} />',
          '<img loading="lazy" decoding="async" src={thumbSrc} alt={`Frame ${f.frame_id}`} />',
        ],
        [
          '<img src={imageSrc} alt={keyframe.title || `Frame ${frameId}`} />',
          '<img loading="lazy" decoding="async" src={imageSrc} alt={keyframe.title || `Frame ${frameId}`} />',
        ],
      ];
      for (const [before, after] of replacements) {
        if (transformed.includes(before)) transformed = transformed.replace(before, after);
      }
      return transformed === code ? null : { code: transformed, map: null };
    },
  };
}

export default async function runtimeViteConfig() {
  const reactModule = await import(pathToFileURL(reactPluginPath).href);
  const react = reactModule.default;
  return {
    root: frontendRoot,
    plugins: [runtimeTransform(), react()],
    build: {
      outDir: outputRoot,
      emptyOutDir: true,
    },
  };
}
