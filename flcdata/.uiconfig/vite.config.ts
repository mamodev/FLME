import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path';

import { nodePolyfills} from 'vite-plugin-node-polyfills';

// https://vite.dev/config/
export default defineConfig({
  root: './src',
  plugins: [react(), nodePolyfills({
    globals: {
      Buffer: true,
      process: true,
    },
    protocolImports: true,
  })],

  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
})
