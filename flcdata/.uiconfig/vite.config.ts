import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path';

import { nodePolyfills} from 'vite-plugin-node-polyfills';

// https://vite.dev/config/
export default defineConfig({
  root: './src',
  build: {
    outDir: '../.app-dist',
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, '../src/index.html'),
      },
      output: {
        entryFileNames: 'index.js',
        chunkFileNames: 'index.js',
        assetFileNames: '[name][extname]',
      },
      plugins: [
        nodePolyfills({
          globals: {
            Buffer: true,
            process: true,
          },
          protocolImports: true,
        }),
      ],
    },
  },

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
