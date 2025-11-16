import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    host: true, // Listen on all interfaces for devcontainer
    // Improve HMR performance for devcontainer
    hmr: {
      overlay: false,
      clientPort: 3000,
    },
    // Reduce request overhead
    cors: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
    watch: {
      // Use polling for bind mounts in devcontainer (slower but reliable)
      usePolling: true,
      interval: 300,
      // Exclude parent directory and heavy folders
      ignored: [
        '**/node_modules/**',
        '**/.git/**',
        '**/dist/**',
        '**/build/**',
        '**/__pycache__/**',
        '**/pyrit.egg-info/**',
        '**/doc/**',
        '**/tests/**',
        '**/dbdata/**',
        '**/assets/**',
        '../pyrit/**',  // Don't watch Python backend
        '../.devcontainer/**',
        '../docker/**',
      ],
    },
    // Reduce initial page load time
    fs: {
      // Only allow serving files from frontend directory
      strict: true,
      allow: ['.'],
    },
  },
  // Optimize build performance for devcontainer
  optimizeDeps: {
    // Force pre-bundling of large dependencies
    include: [
      'react', 
      'react-dom', 
      'react/jsx-runtime',
      '@fluentui/react-components',
      '@fluentui/react-icons',
      'axios',
    ],
    // Use esbuild for fast transforms
    esbuildOptions: {
      target: 'esnext',
      keepNames: false,
    },
    exclude: [],
  },
  // Reduce CSS-in-JS transform overhead from Griffel (Fluent UI)
  css: {
    devSourcemap: false, // Disable sourcemaps in dev
  },
  build: {
    // Optimize chunk splitting
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'fluent-vendor': ['@fluentui/react-components', '@fluentui/react-icons'],
        },
      },
    },
    chunkSizeWarningLimit: 1000,
    sourcemap: false, // Disable sourcemaps for faster builds
  },
  // Reduce logging noise
  logLevel: 'warn',
})
