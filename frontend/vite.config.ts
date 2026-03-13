import { createLogger, defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const backendProxyTarget = process.env.PYRIT_BACKEND_PROXY_URL || 'http://127.0.0.1:8000'
const frontendPort = Number(process.env.PYRIT_FRONTEND_PORT || '3000')

// Suppress noisy ECONNREFUSED proxy errors while the backend is starting.
// Without this, Vite logs dozens of "http proxy error" stack traces.
const logger = createLogger()
const originalError = logger.error
let proxyWarned = false
logger.error = (msg, options) => {
  if (typeof msg === 'string' && msg.includes('http proxy error')) {
    if (!proxyWarned) {
      console.log(`[vite] Waiting for backend at ${backendProxyTarget}...`)
      proxyWarned = true
    }
    return
  }
  originalError(msg, options)
}

// https://vitejs.dev/config/
export default defineConfig({
  customLogger: logger,
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: frontendPort,
    host: true, // Listen on all interfaces for devcontainer
    // Improve HMR performance for devcontainer
    hmr: {
      overlay: false,
    },
    // Reduce request overhead
    cors: true,
    proxy: {
      '/api': {
        // Use 127.0.0.1 to avoid Node.js 17+ resolving localhost to IPv6 ::1
        target: backendProxyTarget,
        changeOrigin: true,
        // Return 502 on proxy errors so in-flight requests fail fast
        // instead of hanging until the backend comes up.
        configure: (proxy) => {
          proxy.on('error', (_err, _req, res) => {
            if (res && 'writeHead' in res && !res.headersSent) {
              (res as import('http').ServerResponse).writeHead(502);
              (res as import('http').ServerResponse).end();
            }
          });
        },
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
