import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.BACKEND_URL || 'http://backend:8000',  // Use Docker service name in containers
        changeOrigin: true,
      },
      '/health': {
        target: process.env.BACKEND_URL || 'http://backend:8000',
        changeOrigin: true,
      },
      '/metrics': {
        target: process.env.BACKEND_URL || 'http://backend:8000',
        changeOrigin: true,
      },
    },
  },
})
