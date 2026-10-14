import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  root: 'frontend',
  plugins: [react()],
  build: { outDir: 'dist', emptyOutDir: true },
  server: {
    host: '127.0.0.1',
    proxy: { '/youtube': 'http://127.0.0.1:5000', '/feedback': 'http://127.0.0.1:5000', '/resummarize': 'http://127.0.0.1:5000' },
  },
})
