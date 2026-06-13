import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icons/icon-192.png', 'icons/icon-512.png'],
      manifest: {
        name: 'Christmas Light Estimator',
        short_name: 'Light Estimator',
        description: 'AI-assisted Christmas light markup — by Lighting Colorado Christmas',
        theme_color: '#11432e',
        background_color: '#0a2a1c',
        display: 'standalone',
        orientation: 'any',
        start_url: '/',
        icons: [
          { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,png,webmanifest}'],
        maximumFileSizeToCacheInBytes: 6 * 1024 * 1024,
        navigateFallback: '/index.html',
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.endsWith('.onnx') || url.pathname.endsWith('.wasm'),
            handler: 'CacheFirst',
            options: { cacheName: 'ml-assets', cacheableResponse: { statuses: [0, 200] }, expiration: { maxEntries: 6 } },
          },
          {
            urlPattern: ({ url }) => url.origin === 'https://fonts.googleapis.com' || url.origin === 'https://fonts.gstatic.com',
            handler: 'CacheFirst',
            options: { cacheName: 'google-fonts', cacheableResponse: { statuses: [0, 200] }, expiration: { maxEntries: 20 } },
          },
        ],
      },
    }),
  ],
  server: { port: 5173 },
})
