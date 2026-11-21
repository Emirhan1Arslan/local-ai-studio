import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/v1": {
        target: "http://127.0.0.1:8710",
        changeOrigin: true,
        ws: true,
      },
      "/outputs": {
        target: "http://127.0.0.1:8710",
        changeOrigin: true,
      },
      // Kaynak dosyalar (mikrofon kaydı / yüklenen ses): transkript kartındaki
      // oynatıcı buradan okur. Eksikse istek vite'ın SPA fallback'ine düşer ve
      // ses yerine index.html döner — oynatıcı görünür ama çalmaz.
      "/uploads": {
        target: "http://127.0.0.1:8710",
        changeOrigin: true,
      },
    },
  },
});
