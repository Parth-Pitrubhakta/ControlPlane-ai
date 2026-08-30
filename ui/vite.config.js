import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The gateway runs on 8080. Proxying keeps the browser on one origin, so the
// dashboard works the same whether it is served by vite or built and hosted.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:8080", changeOrigin: true },
      "/health": { target: "http://127.0.0.1:8080", changeOrigin: true },
      "/v1": { target: "http://127.0.0.1:8080", changeOrigin: true },
    },
  },
});
