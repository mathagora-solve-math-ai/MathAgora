import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const hmrClientPort = process.env.VITE_HMR_CLIENT_PORT
  ? Number(process.env.VITE_HMR_CLIENT_PORT)
  : undefined;
const basePath = process.env.VITE_BASE_PATH || "/mathagora/";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: basePath,
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    proxy: {
      "/mathagora/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/mathagora/, ""),
      },
      "/csat_acl2026demo/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/csat_acl2026demo/, ""),
      },
    },
    hmr: hmrClientPort
      ? {
          clientPort: hmrClientPort,
        }
      : undefined,
  },
});
