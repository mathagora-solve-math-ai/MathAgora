import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const hmrClientPort = process.env.VITE_HMR_CLIENT_PORT
  ? Number(process.env.VITE_HMR_CLIENT_PORT)
  : undefined;

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: "/csat_acl2026demo/",
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    proxy: {
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
