import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const hmrClientPort = process.env.VITE_HMR_CLIENT_PORT
  ? Number(process.env.VITE_HMR_CLIENT_PORT)
  : undefined;

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: "/mathagora/",
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
    },
    hmr: hmrClientPort
      ? {
          clientPort: hmrClientPort,
        }
      : undefined,
  },
});
