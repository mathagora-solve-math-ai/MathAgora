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
    hmr: hmrClientPort
      ? {
          clientPort: hmrClientPort,
        }
      : undefined,
  },
});
