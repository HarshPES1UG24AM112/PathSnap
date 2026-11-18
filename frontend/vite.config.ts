import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// host 0.0.0.0 so the evaluator UI is reachable from other machines on the LAN
export default defineConfig({
  plugins: [react()],
  server: { host: "0.0.0.0", port: 5173 },
  preview: { host: "0.0.0.0", port: 5173 },
});
