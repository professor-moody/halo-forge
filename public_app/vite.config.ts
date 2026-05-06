import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { TanStackRouterVite } from "@tanstack/router-plugin/vite";
import path from "node:path";

// halo-forge public_app: Vite + React 19 + Tanstack Router + Tailwind 4.
// Talks to the FastAPI backend mounted at /api/public on
// halo_forge.public_api.app.create_app() — proxied here in dev so the
// frontend always uses same-origin paths.
export default defineConfig({
  plugins: [
    // Router plugin must run before @vitejs/plugin-react so generated
    // routeTree.gen.ts is in place when the React plugin transforms imports.
    TanStackRouterVite({
      target: "react",
      autoCodeSplitting: true,
      routesDirectory: "./src/routes",
      generatedRouteTree: "./src/routeTree.gen.ts",
    }),
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
