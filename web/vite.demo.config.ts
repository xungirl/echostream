/**
 * The serverless browser demo (Stage 13): a separate static site.
 *   npm run build:demo   -> dist-demo/  (relative asset paths: works from any sub-path,
 *                                        e.g. https://xungirl.com/echostream/demo/)
 *   npm run dev:demo     -> http://127.0.0.1:5174/
 */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  root: "demo",
  base: "./",
  publicDir: "../demo-public",
  plugins: [react()],
  server: { port: 5174 },
  preview: { port: 5174 },
  build: { outDir: "../dist-demo", emptyOutDir: true, target: "es2020" },
});
