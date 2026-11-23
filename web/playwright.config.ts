/**
 * Smoke test + measurements of the built demo (npm run build:demo first), served from a sub-path
 * like https://xungirl.com/echostream/demo/ by e2e/serve.mjs. Headless Chromium.
 */
import { defineConfig, devices } from "@playwright/test";

const PORT = 4178;

export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  retries: 0,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${PORT}/echostream/demo/`, ...devices["Desktop Chrome"] },
  webServer: { command: `node e2e/serve.mjs ${PORT}`, url: `http://127.0.0.1:${PORT}/echostream/demo/`, reuseExistingServer: false },
});
