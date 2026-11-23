/**
 * Stage 13 measurements of the built demo in headless Chromium (MEASURE=1 npm run test:e2e):
 *   - bundle size on disk and as transferred (JSON gzipped by the server, model.bin.gz as is)
 *   - time to interactive: navigation start -> first screen rendered with live buttons,
 *     5 fresh page loads from localhost, and 3 with simulated 20 Mbit/s, 40 ms latency
 *   - per-screen compute: >= 100 recomputes per viewer (each after one reaction), in the page
 * Writes ../results/stage13_browser_demo.json.
 */
import { expect, test, type Page } from "@playwright/test";
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { cpus, platform, release, totalmem } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
import { gzipSync } from "node:zlib";

test.skip(process.env.MEASURE !== "1", "set MEASURE=1 to run the measurements");

const MODEL = resolve(HERE, "../demo-public/model");
const OUT = resolve(HERE, "../../results/stage13_browser_demo.json");
const RECOMPUTES = 120;

const q = (xs: number[], p: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
};
const summary = (xs: number[]) => ({ n: xs.length, median: +q(xs, 0.5).toFixed(2), p95: +q(xs, 0.95).toFixed(2), min: +Math.min(...xs).toFixed(2), max: +Math.max(...xs).toFixed(2) });

async function loadOnce(page: Page) {
  await page.goto("./");
  await page.waitForFunction(() => window.__echostream?.ttiMs !== undefined, null, { timeout: 120_000 });
  return page.evaluate(() => ({ tti: window.__echostream!.ttiMs!, ...window.__echostream!.load! }));
}

test("measure bundle size, time to interactive and per-screen compute", async ({ browser, browserName }) => {
  test.setTimeout(600_000);
  // ---- bundle size
  const files = readdirSync(MODEL).filter((f) => f !== "LICENSE.md");
  const perFile = Object.fromEntries(
    files.map((f) => {
      const raw = statSync(resolve(MODEL, f)).size;
      const transfer = f.endsWith(".gz") ? raw : gzipSync(readFileSync(resolve(MODEL, f)), { level: 9 }).length;
      return [f, { on_disk: raw, transfer }];
    }),
  );
  const manifest = JSON.parse(readFileSync(resolve(MODEL, "manifest.json"), "utf8"));
  const size = {
    on_disk_bytes: Object.values(perFile).reduce((s, x) => s + x.on_disk, 0),
    transfer_bytes: Object.values(perFile).reduce((s, x) => s + x.transfer, 0),
    model_bin_uncompressed_bytes: manifest.model_bytes,
    per_file: perFile,
    quant: manifest.quant.name,
  };

  // ---- time to interactive, localhost (fresh context = empty cache)
  const local = [];
  for (let k = 0; k < 5; k++) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    local.push(await loadOnce(await ctx.newPage()));
    await ctx.close();
  }
  // ---- time to interactive with a simulated network (Chromium DevTools throttling)
  const net = { downloadThroughput: (20e6 / 8), uploadThroughput: (5e6 / 8), latency: 40 };
  const throttled = [];
  for (let k = 0; k < 3; k++) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    const cdp = await ctx.newCDPSession(page);
    await cdp.send("Network.enable");
    await cdp.send("Network.emulateNetworkConditions", { offline: false, ...net });
    throttled.push(await loadOnce(page));
    await ctx.close();
  }

  // ---- per-screen compute in the page, every viewer
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await ctx.newPage();
  await loadOnce(page);
  const uids: number[] = await page.evaluate(() => [...document.querySelectorAll('[role="radio"]')].map((b) => Number(b.getAttribute("data-testid")!.slice(5))));
  const perUser: Record<string, ReturnType<typeof summary>> = {};
  const all: number[] = [];
  for (const uid of uids) {
    const ms: number[] = await page.evaluate(([n, u]) => window.__echostream!.bench!(n, u), [RECOMPUTES, uid] as const);
    perUser[uid] = summary(ms);
    all.push(...ms);
  }
  await ctx.close();
  expect(all.length).toBeGreaterThanOrEqual(100);

  const out = {
    what: "Stage 13 browser demo: bundle size, time to interactive, per-screen compute (browser time, not server latency)",
    date: new Date().toISOString().slice(0, 10),
    machine: { cpu: cpus()[0]?.model, cores: cpus().length, memory_gb: Math.round(totalmem() / 2 ** 30), os: `${platform()} ${release()}` },
    browser: `${browserName} ${browser.version()} (headless, Playwright)`,
    size,
    tti_ms: {
      definition: "navigation start -> first screen rendered (model downloaded, decoded, first recommendation computed)",
      localhost: { ...summary(local.map((x) => x.tti)), runs: local },
      simulated_20mbps_40ms: { ...summary(throttled.map((x) => x.tti)), runs: throttled, network: net },
    },
    compute_ms: {
      definition: `recommend() for one screen after one reaction: realtime state -> user tower -> 5 recall paths -> merge -> 40 features x 200 -> 210 trees -> MMR; ${RECOMPUTES} per viewer, in the page's main thread`,
      all: summary(all),
      per_user: perUser,
    },
  };
  writeFileSync(OUT, JSON.stringify(out, null, 1) + "\n");
  console.log(JSON.stringify({ size: out.size.transfer_bytes, tti: out.tti_ms.localhost, throttled: out.tti_ms.simulated_20mbps_40ms.median, compute: out.compute_ms.all }, null, 1));
});
