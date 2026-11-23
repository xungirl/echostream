/**
 * Smoke test of the BUILT demo served from /echostream/demo/: load, pick a viewer, react 3 times,
 * the feed must change, no console errors. SCREENSHOTS=1 also writes docs/screenshots/demo_*.png.
 */
import { expect, test, type Page } from "@playwright/test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

const SHOTS = process.env.SCREENSHOTS === "1";
const shot = (name: string) => resolve(HERE, "../../docs/screenshots", `demo_${name}.png`);

function watchErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("requestfailed", (r) => errors.push(`request failed: ${r.url()}`));
  return errors;
}

const feedIds = (page: Page) => page.locator('[data-testid="feed"] > li').evaluateAll((els) => els.map((e) => e.getAttribute("data-testid")));

async function react(page: Page, card: number, kind: "Skip" | "Watched" | "Like", screenNo: number) {
  await page.locator('[data-testid="feed"] > li').nth(card).getByRole("button", { name: new RegExp(`^${kind} video`) }).click();
  await expect(page.getByTestId("compute-time")).toContainText(`screen ${screenNo}`);
}

test("load, pick a viewer, react three times: the feed changes, no console errors", async ({ page }) => {
  const errors = watchErrors(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("./");
  await expect(page.getByTestId("hint")).toBeVisible({ timeout: 60_000 });
  if (SHOTS) {
    await page.waitForTimeout(500);
    await page.screenshot({ path: shot("hint_1440") });
  }
  await page.getByRole("button", { name: "Got it" }).click();

  await page.getByTestId("user-2191").click(); // Pets · Comedy
  await expect(page.getByTestId("user-2191")).toHaveAttribute("aria-checked", "true");
  const first = await feedIds(page);
  expect(first).toHaveLength(10);

  await react(page, 0, "Like", 2);
  await react(page, 1, "Skip", 3);
  await react(page, 2, "Like", 4);
  const after = await feedIds(page);
  expect(after).toHaveLength(10);
  expect(after).not.toEqual(first);
  await expect(page.getByTestId("echo")).toBeVisible();
  await expect(page.getByTestId("session-line")).toContainText("3 reactions · 2 likes · 1 skip");
  await expect(page.getByTestId("compute-time")).toContainText(/computed in your browser in [\d.]+ ms/);
  if (SHOTS) {
    await page.waitForTimeout(2500); // let the glow settle
    await page.screenshot({ path: shot("1440_light"), fullPage: true });
    await page.emulateMedia({ colorScheme: "dark" });
    await page.waitForTimeout(600); // let colour transitions finish
    await page.screenshot({ path: shot("1440_dark"), fullPage: true });
  }
  expect(errors).toEqual([]);
});

test("phone width (375 px): panels stack and still work", async ({ page }) => {
  const errors = watchErrors(page);
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto("./");
  await page.getByRole("button", { name: "Got it" }).click({ timeout: 60_000 });
  const before = await feedIds(page);
  await react(page, 0, "Like", 2);
  expect(await feedIds(page)).not.toEqual(before);
  await expect(page.getByTestId("phone-echo")).toContainText(/Like → \d+ new/);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0); // no horizontal scroll
  if (SHOTS) {
    await page.waitForTimeout(2500);
    await page.screenshot({ path: shot("375_light") });
    await page.emulateMedia({ colorScheme: "dark" });
    await page.waitForTimeout(600); // let colour transitions finish
    await page.screenshot({ path: shot("375_dark"), fullPage: true });
  }
  expect(errors).toEqual([]);
});

test("keyboard only: Tab to a reaction button and press Enter", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("./");
  await page.getByRole("button", { name: "Got it" }).click({ timeout: 60_000 });
  await page.getByTestId("user-4247").focus();
  let guard = 0;
  while (!(await page.evaluate(() => document.activeElement?.getAttribute("aria-label")?.startsWith("Skip video 1"))) && guard++ < 40) {
    await page.keyboard.press("Tab");
  }
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("compute-time")).toContainText("screen 2");
});
