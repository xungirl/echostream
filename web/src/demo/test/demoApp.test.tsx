/**
 * UI basics of the browser demo, with the real committed bundle (fetch is served from disk).
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import DemoApp from "../DemoApp";
import { BUNDLE_DIR } from "../engine/testBundle";

function serveBundle() {
  const fetch = vi.fn(async (url: string) => {
    const name = String(url).replace(/^.*\/model\//, "");
    const bytes = readFileSync(resolve(BUNDLE_DIR, name));
    return new Response(bytes);
  });
  vi.stubGlobal("fetch", fetch);
  return fetch;
}

// Node >= 22 defines a global localStorage stub that hides jsdom's; use a plain in-memory Storage
function memoryStorage(): Storage {
  const m = new Map<string, string>();
  return {
    get length() {
      return m.size;
    },
    clear: () => m.clear(),
    getItem: (k) => m.get(k) ?? null,
    key: (i) => [...m.keys()][i] ?? null,
    removeItem: (k) => void m.delete(k),
    setItem: (k, v) => void m.set(k, String(v)),
  };
}

beforeEach(() => {
  const ls = memoryStorage();
  vi.stubGlobal("localStorage", ls);
  Object.defineProperty(window, "localStorage", { value: ls, configurable: true });
});

const cards = () => within(screen.getByTestId("feed")).getAllByRole("listitem");

describe("browser demo", () => {
  it("loads the model, shows the first-visit hint, then a screen of 10 videos", async () => {
    serveBundle();
    render(<DemoApp modelBase="./model/" />);
    expect(screen.getByTestId("loading")).toBeInTheDocument();
    await screen.findByTestId("feed", {}, { timeout: 20000 });
    expect(screen.getByTestId("hint")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Got it"));
    expect(screen.queryByTestId("hint")).toBeNull();
    expect(window.localStorage.getItem("echostream-demo-hint-seen")).toBe("1");
    expect(cards()).toHaveLength(10);
    expect(screen.getByTestId("compute-time").textContent).toMatch(/computed in your browser in [\d.]+ ms/);
    // 6 real viewers + the brand-new user
    expect(screen.getAllByRole("radio")).toHaveLength(7);
    expect(screen.getByRole("heading", { name: "How this screen was made" })).toBeInTheDocument();
    expect(screen.getByTestId("path-two_tower").textContent).toMatch(/200 found/);
  }, 30000);

  it("a Like recomputes the screen: the liked video leaves, new ones glow, the echo explains", async () => {
    window.localStorage.setItem("echostream-demo-hint-seen", "1");
    serveBundle();
    render(<DemoApp modelBase="./model/" />);
    await screen.findByTestId("feed", {}, { timeout: 20000 });
    const before = cards().map((c) => c.getAttribute("data-testid"));
    const first = cards()[0];
    fireEvent.click(within(first).getByRole("button", { name: /^Like video 1/ }));
    await waitFor(() => expect(screen.getByTestId("echo")).toBeInTheDocument(), { timeout: 5000 });
    const after = cards().map((c) => c.getAttribute("data-testid"));
    expect(after).not.toContain(before[0]); // a video you reacted to counts as seen
    const fresh = cards().filter((c) => c.classList.contains("fresh"));
    expect(fresh.length).toBe(after.filter((id) => !before.includes(id)).length);
    expect(fresh.length).toBeGreaterThan(0);
    expect(screen.getByTestId("phone-echo").textContent).toMatch(/Like → \d+ new, \d+ left/);
    expect(screen.getByTestId("session-line").textContent).toMatch(/1 reaction · 1 like · 0 skips/);
  }, 30000);

  it("keyboard: S on a focused video skips it; the brand-new user starts with an empty profile", async () => {
    window.localStorage.setItem("echostream-demo-hint-seen", "1");
    serveBundle();
    render(<DemoApp modelBase="./model/" />);
    await screen.findByTestId("feed", {}, { timeout: 20000 });
    const btn = within(cards()[1]).getByRole("button", { name: /^Watched video 2/ });
    btn.focus();
    fireEvent.keyDown(btn, { key: "s" });
    await waitFor(() => expect(screen.getByTestId("session-line").textContent).toMatch(/1 skip\b/), { timeout: 5000 });

    const newUser = screen.getAllByRole("radio").find((r) => r.textContent?.includes("Brand-new user"))!;
    fireEvent.click(newUser);
    expect(newUser).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText(/No likes yet/)).toBeInTheDocument();
    expect(cards()).toHaveLength(10);
  }, 30000);
});
