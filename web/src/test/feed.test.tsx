import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import Feed from "../pages/Feed";
import { mockFetch, reply } from "./mockFetch";

const item = (id: number, title: string) => ({
  item_id: id, title, category: "cat", source: "two_tower",
  sources: { two_tower: 0.5, itemcf: 0.2 }, recall_score: 0.5, rank_score: 0.04,
});
const rec = (items: ReturnType<typeof item>[], degraded: string[] = []) => ({
  user_id: 42, model_version: "v1", served_by: "api0", items,
  timings_ms: { features: 1, user_vector: 1, recall: 2, merge: 1, rank: 1, total: 6 },
  shards: [], degraded,
});

describe("Feed", () => {
  it("POSTs a watched event, re-fetches, highlights new items and lists dropped ones", async () => {
    let calls = 0;
    const fetch = mockFetch((url, init) => {
      if (url.includes("/recommend?user_id=42&n=10")) {
        calls += 1;
        return calls === 1 ? reply(rec([item(1, "alpha"), item(2, "beta")], ["ann_unavailable", "cache_hit"])) : reply(rec([item(2, "beta"), item(3, "gamma")]));
      }
      if (url.endsWith("/events") && init?.method === "POST") return reply({ accepted: 1, streams: ["events:2"] });
    });

    render(<Feed initialUser={42} refreshDelayMs={0} />);
    const card = (await screen.findByText("alpha")).closest("article")!;
    expect(screen.getByText("ann_unavailable")).toHaveClass("chip", "bad");
    expect(screen.getByText("cache_hit")).toHaveClass("neutral");

    fireEvent.click(within(card).getByText("Watched ▶"));

    await waitFor(() => expect(screen.getByTestId("card-3")).toHaveClass("new"));
    expect(screen.getByTestId("card-2")).not.toHaveClass("new");
    expect(within(screen.getByTestId("gone")).getByText("alpha")).toBeInTheDocument();

    const post = fetch.mock.calls.find(([, init]) => init?.method === "POST")!;
    expect(String(post[0])).toBe("http://127.0.0.1:17000/events");
    const body = JSON.parse(String(post[1]!.body));
    expect(body.events).toHaveLength(1);
    expect(body.events[0]).toMatchObject({ user_id: 42, item_id: 1, watch_ratio: 2.5 });
    expect(body.events[0].event_id).toMatch(/^[^:\s]+$/);

    // order: GET, POST, GET
    const kinds = fetch.mock.calls.map(([u, init]) => `${init?.method ?? "GET"} ${String(u).split("?")[0].split("17000")[1]}`);
    expect(kinds).toEqual(["GET /recommend", "POST /events", "GET /recommend"]);
  });

  it("sends watch_ratio 0.1 for Skip", async () => {
    const fetch = mockFetch((url, init) => {
      if (url.includes("/recommend")) return reply(rec([item(1, "alpha")]));
      if (init?.method === "POST") return reply({ accepted: 1, streams: ["events:0"] });
    });
    render(<Feed initialUser={42} refreshDelayMs={0} />);
    const card = (await screen.findByText("alpha")).closest("article")!;
    fireEvent.click(within(card).getByText("Skip ⏭"));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(3));
    const post = fetch.mock.calls.find(([, init]) => init?.method === "POST")!;
    expect(JSON.parse(String(post[1]!.body)).events[0].watch_ratio).toBe(0.1);
  });
});
