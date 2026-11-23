import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import Cluster, { actionsFor } from "../pages/Cluster";
import { mockFetch, reply } from "./mockFetch";

const cluster = {
  shards: [
    [
      { shard: 0, replica: 0, up: true, ejected: false, delay_ms: 0, versions: ["v1"], n_items: { v1: 2660 }, pid: 1, uptime_s: 3.1 },
      { shard: 0, replica: 1, up: false, url: "http://127.0.0.1:17101" },
    ],
  ],
  feature_nodes: { feat0: true, feat1: true, feat2: false },
  events: {
    up: true, workers_alive: ["w0"], dlq_length: 0,
    partitions: [{ partition: 0, length: 0, pending: 0, lag: 0, owner: "w0" }],
    workers: { w0: { applied: "123", duplicates: "0", partitions: "0", dlq: "0" } },
  },
  api_replica: "api1",
};
const sup = {
  "shard0-r1": { pid: null, group: "shards", port: 17101, running: false, restarts: 3, held_down: true, frozen: false, uptime_s: null },
  "redis-feat2": { pid: 9, group: "features", port: 17202, running: true, restarts: 0, held_down: false, frozen: true, uptime_s: 5 },
  worker0: { pid: 10, group: "workers", port: null, running: true, restarts: 0, held_down: false, frozen: false, uptime_s: 5 },
  api1: { pid: 11, group: "api", port: 17001, running: true, restarts: 0, held_down: false, frozen: false, uptime_s: 5 },
};

describe("Cluster", () => {
  it("renders down replica, down feature node, held-down process, and sends chaos", async () => {
    const fetch = mockFetch((url, init) => {
      if (url.endsWith("/cluster")) return reply(cluster);
      if (url.endsWith("/supervisor/status")) return reply(sup);
      if (url.includes("/chaos/") && init?.method === "POST") return reply({ ok: true });
    });
    render(<Cluster pollMs={60_000} />);

    expect(await screen.findByTestId("replica-0-1")).toHaveTextContent("down");
    expect(screen.getByTestId("replica-0-0")).toHaveTextContent("up");
    expect(screen.getByTestId("feat-feat2")).toHaveTextContent("feat2 down");
    await waitFor(() => expect(screen.getByTestId("proc-shard0-r1")).toHaveTextContent("held down"));
    expect(screen.getByTestId("proc-redis-feat2")).toHaveTextContent("frozen");
    expect(screen.getByText(/ECHO_DEMO_MODE=1/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "kill shard0-r1" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith("http://127.0.0.1:17000/chaos/kill?target=shard0-r1", expect.objectContaining({ method: "POST" })));

    fireEvent.click(within(screen.getByTestId("proc-shard0-r1")).getByRole("button", { name: "slow 500ms shard0-r1" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith("http://127.0.0.1:17000/chaos/delay?target=shard0-r1&ms=500", expect.objectContaining({ method: "POST" })));

    fireEvent.click(screen.getByRole("button", { name: "freeze redis-feat2" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith("http://127.0.0.1:17000/chaos/freeze?target=redis-feat2", expect.objectContaining({ method: "POST" })));
  });

  it("only offers freeze to feature redis and delay to shards", () => {
    expect(actionsFor("worker0").map((a) => a.action)).toEqual(["kill", "stop", "start"]);
    expect(actionsFor("redis-feat1").map((a) => a.action)).toContain("freeze");
    expect(actionsFor("shard1-r0").map((a) => a.ms)).toContain(0);
  });

  it("shows the API error when the chaos endpoint refuses (not demo mode)", async () => {
    mockFetch((url) => {
      if (url.endsWith("/cluster")) return reply(cluster);
      if (url.endsWith("/supervisor/status")) return reply(sup);
      if (url.includes("/chaos/")) return reply({ detail: "chaos actions are only enabled in demo mode" }, 403);
    });
    render(<Cluster pollMs={60_000} />);
    fireEvent.click(await screen.findByRole("button", { name: "stop worker0" }));
    expect(await screen.findByText(/stop worker0 failed: HTTP 403/)).toBeInTheDocument();
  });
});
