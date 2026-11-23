import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import AB from "../pages/AB";
import Quality from "../pages/Quality";
import { mockFetch, reply } from "./mockFetch";

describe("A/B page", () => {
  it('shows "not measured yet" when ab.json is missing', async () => {
    const fetch = mockFetch(() => undefined); // every file -> 404
    render(<AB />);
    expect(await screen.findByText("not measured yet")).toBeInTheDocument();
    expect(String(fetch.mock.calls[0][0])).toBe("/data/ab.json");
  });

  it("renders arms, comparisons and offline/online disagreements from ab.json", async () => {
    mockFetch((url) =>
      url === "/data/ab.json"
        ? reply({
            population: 100,
            arms: { V1_popularity: { n_users: 50, pos_rate: 0.2, ci95: [0.18, 0.22] }, V2_als: { n_users: 50, pos_rate: 0.25, ci95: [0.23, 0.27] } },
            primary: { C1: { compare: "V2_als - V1_popularity", diff: 0.05, ci95: [0.02, 0.08], welch_p: 0.001, holm_p: 0.002, "significant_at_0.05": true } },
            offline_vs_online: {
              big_matrix_test_ndcg10: {
                offline: { V1_popularity: 0.3, V2_als: 0.1 },
                online_pos_rate: { V1_popularity: 0.2, V2_als: 0.25 },
                kendall_tau: -1, spearman_rho: -1, disagreeing_pairs: ["V1_popularity vs V2_als"],
              },
            },
            deviations: [],
          })
        : undefined,
    );
    render(<AB />);
    expect(await screen.findByText("V2_als - V1_popularity")).toBeInTheDocument();
    expect(screen.getByText(/V1_popularity vs V2_als/)).toBeInTheDocument();
    expect(document.querySelectorAll("tr.disagree")).toHaveLength(2);
    expect(screen.getByText("Within-subjects · 组内（每个用户看所有方案）").nextElementSibling).toHaveTextContent("not measured yet");
  });
});

describe("Quality page", () => {
  it('shows "not measured yet" for every missing file', async () => {
    mockFetch(() => undefined);
    render(<Quality />);
    expect((await screen.findAllByText("not measured yet")).length).toBeGreaterThanOrEqual(4);
  });
});
