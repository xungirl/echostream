import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import Journey, { statusOf } from "../components/Journey";

const items = [
  { stage: 0, title: "Data", done: true, result: "r0" },
  { stage: 1, title: "Eval", done: true, result: "r1" },
  { stage: 2, title: "Serving", done: true, result: "r2" },
  { stage: 3, title: "Next", zh: "下一步", done: false, result: "planned" },
];

describe("Journey timeline", () => {
  it("marks done, todo and the newest done item as current", () => {
    render(<Journey items={items} />);
    expect(document.querySelectorAll(".rm-item.done")).toHaveLength(3);
    expect(document.querySelectorAll(".rm-item.todo")).toHaveLength(1);
    const current = document.querySelectorAll(".rm-item.current");
    expect(current).toHaveLength(1);
    expect(current[0]).toBe(screen.getByTestId("stage-2"));
    expect(screen.getByTestId("stage-2")).toHaveTextContent("latest");
    expect(screen.getByTestId("stage-3")).toHaveClass("todo");
    expect(screen.getByTestId("stage-3")).not.toHaveClass("done");
    expect(document.querySelector(".rm-fill")).not.toBeNull();
  });

  it("picks the last done item even if a later stage is todo before it", () => {
    expect(statusOf([{ stage: 0, title: "a", done: true }, { stage: 1, title: "b", done: false }, { stage: 2, title: "c", done: true }])).toEqual([
      "done",
      "todo",
      "done current",
    ]);
    expect(statusOf([{ stage: 0, title: "a", done: false }])).toEqual(["todo"]);
  });
});
