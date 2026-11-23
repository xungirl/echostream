import { describe, expect, it } from "vitest";
import type { Model } from "./engine/bundle";
import { hasEnglishTitles, titleOf } from "./titles";

const model = (titles: string[], titles_en?: (string | null)[] | null) =>
  ({ catalog: { titles, titles_en, category: [], categories: [], category_names_note: "" } }) as unknown as Model;

describe("titleOf", () => {
  const m = model(["你好", "", "无翻译"], ["Hello", null, null]);
  it("shows the English title and keeps the original for the tooltip", () => {
    expect(titleOf(m, 0, "en")).toEqual({ text: "Hello", lang: "en", original: "你好" });
  });
  it("falls back to the Chinese original when no translation exists", () => {
    expect(titleOf(m, 2, "en")).toEqual({ text: "无翻译", lang: "zh", original: "无翻译" });
  });
  it("labels an empty caption in the language asked for", () => {
    expect(titleOf(m, 1, "en").text).toBe("(untitled)");
    expect(titleOf(m, 1, "zh").text).toBe("（无标题）");
  });
  it("shows the original when Chinese is asked for", () => {
    expect(titleOf(m, 0, "zh")).toEqual({ text: "你好", lang: "zh", original: "你好" });
  });
  it("works without the optional titles file", () => {
    const none = model(["你好"], null);
    expect(hasEnglishTitles(none)).toBe(false);
    expect(titleOf(none, 0, "en").text).toBe("你好");
  });
});
