/** Which language video titles are shown in, and how to look one up. */
import { createContext, useContext } from "react";
import type { Model } from "./engine/bundle";

export type TitleLang = "en" | "zh";

export const TitleLangContext = createContext<TitleLang>("en");
export const useTitleLang = () => useContext(TitleLangContext);

export interface Title {
  text: string;
  lang: TitleLang;
  /** The original Chinese caption, shown as a tooltip when the English one is displayed. */
  original: string;
}

/** English (machine-translated) when asked for and available, else the original Chinese caption. */
export function titleOf(model: Model, item: number, want: TitleLang): Title {
  const zh = model.catalog.titles[item] || "";
  const en = model.catalog.titles_en?.[item] || "";
  if (want === "en") {
    if (en) return { text: en, lang: "en", original: zh };
    if (!zh) return { text: "(untitled)", lang: "en", original: "" };
  }
  return { text: zh || "（无标题）", lang: "zh", original: zh };
}

export const hasEnglishTitles = (model: Model) => !!model.catalog.titles_en?.some(Boolean);
