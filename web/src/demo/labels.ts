/** Plain-English names, one-line explanations and colors for the five recall paths. */
import type { PathName } from "./engine/bundle";

export const PATH_INFO: Record<PathName, { name: string; tech: string; what: string; color: string }> = {
  two_tower: {
    name: "Taste match",
    tech: "two-tower model",
    what: "A neural net turns your likes into a taste vector and compares it with all 10,728 videos.",
    color: "#7B61FF",
  },
  itemcf: {
    name: "Liked together",
    tech: "ItemCF",
    what: "People who liked the videos you liked also liked these.",
    color: "#00C2FF",
  },
  pop_recent: {
    name: "Trending",
    tech: "7-day popularity",
    what: "Most liked by everyone in the last 7 days of the dataset.",
    color: "#4597FF",
  },
  pop_platform: {
    name: "Popular on Kuaishou",
    tech: "platform-wide completes",
    what: "Most often watched to the end across the whole app.",
    color: "#D154CD",
  },
  content: {
    name: "New & similar",
    tech: "caption embeddings",
    what: "Brand-new videos (no views yet) whose captions read like your likes.",
    color: "#14B8A6",
  },
};

export type Reaction = "skip" | "watch" | "like";

/** KuaiRec has no like button: the model's "liked" = watched more than twice (watch_ratio > 2). */
export const REACTIONS: { kind: Reaction; label: string; key: string; watchRatio: number; hint: string }[] = [
  { kind: "skip", label: "Skip", key: "s", watchRatio: 0.1, hint: "left after 10% of the video" },
  { kind: "watch", label: "Watched", key: "w", watchRatio: 1.0, hint: "watched once to the end" },
  { kind: "like", label: "Like", key: "l", watchRatio: 3.0, hint: "watched 3 times: what the model counts as a like" },
];
