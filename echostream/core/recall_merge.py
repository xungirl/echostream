"""Multi-path recall merge with per-path quotas and de-duplication.

Each recall path (two-tower ANN, ItemCF, popularity, content) returns its own
ranked list. We take up to `quota[path]` items from each path in priority
order, skip duplicates and items the user already saw, then fill any unused
slots round-robin from leftovers so the total is reached whenever possible.
Every merged item remembers ALL paths that proposed it (useful for analysis and
as ranker features).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    item: int
    source: str                     # the path that claimed it first
    score: float                    # that path's score
    all_sources: dict[str, float] = field(default_factory=dict)  # path -> score


def merge_with_quotas(
    paths: dict[str, list[tuple[int, float]]],
    quotas: dict[str, int],
    total: int,
    exclude: set[int] | frozenset[int] = frozenset(),
) -> list[Candidate]:
    """`paths` is ordered by priority (dict insertion order)."""
    by_item: dict[int, Candidate] = {}
    merged: list[Candidate] = []
    cursors = {name: 0 for name in paths}

    # Record every path's opinion about every item (for features), regardless of quota.
    def note(name: str, item: int, score: float) -> None:
        if item in by_item:
            by_item[item].all_sources.setdefault(name, score)

    def take_from(name: str, limit: int) -> int:
        taken = 0
        lst = paths[name]
        while cursors[name] < len(lst) and taken < limit and len(merged) < total:
            item, score = lst[cursors[name]]
            cursors[name] += 1
            if item in exclude:
                continue
            if item in by_item:
                note(name, item, score)
                continue
            cand = Candidate(item, name, float(score), {name: float(score)})
            by_item[item] = cand
            merged.append(cand)
            taken += 1
        return taken

    for name in paths:
        take_from(name, quotas.get(name, 0))

    # Fill leftover slots round-robin so a short path does not waste budget.
    progress = True
    while len(merged) < total and progress:
        progress = False
        for name in paths:
            if take_from(name, 1):
                progress = True

    # Annotate remaining overlaps for analysis.
    for name, lst in paths.items():
        for item, score in lst[cursors[name]:]:
            note(name, item, score)
    return merged
