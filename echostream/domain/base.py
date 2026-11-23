"""DomainAdapter: everything dataset-specific lives behind this interface.

Serving, evaluation and the simulator only talk to an adapter, so adding a new
domain (e.g. e-commerce with Amazon Reviews 2023: label = rating >= 4 or
purchase, text = product title) means writing one new subclass — not touching
the recall/rank/serving code. KuaiRec is the only implementation today.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitSpec:
    """Inclusive date windows (YYYYMMDD ints) for train / val / test."""
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]


class DomainAdapter(ABC):
    name: str
    n_users: int
    n_items: int

    # ---- interactions + labels -------------------------------------------
    @abstractmethod
    def interactions(self) -> pd.DataFrame:
        """Columns: user_id, item_id, ts (unix seconds), date (YYYYMMDD), value.
        `value` is the domain's raw engagement signal (watch_ratio for video)."""

    @abstractmethod
    def is_positive(self, value: np.ndarray) -> np.ndarray:
        """The label definition: which interactions count as 'the user liked it'."""

    @abstractmethod
    def event_type(self, value: float) -> str:
        """Map a raw engagement value to a serving event type (play/complete/skip)."""

    @abstractmethod
    def split_spec(self) -> SplitSpec: ...

    # ---- side information ---------------------------------------------------
    @abstractmethod
    def item_categories(self) -> dict[int, frozenset[int]]: ...

    @abstractmethod
    def item_text(self) -> dict[int, str]:
        """Text used by the content (cold-start) recall path."""

    @abstractmethod
    def item_display(self) -> pd.DataFrame:
        """item_id, title, category_name -- what the demo UI shows on a card."""

    @abstractmethod
    def user_categorical_features(self) -> tuple[np.ndarray, list[int]]:
        """(matrix [n_users, F] of int codes, vocab size per field)."""

    @abstractmethod
    def item_stat_features(self, cutoff_date: int) -> pd.DataFrame:
        """Per-item statistics computed ONLY from data strictly before cutoff_date."""

    # ---- optional: an unbiased, fully-observed evaluation set ---------------
    def fully_observed(self) -> pd.DataFrame | None:
        return None
