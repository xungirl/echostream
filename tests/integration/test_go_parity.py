"""Python worker vs Go worker on the same events (duplicates, re-sends, poison,
ghost-consumer takeover, kill -9 mid-batch): feature store, DLQ, counters and
Postgres archive must be identical. See scripts/go_parity.py."""
import sys
from pathlib import Path

import pytest

from tests.integration.conftest import build_go_worker

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import go_parity  # noqa: E402

pytestmark = pytest.mark.integration


def test_python_and_go_workers_produce_identical_state():
    build_go_worker()
    s = go_parity.parity(("python", "go"), base=26000, log=lambda *a: None)
    d = s["diff"]
    assert s["ground_truth_ok"], {i: s[i]["users_mismatched_vs_ground_truth"] for i in ("python", "go")}
    assert d["feature_keys"]["identical"], d["feature_keys"]
    assert d["dlq"]["identical"], d["dlq"]
    assert all(c["identical"] for c in d["counters"].values()), d["counters"]
    if s["archive_checked"]:
        assert d["archive"]["identical"], d["archive"]
    assert s["identical"]
    assert s["feature_keys_compared"] > 10_000 and s["dlq_entries_compared"] >= 20
    for impl in ("python", "go"):
        assert s[impl]["pel_empty"]
        assert s[impl]["after_graceful_shutdown"] == {"leases_held": 0, "w0_alive": False}
        assert s[impl]["killed_with_pending"] > 0          # the kill really interrupted a batch
