import numpy as np

from echostream.sim.simulator import Policy, UserState, run_session


class TopByIdPolicy(Policy):
    """Shows the n smallest allowed item ids (deterministic, easy to reason about)."""
    name = "ids"

    def slate(self, st, exclude, n):
        allowed = [i for i in range(10) if i not in set(exclude.tolist())]
        return allowed[:n]


def test_session_never_repeats_and_respects_catalog_and_updates_history():
    not_catalog = np.array([False] * 8 + [True] * 2)       # items 8, 9 are outside the catalog
    responses = {0: 3.0, 1: 0.1, 2: 2.5, 3: 0.5, 4: 2.1, 6: 1.0, 7: 0.2}  # item 5 unobserved -> never shown
    st = UserState(0, np.array([], dtype=np.int64), [])
    shown = run_session(TopByIdPolicy(), st, responses, not_catalog, seen=np.array([7]), slates=3, n=2)
    items = [i for i, _ in shown]
    assert items == [0, 1, 2, 3, 4, 6]                      # no repeats, no 5 (unobserved), 7 (seen), 8/9
    assert st.hist_pos.tolist() == [0, 2, 4]                # positives (wr > 2) appended in order
    assert [w for _, w in st.recent] == [3.0, 0.1, 2.5, 0.5, 2.1, 1.0]
