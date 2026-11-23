"""Turn a core function into a drill: create branch drill/<name>, replace the
function body with `raise NotImplementedError` (docstring kept), so its tests go red.

    uv run python scripts/make_drill.py --list
    uv run python scripts/make_drill.py ndcg          # then: uv run pytest <tests shown>
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DRILLS = {
    # name: (file, function (Class.method allowed), tests, difficulty)
    "ndcg":          ("echostream/core/metrics.py", "ndcg_at_k", "tests/unit/test_metrics.py", "easy"),
    "recall":        ("echostream/core/metrics.py", "recall_at_k", "tests/unit/test_metrics.py", "easy"),
    "bootstrap":     ("echostream/core/metrics.py", "bootstrap_ci", "tests/unit/test_metrics.py", "easy"),
    "gauc":          ("echostream/core/metrics.py", "gauc", "tests/unit/test_metrics.py", "medium"),
    "split":         ("echostream/core/split.py", "temporal_split", "tests/unit/test_split.py", "easy"),
    "scatter_gather": ("echostream/core/scatter_gather.py", "merge_topk", "tests/unit/test_scatter_gather.py", "easy"),
    "quota_merge":   ("echostream/core/recall_merge.py", "merge_with_quotas", "tests/unit/test_recall_merge.py", "medium"),
    "hash_ring":     ("echostream/core/hashing.py", "ConsistentHashRing.get_node", "tests/unit/test_hashing.py", "medium"),
    "itemcf":        ("echostream/core/itemcf.py", "itemcf_similarity", "tests/unit/test_itemcf.py tests/unit/test_baselines.py", "medium"),
    "mmr":           ("echostream/core/mmr.py", "mmr_rerank", "tests/unit/test_mmr.py", "medium"),
    "logq_loss":     ("echostream/core/losses.py", "in_batch_softmax_loss", "tests/unit/test_losses.py", "hard"),
    "user_tower":    ("echostream/core/two_tower.py", "user_tower_numpy", "tests/unit/test_two_tower.py", "medium"),
    "gate":          ("echostream/core/gate.py", "check_metric", "tests/unit/test_gate.py", "medium"),
    "ann":           ("echostream/core/ann.py", "ann_search", "tests/unit/test_ann.py", "easy"),
}


def find(tree: ast.Module, qualname: str) -> ast.FunctionDef:
    parts = qualname.split(".")
    body = tree.body
    node = None
    for p in parts:
        node = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == p)
        body = node.body
    assert isinstance(node, ast.FunctionDef)
    return node


def stub(path: Path, qualname: str) -> None:
    src = path.read_text()
    fn = find(ast.parse(src), qualname)
    lines = src.splitlines(keepends=True)
    first = fn.body[0]
    has_doc = isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant) \
        and isinstance(first.value.value, str)
    start = (first.end_lineno if has_doc else first.lineno - 1)   # keep the docstring lines
    if not has_doc:
        start = fn.body[0].lineno - 1
    indent = " " * (fn.body[-1].col_offset if not has_doc else first.col_offset)
    new = lines[:start] + [f'{indent}raise NotImplementedError("drill: implement {qualname}")\n'] + lines[fn.end_lineno:]
    path.write_text("".join(new))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("drill", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-branch", action="store_true", help="stub in place (used by tests)")
    args = ap.parse_args()
    if args.list or not args.drill:
        for name, (f, fn, tests, diff) in DRILLS.items():
            print(f"{name:15s} {diff:7s} {f}::{fn}  tests: {tests}")
        return
    if args.drill not in DRILLS:
        sys.exit(f"unknown drill {args.drill}; --list shows all")
    f, fn, tests, _ = DRILLS[args.drill]
    if not args.no_branch:
        if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0:
            sys.exit("working tree has uncommitted changes; commit or stash first")
        subprocess.run(["git", "checkout", "-b", f"drill/{args.drill}"], cwd=ROOT, check=True)
    stub(ROOT / f, fn)
    print(f"stubbed {f}::{fn}\nnow run:  OMP_NUM_THREADS=1 uv run pytest {tests} -q   (should be red)")
    print("when green again:  git checkout main  (the drill branch keeps your attempt)")


if __name__ == "__main__":
    main()
