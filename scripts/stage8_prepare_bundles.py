"""Stage 8: build the model versions used in the rollout demo.

  v2        : two-tower retrained with another seed (a normal model refresh)       -> should be promoted
  v3-slow   : same as v2 but a 2,000-tree / 255-leaf ranker (offline-fine, slow)    -> latency SLO rollback
  v4-plain  : two-tower trained WITHOUT logQ and mixed negatives (quality regression)-> quality rollback
"""
import json
import subprocess
import sys

import lightgbm as lgb
import numpy as np

from echostream.domain.kuairec import KuaiRecAdapter
from echostream.offline.phase import ARTIFACTS, load_phase
from echostream.rank.offline import candidates_and_features

QUOTAS = {k: v for k, v in json.load(open("results/stage3_multipath.json"))["selected_quotas"].items() if k != "name"}


def tt(tag, *sets):
    if not (ARTIFACTS / "two_tower" / tag).exists():
        subprocess.run([sys.executable, "-m", "echostream.offline.train_two_tower", "--phase", "test", "--tag", tag,
                        "--set", *sets], check=True)


def bundle(version, two_tower, ranker):
    if not (ARTIFACTS / "serving" / version / "manifest.json").exists():
        subprocess.run([sys.executable, "scripts/export_bundle.py", "--version", version, "--two-tower", two_tower,
                        "--ranker", ranker], check=True)


def big_ranker(path):
    if path.exists():
        return
    a = KuaiRecAdapter()
    _, _, X, y, _, _, _ = candidates_and_features(a, load_phase(a, "rank"), "rank_main", QUOTAS)
    params = {"objective": "binary", "learning_rate": 0.02, "num_leaves": 255, "min_data_in_leaf": 50,
              "verbose": -1, "num_threads": 8, "seed": 0}
    lgb.train(params, lgb.Dataset(X, y), 2000).save_model(str(path))


def main():
    tt("main_seed1", "seed=1")
    tt("main_plain", "logq=false", "mixed_negatives=0")
    big = ARTIFACTS / "ranker" / "big_2000x255.txt"
    big_ranker(big)
    main_ranker = str(ARTIFACTS / "ranker" / "main.txt")
    bundle("v2", "main_seed1", main_ranker)
    bundle("v3-slow", "main_seed1", str(big))
    bundle("v4-plain", "main_plain", main_ranker)
    print("bundles ready")


if __name__ == "__main__":
    main()
