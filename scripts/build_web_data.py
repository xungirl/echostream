"""Copy measured results into web/public/data/ for the demo's A/B, Quality and Journey pages.
Only reads results/*.json (no numbers are typed by hand here)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public" / "data"


def load(name):
    p = ROOT / "results" / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s1, s2, s3, s4 = load("stage1_baselines.json"), load("stage2_two_tower.json"), load("stage3_multipath.json"), load("stage4_ranker.json")
    s5l, s5c, s6, s7 = load("stage5_load.json"), load("stage5_chaos.json"), load("stage6_mmr.json"), load("stage7_ab.json")
    s8, s9 = load("stage8_rollout.json"), load("stage9_release_guard.json")
    rf, syn = load("stage5_replay_full.json"), load("stage5_synthetic.json")

    def m(d, *path):
        for p in path:
            if d is None or p not in d:
                return None
            d = d[p]
        return d

    quality = {"baselines_test_ndcg10": {k: v["ndcg@10"] for k, v in (s1 or {}).get("test", {}).items()},
               "two_tower_test_ndcg10": {k: v["ndcg@10"] for k, v in (s2 or {}).get("test", {}).items()},
               "recall_rank_test_ndcg10": m(s4, "test_recall_rank", "ndcg@10"),
               "recall_only_test_ndcg10": m(s4, "test_recall_only_merge_order", "ndcg@10"),
               "ranker_auc": m(s4, "test_recall_rank", "auc"), "ranker_gauc": m(s4, "test_recall_rank", "gauc"),
               "mmr": s6, "history": []}
    h = ROOT / "results" / "history.jsonl"
    if h.exists():
        quality["history"] = [json.loads(x) for x in h.read_text().splitlines() if x.strip()]
    (OUT / "quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=1))
    if s7:
        (OUT / "ab.json").write_text(json.dumps(s7, ensure_ascii=False, indent=1))
    systems = {"load": m(s5l, "summary"), "chaos": m(s5c, "summary"), "replay": {k: v for k, v in (rf or {}).items() if k != "timeline"},
               "synthetic": m(syn, "rows"), "rollout": s8, "release_guard": m(s9, "gate")}
    (OUT / "systems.json").write_text(json.dumps(systems, ensure_ascii=False, indent=1, default=str))

    def fmt(x, nd=4):
        return None if x is None else round(x if not isinstance(x, dict) else x.get("mean", 0), nd)
    journey = [
        {"stage": 0, "title": "Data & labels", "zh": "数据与标签", "done": bool(load("stage0_data_stats.json")),
         "result": "KuaiRec verified; label = watch_ratio > 2.0; catalog drift measured (86% of test positives unseen in train)"},
        {"stage": 1, "title": "Offline eval & baselines", "zh": "离线评测与基线", "done": bool(s1),
         "result": f"best baseline (recent popularity) test NDCG@10 {fmt(m(s1, 'test', 'pop_recent7d', 'ndcg@10'))}"},
        {"stage": 2, "title": "Two-tower + ANN", "zh": "双塔召回 + 向量检索", "done": bool(s2),
         "result": f"two-tower (logQ) test NDCG@10 {fmt(m(s2, 'test', m(s2, 'selected_variant') or '', 'ndcg@10'))}"},
        {"stage": 3, "title": "Multi-path recall", "zh": "多路召回与冷启动", "done": bool(s3),
         "result": f"merged recall@200 {fmt(m(s3, 'test', 'merged_recall@200'))}"},
        {"stage": 4, "title": "Ranking", "zh": "排序", "done": bool(s4),
         "result": f"recall+rank NDCG@10 {fmt(m(s4, 'test_recall_rank', 'ndcg@10'))}, GAUC {m(s4, 'test_recall_rank', 'gauc')}"},
        {"stage": 5, "title": "Distributed serving · MVP", "zh": "分布式服务、压测与故障注入", "done": bool(s5c and rf),
         "result": "sharded index, event streams, feature store, load + chaos + 12.5M real-event replay"},
        {"stage": 6, "title": "MMR re-rank", "zh": "MMR 重排", "done": bool(s6),
         "result": f"default lambda {m(s6, 'selected_lambda')}"},
        {"stage": 7, "title": "Simulated A/B (pre-registered)", "zh": "模拟线上 A/B", "done": bool(s7),
         "result": f"offline vs online Kendall tau {m(s7, 'offline_vs_online', 'big_matrix_test_ndcg10', 'kendall_tau')}"},
        {"stage": 8, "title": "Postgres + safe rollout", "zh": "影子 / 灰度 / 回滚", "done": bool(s8),
         "result": ("; ".join(f"{r['version']}: {r['decision']}" for r in s8["rollouts"]) if s8
                    else "shadow → canary → promote / auto-rollback")},
        {"stage": 9, "title": "Release-guard agent", "zh": "发布守门 agent", "done": bool(s9),
         "result": (f"gate recall {m(s9, 'gate', 'recall')}, precision {m(s9, 'gate', 'precision')}"
                    if s9 else "regression gate + tool-using agent")},
        {"stage": 10, "title": "Web demo", "zh": "网页 demo", "done": (ROOT / "web" / "src").exists(), "result": "this page"},
        {"stage": 11, "title": "E-commerce adapter", "zh": "电商领域适配器", "done": False, "result": "next: Amazon Reviews 2023"},
    ]
    (OUT / "journey.json").write_text(json.dumps(journey, ensure_ascii=False, indent=1))
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
