"""Machine-translate the demo's Chinese video titles to English (Stage 13b).

The browser demo is shown to English-speaking readers, so each KuaiRec caption gets
an English version. Titles are translated once with Qwen (DashScope, qwen3.8-flash)
and cached in echostream/browser/titles_en_cache.json (Chinese -> English), which is
committed: rebuilding the demo never calls the API again.

    set -a; . ~/flowbeat/backend/.env; set +a      # DASHSCOPE_API_KEY, read at runtime only
    uv run python scripts/translate_titles.py       # translate what is missing (resumable)
    uv run python scripts/translate_titles.py --write-only   # just rebuild titles_en.json

Cost: every call goes through the same CostLedger as the release-guard agent, with
its own file (results/translate_cost_ledger.jsonl) and a hard cap (default ¥3).
The ledger prices are conservative placeholders; the DashScope bill is the truth.

Output: web/demo-public/model/titles_en.json, a list aligned with catalog.json
"titles" (null where the caption is empty). The page labels these as machine-translated
and keeps the original Chinese one click away.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import httpx

from echostream.agent.providers import BudgetExceeded, CostLedger

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "web/demo-public/model/catalog.json"
CACHE = ROOT / "echostream/browser/titles_en_cache.json"
OUT = ROOT / "web/demo-public/model/titles_en.json"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen3.8-flash"

SYSTEM = (
    "You translate captions of Chinese short videos (Kuaishou) into natural, concise English for an "
    "American reader. Rules: translate meaning, not word by word; keep it about as short as the original; "
    "keep emojis; translate hashtags into short English hashtags in CamelCase (e.g. #美食 -> #Food); drop "
    "platform boilerplate hashtags such as #我要上热门 / #作品推广 / #快手创作者中心 only if the caption "
    "has other content, otherwise translate them; write Chinese person and place names in pinyin; never add "
    "information that is not in the original. Input is a JSON object {id: caption}. Reply with only a JSON "
    "object {id: translation} with exactly the same ids."
)


VERIFY = (
    "You check machine translations. Input is a JSON object {id: [chinese_caption, english_text]}. For each id, "
    "decide whether english_text is a translation of THAT chinese_caption (dropping boilerplate hashtags or "
    "@mentions is fine; a loose or shortened translation is fine). Flag it only if it is about something else, "
    "translates a different caption, or invents content that is not in the caption. Reply with only a JSON "
    'object {"bad": [ids]}.'
)
VERIFY_LOG = ROOT / "results" / "translate_verify.json"


def post(key: str, ledger: CostLedger, lock: Lock, system: str, user: str, max_out: int) -> str:
    with lock:
        ledger.check(len(user) + len(system) // 2 + 200, max_out)
    r = httpx.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=120, json={
        "model": MODEL, "temperature": 0, "max_tokens": max_out, "enable_thinking": False,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    r.raise_for_status()
    body = r.json()
    with lock:
        ledger.charge(MODEL, body.get("usage", {}))
    return re.sub(r"^```(?:json)?|```$", "", body["choices"][0]["message"]["content"].strip()).strip()


def verify_batch(key: str, ledger: CostLedger, lock: Lock, batch: list[str], cache: dict[str, str]) -> list[str]:
    """Return the captions in `batch` whose cached English is flagged as not their translation."""
    user = json.dumps({str(i): [t, cache[t]] for i, t in enumerate(batch)}, ensure_ascii=False)
    for attempt in range(4):
        try:
            bad = json.loads(post(key, ledger, lock, VERIFY, user, 300)).get("bad", [])
            return [batch[int(i)] for i in bad if str(i).isdigit() and int(i) < len(batch)]
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError, AttributeError) as e:
            if attempt == 3:
                print(f"verify batch failed: {type(e).__name__}")
                return []
            time.sleep(2 * (attempt + 1))
    return []


def load_cache() -> dict[str, str]:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def save_cache(cache: dict[str, str]) -> None:
    CACHE.write_text(json.dumps(dict(sorted(cache.items())), ensure_ascii=False, indent=0) + "\n")


def translate_batch(key: str, ledger: CostLedger, lock: Lock, batch: list[str]) -> dict[str, str]:
    payload = {str(i): t for i, t in enumerate(batch)}
    user = json.dumps(payload, ensure_ascii=False)
    max_out = 120 + 4 * sum(len(t) for t in batch)
    with lock:
        ledger.check(len(user) + len(SYSTEM) // 2 + 200, max_out)
    for attempt in range(4):
        try:
            r = httpx.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=120, json={
                "model": MODEL, "temperature": 0, "max_tokens": max_out, "enable_thinking": False,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]})
            r.raise_for_status()
            body = r.json()
            with lock:
                ledger.charge(MODEL, body.get("usage", {}))
            text = body["choices"][0]["message"]["content"]
            text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
            out = json.loads(text)
            got = {batch[int(k)]: str(v).strip() for k, v in out.items()
                   if k.isdigit() and int(k) < len(batch) and str(v).strip()}
            return got
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, ValueError) as e:
            if attempt == 3:
                print(f"batch failed after 4 tries: {type(e).__name__}: {str(e)[:120]}")
                return {}
            time.sleep(2 * (attempt + 1))
    return {}


def write_output(titles: list[str], cache: dict[str, str]) -> tuple[int, int]:
    aligned = [cache.get(t) if t else None for t in titles]
    OUT.write_text(json.dumps({
        "note": "English titles are machine-translated from the original Chinese KuaiRec captions "
                f"with {MODEL} (Alibaba Qwen); the originals are in catalog.json.",
        "model": MODEL, "titles_en": aligned}, ensure_ascii=False, separators=(",", ":")) + "\n")
    need = sum(1 for t in titles if t)
    return sum(1 for a in aligned if a), need


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0, help="translate at most N titles (trial run)")
    ap.add_argument("--write-only", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="ask the model to flag English titles that do not translate their caption; "
                         "flagged ones are dropped from the cache and re-translated in batches of 5")
    a = ap.parse_args()

    titles: list[str] = json.loads(CATALOG.read_text())["titles"]
    cache = load_cache()
    if not a.write_only:
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise SystemExit("DASHSCOPE_API_KEY is not set")
        # conservative per-1K prices (¥); override with ECHO_PRICE_IN_PER_1K / ECHO_PRICE_OUT_PER_1K
        ledger = CostLedger(cap_yuan=a.cap,
                            price_in_per_1k=float(os.environ.get("ECHO_PRICE_IN_PER_1K", "0.0005")),
                            price_out_per_1k=float(os.environ.get("ECHO_PRICE_OUT_PER_1K", "0.005")),
                            path=ROOT / "results" / "translate_cost_ledger.jsonl")
        if a.verify:
            pairs = sorted({t for t in titles if t and t in cache})
            vb = [pairs[i:i + 40] for i in range(0, len(pairs), 40)]
            lock0, flagged = Lock(), []
            with ThreadPoolExecutor(a.workers) as pool:
                for f in as_completed([pool.submit(verify_batch, key, ledger, lock0, b, cache) for b in vb]):
                    flagged += f.result()
            VERIFY_LOG.write_text(json.dumps({"checked": len(pairs), "flagged": len(flagged),
                                              "pairs": [[t, cache[t]] for t in sorted(flagged)]},
                                             ensure_ascii=False, indent=1) + "\n")
            print(f"verify: {len(flagged)} of {len(pairs)} flagged -> {VERIFY_LOG.relative_to(ROOT)}; "
                  f"ledger ¥{ledger.spent:.4f}")
            for t in flagged:
                cache.pop(t, None)
            a.batch = 5
        todo = sorted({t for t in titles if t and t not in cache})
        if a.limit:
            todo = todo[: a.limit]
        print(f"{len(todo)} titles to translate; ledger so far ¥{ledger.spent:.4f} of ¥{a.cap}")
        batches = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
        lock, done = Lock(), 0
        try:
            with ThreadPoolExecutor(a.workers) as pool:
                futs = [pool.submit(translate_batch, key, ledger, lock, b) for b in batches]
                for f in as_completed(futs):
                    got = f.result()
                    with lock:
                        cache.update(got)
                        done += 1
                        if done % 10 == 0:
                            save_cache(cache)
                            print(f"  {done}/{len(batches)} batches, {len(cache)} cached, ledger ¥{ledger.spent:.4f}")
        except BudgetExceeded as e:
            print(f"stopped by the cost cap: {e}")
        save_cache(cache)
        print(f"ledger total ¥{ledger.spent:.4f}")
    have, need = write_output(titles, cache)
    print(f"titles_en.json: {have}/{need} non-empty titles translated -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
