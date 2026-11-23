# echostream — every target runs locally; nothing touches a cloud.
PY := uv run python
export KMP_DUPLICATE_LIB_OK := TRUE

.PHONY: setup data stage0 stage1 stage2 stage3 stage4 bundle train-all cluster-up cluster-down \
        test-unit test-integration test ci load chaos replay replay-paced synthetic demo clean-run

setup:            ## install Python deps (uv) and web deps (npm)
	uv sync
	cd web && npm ci

data:             ## download KuaiRec from Zenodo, verify SHA-256, convert to parquet
	$(PY) -m echostream.data.download

stage0:
	$(PY) scripts/stage0_data_report.py
stage1:
	$(PY) scripts/stage1_baselines.py
stage2:
	$(PY) -m echostream.offline.embed_captions
	$(PY) scripts/stage2_two_tower.py
	$(PY) scripts/stage2_ann_bench.py
stage3:
	$(PY) scripts/stage3_multipath.py
stage4:
	$(PY) scripts/stage4_ranker.py --phase rank
bundle:           ## export the serving bundle (artifacts/serving/v1) and point `current` at it
	$(PY) scripts/export_bundle.py --version v1 --set-current
train-all: stage0 stage1 stage2 stage3 stage4 bundle

cluster-up:       ## start the multi-node-on-localhost cluster in the background (ports 17000+)
	mkdir -p .run && nohup $(PY) -m echostream.cluster.supervisor --profile full > .run/supervisor.log 2>&1 &
	$(PY) -c "from echostream.cluster.client import wait_ready; from echostream.serving.config import CLUSTER; wait_ready(CLUSTER, 'full', 180); print('cluster ready: http://127.0.0.1:17000')"
cluster-down:
	-kill `cat .run/supervisor.pid` 2>/dev/null; sleep 3

test-unit:
	OMP_NUM_THREADS=1 uv run pytest tests/unit -q
test-integration: ## starts/stops its own clusters on ports 27000/28000
	OMP_NUM_THREADS=1 uv run pytest tests/integration -q
test: test-unit test-integration
ci:               ## everything that needs no network and no LLM key
	OMP_NUM_THREADS=1 uv run pytest tests -q
	@if [ -d web/node_modules ]; then cd web && npx vitest run; else echo "web deps missing: run make setup"; fi

load:
	$(PY) scripts/stage5_load.py
chaos:
	$(PY) scripts/stage5_chaos.py
replay:           ## replay all 12.5M real events at max rate + exact correctness check
	$(PY) scripts/stage5_replay.py --rate 0 --tag full
replay-paced:
	$(PY) scripts/stage5_replay.py --rate 10000 --limit 1200000 --tag paced10k
synthetic:
	$(PY) scripts/stage5_synthetic.py

clean-run:
	rm -rf .run

eval:             ## evaluate the current pipeline on the test phase, append to results/history.jsonl
	$(PY) -c "from echostream.agent.eval_run import Candidate, evaluate_candidate; import json; print(json.dumps(evaluate_candidate(Candidate('manual'))['metrics'], indent=1))"
release-guard:    ## Stage 9: inject regressions + harmless changes, gate them, run the agent
	$(PY) scripts/stage9_release_guard.py
agent:            ## agent with the deterministic fake provider (no network)
	$(PY) scripts/agent_run.py --provider fake
agent-real:       ## agent with DashScope qwen3.8-flash; needs DASHSCOPE_API_KEY; hard cap ¥2 (cost ledger)
	@test -n "$$DASHSCOPE_API_KEY" || (echo "set DASHSCOPE_API_KEY first" && exit 1)
	$(PY) scripts/agent_run.py --provider dashscope --cap-yuan 2
