# echostream — every target runs locally; nothing touches a cloud.
PY := uv run python
GO ?= go
export KMP_DUPLICATE_LIB_OK := TRUE
# which event worker the cluster runs: go (default since Stage 12) | python | mixed
# (e.g. make cluster-up ECHO_WORKER_IMPL=python). Without a Go toolchain the default falls back to python.
HAVE_GO := $(shell command -v $(GO) >/dev/null 2>&1 && echo 1)
export ECHO_WORKER_IMPL ?= $(if $(HAVE_GO),go,python)

.PHONY: setup data stage0 stage1 stage2 stage3 stage4 bundle train-all cluster-up cluster-down \
        test-unit test-integration test ci load chaos replay replay-paced synthetic demo clean-run \
        go-build go-test parity bench-go

setup:            ## install Python deps (uv), web deps (npm) and build the Go worker (if Go is installed)
	uv sync
	cd web && npm ci
	@if [ -n "$(HAVE_GO)" ]; then $(MAKE) go-build; else echo "go not installed: the cluster will use the Python worker"; fi

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
	@if [ "$(ECHO_WORKER_IMPL)" != "python" ]; then $(MAKE) go-build; fi
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
	@if [ -n "$(HAVE_GO)" ]; then $(MAKE) go-test; else echo "go not installed: skipping Go tests"; fi
	OMP_NUM_THREADS=1 uv run pytest tests -q
	@if [ -d web/node_modules ]; then cd web && npx vitest run; else echo "web deps missing: run make setup"; fi

go-build:         ## build the Go event worker -> go/worker/bin/echostream-worker
	cd go/worker && $(GO) build -o bin/echostream-worker ./cmd/echostream-worker
go-test: go-build ## go vet + race-detector tests (3 runs each); worker tests start their own redis-server
	cd go/worker && $(GO) vet ./... && $(GO) test -race -count=3 ./...
parity: go-build  ## same events through the Python and the Go worker -> diff feature store, DLQ, counters
	$(PY) scripts/go_parity.py
bench-go: go-build ## Stage 12: drain sweep 1/2/4/8 x python/go, chaos, API replay, paced latency (~3 h, quiet machine)
	$(PY) scripts/stage12_go_bench.py matrix
	$(PY) scripts/stage12_go_bench.py summarize

load:
	$(PY) scripts/stage5_load.py
chaos:
	$(PY) scripts/stage5_chaos.py
replay:           ## replay all 12.5M real events at max rate + exact correctness check (Stage 5: Python workers)
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

stage6:
	$(PY) scripts/stage6_mmr.py
stage7:           ## pre-registered simulated A/B (PREREG.md)
	$(PY) scripts/stage7_sim_ab.py
stage8:           ## Postgres + shadow/canary/rollback demo + cache + rebuild
	$(PY) scripts/stage8_prepare_bundles.py
	$(PY) scripts/stage8_rollout.py
rebuild-features: ## rebuild the Redis feature store from the Postgres event archive
	$(PY) -m echostream.store.rebuild

web-data:
	$(PY) scripts/build_web_data.py
demo:             ## one command: cluster (demo mode, chaos buttons enabled) + web UI on http://127.0.0.1:5173
	$(PY) scripts/build_web_data.py
	ECHO_DEMO_MODE=1 $(MAKE) cluster-up
	@echo "web UI: http://127.0.0.1:5173  (Ctrl-C, then 'make demo-down')"
	cd web && npm run dev -- --host 127.0.0.1 --port 5173
demo-down: cluster-down
