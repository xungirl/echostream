"""A tiny process supervisor for the "multi-node on localhost" cluster.

Why not docker compose? (1) Chaos tests must kill/freeze ONE process and
measure recovery in milliseconds from pytest; with plain processes that is a
signal, with colima it is a VM round trip. (2) Latency numbers should not
include the colima VM's virtual network. (3) Start-up is seconds, not minutes.
It does what compose's `restart: always` does: restart crashed processes with
a short backoff, and it exposes a control API for chaos actions.

Run: python -m echostream.cluster.supervisor --profile full
Control API (127.0.0.1:<port_base+9>): GET /status, POST /kill|stop|start|freeze|thaw|delay/{name}
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException

from echostream.serving.config import CLUSTER, RUN_DIR, ClusterConfig

PY = sys.executable
REDIS = shutil.which("redis-server") or "/opt/homebrew/bin/redis-server"
NGINX = shutil.which("nginx") or "/opt/homebrew/bin/nginx"


@dataclass
class ProcSpec:
    name: str
    cmd: list[str]
    group: str
    env: dict[str, str] = field(default_factory=dict)
    port: int | None = None


@dataclass
class Proc:
    spec: ProcSpec
    popen: subprocess.Popen | None = None
    restarts: int = 0
    held_down: bool = False      # stopped on purpose: do not restart
    frozen: bool = False
    started_at: float = 0.0
    events: list[dict] = field(default_factory=list)


def nginx_conf(cfg: ClusterConfig, run: Path) -> Path:
    d = run / "nginx"
    (d / "tmp").mkdir(parents=True, exist_ok=True)
    servers = "\n".join(f"    server 127.0.0.1:{cfg.api_port(i)} max_fails=1 fail_timeout=2s;" for i in range(cfg.n_api))
    conf = f"""worker_processes 2;
daemon off;
pid {d}/nginx.pid;
error_log {run}/logs/nginx_error.log warn;
events {{ worker_connections 4096; }}
http {{
  access_log off;
  client_body_temp_path {d}/tmp/body; proxy_temp_path {d}/tmp/proxy;
  fastcgi_temp_path {d}/tmp/fcgi; uwsgi_temp_path {d}/tmp/uwsgi; scgi_temp_path {d}/tmp/scgi;
  upstream api {{
{servers}
    keepalive 128;
  }}
  server {{
    listen 127.0.0.1:{cfg.lb_port};
    location / {{
      proxy_pass http://api;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      # a dead replica -> retry the request on the other one (POST /events is idempotent by event_id)
      proxy_next_upstream error timeout http_502 http_503 non_idempotent;
      proxy_next_upstream_tries 2;
      proxy_connect_timeout 200ms;
      proxy_read_timeout 5s;
    }}
  }}
}}
"""
    path = d / "nginx.conf"
    path.write_text(conf)
    return path


def build_specs(cfg: ClusterConfig, profile: str, run: Path, extra_env: dict[str, str] | None = None,
                synthetic: dict | None = None) -> list[ProcSpec]:
    env = {**(extra_env or {})}
    specs: list[ProcSpec] = []
    want = {"full": {"redis", "shards", "workers", "api", "lb"},
            "events": {"redis", "workers"},
            "serving": {"redis", "shards", "api", "lb"},
            "shards": {"shards"}}[profile]
    if "redis" in want:
        ev_dir = run / "redis" / "events"
        ev_dir.mkdir(parents=True, exist_ok=True)
        # the event bus is the durability point of at-least-once delivery -> AOF on
        specs.append(ProcSpec("redis-events", [REDIS, "--port", str(cfg.events_redis_port), "--bind", "127.0.0.1",
                                               "--dir", str(ev_dir), "--appendonly", "yes", "--appendfsync", "everysec",
                                               "--save", "", "--protected-mode", "no"], "redis",
                              port=cfg.events_redis_port))
        for i in range(cfg.n_feature_nodes):
            # feature nodes are a cache: derived data, rebuildable (Stage 8) -> no persistence
            specs.append(ProcSpec(f"redis-feat{i}", [REDIS, "--port", str(cfg.feature_redis_port(i)), "--bind",
                                                    "127.0.0.1", "--save", "", "--appendonly", "no",
                                                    "--protected-mode", "no"], "redis",
                                  port=cfg.feature_redis_port(i)))
    if "shards" in want:
        for s in range(cfg.n_shards):
            for r in range(cfg.n_replicas):
                cmd = [PY, "-m", "echostream.serving.shard_server", "--shard", str(s), "--replica", str(r),
                       "--n-shards", str(cfg.n_shards), "--port", str(cfg.shard_port(s, r))]
                if synthetic:
                    cmd += ["--synthetic", str(synthetic["total"]), "--synthetic-kind", synthetic["kind"],
                            "--synthetic-params", synthetic.get("params", "{}")]
                else:
                    cmd += ["--bundle", cfg.bundle_dir]
                specs.append(ProcSpec(f"shard{s}-r{r}", cmd, "shards", env, port=cfg.shard_port(s, r)))
    if "workers" in want:
        for i in range(cfg.n_workers):
            specs.append(ProcSpec(f"worker{i}", [PY, "-m", "echostream.serving.worker", "--id", f"w{i}",
                                                 "--metrics-port", str(cfg.worker_metrics_port(i))], "workers", env))
    if "api" in want:
        for i in range(cfg.n_api):
            specs.append(ProcSpec(f"api{i}", [PY, "-m", "echostream.serving.api", "--port", str(cfg.api_port(i))],
                                  "api", {**env, "ECHO_REPLICA_NAME": f"api{i}"}, port=cfg.api_port(i)))
    if "lb" in want:
        conf = nginx_conf(cfg, run)
        specs.append(ProcSpec("nginx", [NGINX, "-c", str(conf), "-p", str(run / "nginx")], "lb", port=cfg.lb_port))
    return specs


class Supervisor:
    def __init__(self, specs: list[ProcSpec], run: Path, cfg: ClusterConfig):
        self.procs = {s.name: Proc(s) for s in specs}
        self.run = run
        self.cfg = cfg
        (run / "logs").mkdir(parents=True, exist_ok=True)
        self.stopping = False

    def _start(self, p: Proc) -> None:
        log = open(self.run / "logs" / f"{p.spec.name}.log", "ab")
        env = {**os.environ, **p.spec.env}
        p.popen = subprocess.Popen(p.spec.cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
        p.started_at = time.time()
        p.frozen = False
        p.events.append({"t": time.time(), "event": "start", "pid": p.popen.pid})

    def start_all(self) -> None:
        for p in self.procs.values():
            self._start(p)

    async def monitor(self) -> None:
        while not self.stopping:
            for p in self.procs.values():
                if p.popen and p.popen.poll() is not None and not p.held_down and not self.stopping:
                    p.events.append({"t": time.time(), "event": "exit", "code": p.popen.returncode})
                    await asyncio.sleep(0.3)  # restart backoff
                    p.restarts += 1
                    self._start(p)
            await asyncio.sleep(0.1)

    def signal(self, name: str, sig: int) -> int:
        p = self.procs[name]
        if not p.popen or p.popen.poll() is not None:
            raise HTTPException(409, f"{name} is not running")
        os.kill(p.popen.pid, sig)
        p.events.append({"t": time.time(), "event": f"signal {sig}"})
        return p.popen.pid

    def status(self) -> dict:
        return {name: {"pid": p.popen.pid if p.popen else None, "group": p.spec.group, "port": p.spec.port,
                       "running": bool(p.popen and p.popen.poll() is None), "restarts": p.restarts,
                       "held_down": p.held_down, "frozen": p.frozen,
                       "uptime_s": round(time.time() - p.started_at, 1) if p.popen else 0}
                for name, p in self.procs.items()}

    def shutdown(self) -> None:
        self.stopping = True
        for p in self.procs.values():
            if p.popen and p.popen.poll() is None:
                if p.frozen:
                    os.kill(p.popen.pid, signal.SIGCONT)
                p.popen.terminate()
        deadline = time.time() + 8
        for p in self.procs.values():
            if p.popen:
                try:
                    p.popen.wait(timeout=max(0.1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    p.popen.kill()


def control_app(sup: Supervisor) -> FastAPI:
    app = FastAPI()

    def get(name: str) -> Proc:
        if name not in sup.procs:
            raise HTTPException(404, f"no process {name}")
        return sup.procs[name]

    @app.get("/status")
    async def status():
        return sup.status()

    @app.post("/kill/{name}")
    async def kill(name: str):
        """SIGKILL; the supervisor restarts it (crash + restart)."""
        get(name)
        return {"killed": name, "pid": sup.signal(name, signal.SIGKILL), "t": time.time()}

    @app.post("/stop/{name}")
    async def stop(name: str):
        """SIGKILL and keep it down until /start."""
        p = get(name)
        p.held_down = True
        return {"stopped": name, "pid": sup.signal(name, signal.SIGKILL), "t": time.time()}

    @app.post("/start/{name}")
    async def start(name: str):
        p = get(name)
        p.held_down = False
        if p.frozen:
            sup.signal(name, signal.SIGCONT)
            p.frozen = False
        elif not p.popen or p.popen.poll() is not None:
            sup._start(p)
        return {"started": name, "t": time.time()}

    @app.post("/freeze/{name}")
    async def freeze(name: str):
        """SIGSTOP: the process hangs (open sockets, no answers) -- like a stalled host."""
        p = get(name)
        sup.signal(name, signal.SIGSTOP)
        p.frozen = True
        return {"frozen": name, "t": time.time()}

    @app.post("/thaw/{name}")
    async def thaw(name: str):
        p = get(name)
        sup.signal(name, signal.SIGCONT)
        p.frozen = False
        return {"thawed": name, "t": time.time()}

    @app.post("/delay/{name}")
    async def delay(name: str, ms: float = 300.0):
        p = get(name)
        if p.spec.group != "shards":
            raise HTTPException(400, "delay only applies to shard servers")
        async with httpx.AsyncClient() as c:
            r = await c.post(f"http://127.0.0.1:{p.spec.port}/admin/delay", params={"ms": ms}, timeout=2.0)
        return {"delayed": name, **r.json(), "t": time.time()}

    return app


async def amain(profile: str, synthetic: dict | None) -> None:
    run = RUN_DIR
    (run / "logs").mkdir(parents=True, exist_ok=True)
    sup = Supervisor(build_specs(CLUSTER, profile, run, synthetic=synthetic), run, CLUSTER)
    sup.start_all()
    (run / "supervisor.pid").write_text(str(os.getpid()))
    server = uvicorn.Server(uvicorn.Config(control_app(sup), host="127.0.0.1", port=CLUSTER.supervisor_port,
                                           log_level="warning"))
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    loop.add_signal_handler(signal.SIGINT, stop.set)
    tasks = [asyncio.create_task(sup.monitor()), asyncio.create_task(server.serve())]
    await stop.wait()
    sup.shutdown()
    server.should_exit = True
    await asyncio.sleep(0.3)
    for t in tasks:
        t.cancel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full", choices=["full", "events", "serving", "shards"])
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--synthetic-kind", default="ivf")
    ap.add_argument("--synthetic-params", default="{}")
    args = ap.parse_args()
    syn = {"total": args.synthetic, "kind": args.synthetic_kind, "params": args.synthetic_params} if args.synthetic else None
    asyncio.run(amain(args.profile, syn))


if __name__ == "__main__":
    main()
