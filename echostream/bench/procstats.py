"""CPU seconds and RSS of named processes during a benchmark (psutil, sampled every 0.5 s)."""
from __future__ import annotations

import threading
import time

import numpy as np
import psutil


class ProcSampler(threading.Thread):
    """add(name, pid) any time; result() -> {name: {cpu_s, rss_mb_max, rss_mb_median}}.
    A restarted process (same name, new pid) keeps the CPU seconds of its predecessor."""

    def __init__(self, period: float = 0.5):
        super().__init__(daemon=True)
        self.period = period
        self.procs: dict[str, psutil.Process] = {}
        self.cpu_done: dict[str, float] = {}     # CPU of earlier incarnations
        self.cpu0: dict[str, float] = {}
        self.cpu_last: dict[str, float] = {}
        self.rss: dict[str, list[int]] = {}
        self.halt = threading.Event()
        self.lock = threading.Lock()

    def add(self, name: str, pid: int) -> None:
        p = psutil.Process(pid)
        t = p.cpu_times()
        with self.lock:
            if name in self.procs:
                self.cpu_done[name] = self.cpu_done.get(name, 0.0) + self.cpu_last[name] - self.cpu0[name]
            self.procs[name] = p
            self.cpu0[name] = self.cpu_last[name] = t.user + t.system
            self.rss.setdefault(name, [])

    def sample(self) -> None:
        with self.lock:
            for name, p in self.procs.items():
                try:
                    t = p.cpu_times()
                    self.cpu_last[name] = t.user + t.system
                    self.rss[name].append(p.memory_info().rss)
                except psutil.Error:
                    pass

    def run(self) -> None:
        while not self.halt.is_set():
            self.sample()
            time.sleep(self.period)

    def stop(self) -> dict:
        self.sample()
        self.halt.set()
        out = {}
        with self.lock:
            for name in self.procs:
                r = self.rss[name] or [0]
                cpu = self.cpu_done.get(name, 0.0) + self.cpu_last[name] - self.cpu0[name]
                out[name] = {"cpu_s": round(cpu, 2), "rss_mb_max": round(max(r) / 2**20, 1),
                             "rss_mb_median": round(float(np.median(r)) / 2**20, 1)}
        return out
