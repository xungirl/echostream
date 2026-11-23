"""echostream (回响流转) — Every swipe echoes. 每一次划动，都有回响。

macOS note: the faiss and torch wheels each ship their own copy of libomp.
Rule in this repo: a process imports torch OR faiss, not both. Training uses
torch (no faiss); serving and ANN benchmarks use faiss (no torch — the user
tower is exported to numpy). Only the test suite loads both, and it does so
with faiss first + OMP_NUM_THREADS=1 (see tests/conftest.py).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
