import os

# Both torch and faiss are imported in one pytest process; see echostream/__init__.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import faiss  # noqa: E402,F401  -- must load before torch
