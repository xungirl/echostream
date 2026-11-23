"""Stage 13: the browser demo. Export the serving bundle into a small static
bundle the TypeScript engine in web/src/demo/engine/ loads, and run the real
Python read path in-process to produce the golden file that engine must match."""

# lightgbm must be loaded before scipy in this process: if scipy's OpenMP runtime comes first,
# creating a Booster segfaults on macOS (the same libomp clash as torch vs faiss, README Stage 2).
import lightgbm  # noqa: F401
