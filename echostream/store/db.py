"""Postgres access (psycopg 3). One small module so the rest of the code never writes SQL inline."""
from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from echostream.serving.config import ClusterConfig

SCHEMA = Path(__file__).with_name("schema.sql")


def dsn() -> str:
    port = ClusterConfig().postgres_port if "ECHO_PORT_BASE" not in os.environ else int(os.environ["ECHO_PORT_BASE"]) + 40
    return os.environ.get("ECHO_PG_DSN", f"postgresql://echostream@127.0.0.1:{port}/echostream")


def connect(autocommit: bool = True) -> psycopg.Connection:
    return psycopg.connect(dsn(), autocommit=autocommit, row_factory=dict_row, connect_timeout=3)


async def aconnect() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(dsn(), autocommit=True, row_factory=dict_row, connect_timeout=3)


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA.read_text())


# ---- model registry + rollout state ------------------------------------------------
def register_model(conn, version: str, bundle_path: str, offline_metrics: dict | None = None, notes: str = "") -> None:
    conn.execute("""INSERT INTO model_registry (version, bundle_path, offline_metrics, notes)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (version) DO UPDATE SET bundle_path = EXCLUDED.bundle_path,
                      offline_metrics = EXCLUDED.offline_metrics, notes = EXCLUDED.notes""",
                 (version, bundle_path, json.dumps(offline_metrics or {}), notes))


def set_status(conn, version: str, status: str) -> None:
    conn.execute("UPDATE model_registry SET status = %s WHERE version = %s", (status, version))


def log(conn, version: str, action: str, details: dict | None = None) -> None:
    conn.execute("INSERT INTO rollout_log (version, action, details) VALUES (%s, %s, %s)",
                 (version, action, json.dumps(details or {}, default=str)))


def get_rollout(conn) -> dict | None:
    return conn.execute("SELECT * FROM rollout_state WHERE id = 1").fetchone()


def set_rollout(conn, default: str, canary: str | None = None, canary_percent: int = 0,
                shadow: str | None = None) -> None:
    conn.execute("""INSERT INTO rollout_state (id, default_version, canary_version, canary_percent, shadow_version, updated_at)
                    VALUES (1, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE SET default_version = EXCLUDED.default_version,
                      canary_version = EXCLUDED.canary_version, canary_percent = EXCLUDED.canary_percent,
                      shadow_version = EXCLUDED.shadow_version, updated_at = now()""",
                 (default, canary, canary_percent, shadow))


def bundle_paths(conn, versions: list[str]) -> dict[str, str]:
    rows = conn.execute("SELECT version, bundle_path FROM model_registry WHERE version = ANY(%s)", (versions,)).fetchall()
    return {r["version"]: r["bundle_path"] for r in rows}


# ---- event archive ---------------------------------------------------------------
ARCHIVE_SQL = """INSERT INTO event_archive (event_id, user_id, item_id, action, watch_ratio, ts, cats, partition, stream_id)
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (event_id) DO NOTHING"""


async def archive_events(aconn, rows: list[tuple]) -> None:
    async with aconn.cursor() as cur:
        await cur.executemany(ARCHIVE_SQL, rows)
