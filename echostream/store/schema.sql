-- Postgres is the source of truth. Redis only holds derived data (features, caches, streams in flight).
CREATE TABLE IF NOT EXISTS users (
    user_id      INTEGER PRIMARY KEY,
    features     JSONB NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS videos (
    item_id      INTEGER PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT '',
    category     TEXT NOT NULL DEFAULT '',
    cats         TEXT NOT NULL DEFAULT ''          -- comma-separated category ids
);
CREATE TABLE IF NOT EXISTS model_registry (
    version      TEXT PRIMARY KEY,
    bundle_path  TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'registered', -- registered|shadow|canary|live|rolled_back|retired|blocked
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    offline_metrics JSONB NOT NULL DEFAULT '{}',
    notes        TEXT NOT NULL DEFAULT ''
);
-- Exactly one row: the desired rollout state every API replica and shard reconciles to.
CREATE TABLE IF NOT EXISTS rollout_state (
    id             INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    default_version TEXT NOT NULL,
    canary_version  TEXT,
    canary_percent  INTEGER NOT NULL DEFAULT 0,
    shadow_version  TEXT,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS rollout_log (
    id        BIGSERIAL PRIMARY KEY,
    ts        TIMESTAMPTZ NOT NULL DEFAULT now(),
    version   TEXT NOT NULL,
    action    TEXT NOT NULL,
    details   JSONB NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS experiments (
    name      TEXT PRIMARY KEY,
    config    JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Every applied event, written by the workers in batches (idempotent on event_id).
CREATE TABLE IF NOT EXISTS event_archive (
    event_id    TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    item_id     INTEGER NOT NULL,
    action      TEXT NOT NULL,
    watch_ratio REAL NOT NULL,
    ts          DOUBLE PRECISION NOT NULL,
    cats        TEXT NOT NULL DEFAULT '',
    partition   INTEGER NOT NULL,
    stream_id   TEXT NOT NULL,                    -- Redis entry id: order within the partition
    archived_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS event_archive_user ON event_archive (user_id);
