// Package archive writes applied events to Postgres, the source of truth
// (Stage 8). Same SQL as echostream/store/db.py: one INSERT per event, sent as
// one batch (one round trip), idempotent on event_id.
package archive

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"
)

const insertSQL = `INSERT INTO event_archive (event_id, user_id, item_id, action, watch_ratio, ts, cats, partition, stream_id)
VALUES ($1, $2::bigint, $3::bigint, $4, $5::float8, $6::float8, $7, $8::int, $9) ON CONFLICT (event_id) DO NOTHING`

// Row is one archived event.
type Row struct {
	EventID    string
	UserID     int64
	ItemID     int64
	Action     string
	WatchRatio float64
	Ts         float64
	Cats       string
	Partition  int
	StreamID   string
}

// Archiver keeps one lazily opened connection, like the Python worker's self.pg.
type Archiver struct {
	dsn  string
	conn *pgx.Conn
}

// New does not connect yet; the first Write does.
func New(dsn string) *Archiver { return &Archiver{dsn: dsn} }

// Write inserts the rows. On any error the connection is dropped and the
// caller keeps the entries pending, so the archive never misses an event.
func (a *Archiver) Write(ctx context.Context, rows []Row) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if a.conn == nil || a.conn.IsClosed() {
		cfg, err := pgx.ParseConfig(a.dsn)
		if err != nil {
			return err
		}
		cfg.ConnectTimeout = 3 * time.Second
		conn, err := pgx.ConnectConfig(ctx, cfg)
		if err != nil {
			return err
		}
		a.conn = conn
	}
	batch := &pgx.Batch{}
	for _, r := range rows {
		batch.Queue(insertSQL, r.EventID, r.UserID, r.ItemID, r.Action, r.WatchRatio, r.Ts, r.Cats,
			r.Partition, r.StreamID)
	}
	if err := a.conn.SendBatch(ctx, batch).Close(); err != nil {
		_ = a.conn.Close(context.Background())
		a.conn = nil
		return err
	}
	return nil
}

// Close closes the connection if one is open.
func (a *Archiver) Close() {
	if a.conn != nil {
		_ = a.conn.Close(context.Background())
	}
}
