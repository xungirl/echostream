// Command echostream-worker is the Go drop-in replacement for
// `python -m echostream.serving.worker`: same flags, same ECHO_* env vars,
// same Redis keys, same metrics.
//
//	echostream-worker --id w0 --metrics-port 17060
//
// The metrics port also serves /debug/pprof (CPU/heap profiles) on 127.0.0.1.
package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	_ "net/http/pprof" // registers /debug/pprof/* on the default mux
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/xungirl/echostream/go/worker/internal/config"
	"github.com/xungirl/echostream/go/worker/internal/worker"
)

func main() {
	id := flag.String("id", "", "worker id, e.g. w0 (required)")
	metricsPort := flag.Int("metrics-port", 0, "serve /metrics and /debug/pprof on 127.0.0.1:<port> (0 = off)")
	flag.Parse()
	if *id == "" {
		log.Fatal("--id is required")
	}
	cfg, err := config.FromEnv()
	if err != nil {
		log.Fatal(err)
	}
	if *metricsPort != 0 {
		http.Handle("/metrics", promhttp.Handler())
		go func() {
			addr := "127.0.0.1:" + strconv.Itoa(*metricsPort)
			if err := http.ListenAndServe(addr, nil); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Printf("metrics server: %v", err)
			}
		}()
	}

	w := worker.New(*id, cfg)
	defer w.Close()

	// SIGTERM (supervisor) or SIGINT (Ctrl-C) -> graceful shutdown
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGTERM, syscall.SIGINT)
	go func() {
		<-sigs
		w.Shutdown()
	}()

	if err := w.Run(context.Background()); err != nil {
		// exit non-zero so the supervisor restarts the process instead of
		// leaving a silent zombie that holds no leases and processes nothing
		log.Printf("worker crashed: %v", err)
		os.Exit(1)
	}
	// graceful: release leases so others take over immediately
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	w.ReleaseLeases(ctx)
}
