// Package golden loads the JSON files that scripts/gen_go_golden.py writes
// from the Python implementation (go/worker/testdata). Tests only.
package golden

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Load decodes testdata/<name> into v, failing the test if it is missing.
func Load(t testing.TB, name string, v any) {
	t.Helper()
	_, here, _, _ := runtime.Caller(0)
	path := filepath.Join(filepath.Dir(here), "..", "..", "testdata", name)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("golden file missing (run: uv run python scripts/gen_go_golden.py): %v", err)
	}
	if err := json.Unmarshal(data, v); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
}
