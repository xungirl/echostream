package pycompat

import (
	"errors"
	"math"
	"strconv"
	"testing"

	"github.com/xungirl/echostream/go/worker/internal/golden"
)

type numCase struct {
	In        string `json:"in"`
	OK        string `json:"ok"`
	Repr      string `json:"repr"`
	Err       string `json:"err"`
	FitsInt64 bool   `json:"fits_int64"`
}

type pyGolden struct {
	Ints      []numCase `json:"ints"`
	Floats    []numCase `json:"floats"`
	FloatRepr []struct {
		Hex  string `json:"hex"`
		Repr string `json:"repr"`
	} `json:"float_repr"`
	StrRepr []struct {
		In   string `json:"in"`
		Repr string `json:"repr"`
	} `json:"str_repr"`
}

func load(t *testing.T) pyGolden {
	var g pyGolden
	golden.Load(t, "pycompat_golden.json", &g)
	return g
}

func TestParseIntMatchesPython(t *testing.T) {
	for _, c := range load(t).Ints {
		got, err := ParseInt(c.In)
		switch {
		case c.Err != "":
			if err == nil || err.Error() != c.Err {
				t.Errorf("ParseInt(%q) = %d, %v; Python: ValueError %q", c.In, got, err, c.Err)
			}
		case !c.FitsInt64:
			if !errors.Is(err, ErrOutOfRange) {
				t.Errorf("ParseInt(%q): want ErrOutOfRange, got %d, %v", c.In, got, err)
			}
		default:
			if err != nil || strconv.FormatInt(got, 10) != c.OK {
				t.Errorf("ParseInt(%q) = %d, %v; Python: %s", c.In, got, err, c.OK)
			}
		}
	}
}

func TestParseFloatMatchesPython(t *testing.T) {
	for _, c := range load(t).Floats {
		got, err := ParseFloat(c.In)
		if c.Err != "" {
			if err == nil || err.Error() != c.Err {
				t.Errorf("ParseFloat(%q) = %v, %v; Python: ValueError %q", c.In, got, err, c.Err)
			}
			continue
		}
		if err != nil || FormatFloat(got) != c.Repr {
			t.Errorf("ParseFloat(%q) = %s, %v; Python: %s", c.In, FormatFloat(got), err, c.Repr)
		}
	}
}

func TestFormatFloatMatchesPythonRepr(t *testing.T) {
	cases := load(t).FloatRepr
	for _, c := range cases {
		f, err := strconv.ParseFloat(c.Hex, 64) // Python's float.hex() is exact
		if err != nil && c.Hex != "inf" && c.Hex != "-inf" {
			t.Fatalf("hex %q: %v", c.Hex, err)
		}
		if c.Hex == "inf" {
			f = math.Inf(1)
		} else if c.Hex == "-inf" {
			f = math.Inf(-1)
		}
		if got := FormatFloat(f); got != c.Repr {
			t.Errorf("FormatFloat(%s) = %q, Python repr %q", c.Hex, got, c.Repr)
		}
	}
	if FormatFloat(math.NaN()) != "nan" {
		t.Error("nan")
	}
	t.Logf("%d floats formatted exactly like repr()", len(cases))
}

func TestQuoteStringMatchesPythonRepr(t *testing.T) {
	for _, c := range load(t).StrRepr {
		if got := QuoteString(c.In); got != c.Repr {
			t.Errorf("QuoteString(%q) = %s, Python repr %s", c.In, got, c.Repr)
		}
	}
}

func FuzzParseFloatNeverPanics(f *testing.F) {
	for _, s := range []string{"1_0.5", "inf", "١.٥", "1e400", ""} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		_, _ = ParseFloat(s)
		_, _ = ParseInt(s)
	})
}
