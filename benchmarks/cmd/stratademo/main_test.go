package main

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

// TestResultsMatch checks that the demo tables in benchmarks/chunked's
// RESULTS.md are this command's rendering of the raw run committed next
// to it.
func TestResultsMatch(t *testing.T) {
	raw, err := os.Open("../../chunked/testdata/demo.txt")
	if err != nil {
		t.Fatal(err)
	}
	defer raw.Close()
	results, err := parseRuns(raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("no run lines in testdata/demo.txt")
	}
	var out bytes.Buffer
	render(&out, results)

	doc, err := os.ReadFile("../../chunked/RESULTS.md")
	if err != nil {
		t.Fatal(err)
	}
	const begin, end = "<!-- stratademo output begin -->\n", "<!-- stratademo output end -->"
	text := strings.ReplaceAll(string(doc), "\r\n", "\n")
	_, rest, ok := strings.Cut(text, begin)
	section, _, ok2 := strings.Cut(rest, end)
	if !ok || !ok2 {
		t.Fatalf("RESULTS.md lacks the %q ... %q markers", strings.TrimSpace(begin), end)
	}
	if section != out.String() {
		t.Errorf("RESULTS.md demo section differs from the rendering of testdata/demo.txt; regenerate with\n"+
			"go run ./benchmarks/cmd/stratademo -render benchmarks/chunked/testdata/demo.txt\n\ngot:\n%s\nwant:\n%s",
			section, out.String())
	}
}

func TestParseTiles(t *testing.T) {
	for _, tc := range []struct {
		shape string
		w, h  int
	}{{"strips256", 20000, 256}, {"1024x1024", 1024, 1024}, {"7x3", 7, 3}} {
		w, h, err := parseTiles(tc.shape, 20000)
		if err != nil || w != tc.w || h != tc.h {
			t.Errorf("parseTiles(%q) = %d, %d, %v; want %d, %d", tc.shape, w, h, err, tc.w, tc.h)
		}
	}
	for _, shape := range []string{"tiles", "strips0", "strips-3", "0x5", "5x0", "-1x4", "x", "3x", "strips"} {
		if _, _, err := parseTiles(shape, 10); err == nil {
			t.Errorf("parseTiles accepted %q", shape)
		}
	}
}

// FuzzParseTiles checks that parseTiles never panics and only returns
// positive tile dimensions.
func FuzzParseTiles(f *testing.F) {
	for _, s := range []string{"strips256", "1024x1024", "0x0", "strips-1", "9223372036854775807x1", "xx"} {
		f.Add(s, 20000)
	}
	f.Fuzz(func(t *testing.T, shape string, size int) {
		if size <= 0 {
			return
		}
		w, h, err := parseTiles(shape, size)
		if err == nil && (w <= 0 || h <= 0) {
			t.Fatalf("parseTiles(%q, %d) = %d, %d", shape, size, w, h)
		}
	})
}
