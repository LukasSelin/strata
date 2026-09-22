package kernelborder_test

import (
	"testing"

	"golang.org/x/tools/go/analysis/analysistest"

	"strata-lint/kernelborder"
)

func TestKernelBorder(t *testing.T) {
	if err := kernelborder.Analyzer.Flags.Set("simd", "fakesimd/archsimd"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = kernelborder.Analyzer.Flags.Set("simd", "simd/archsimd") })

	analysistest.Run(t, analysistest.TestData(), kernelborder.Analyzer,
		"goodkernel", "otherkernel", "badimport", "badapi", "badconc",
		"public", "example/benchmarks/fusion")
}
