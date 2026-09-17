package nodata

import (
	"math"
	"runtime"
	"strconv"
	"testing"

	"strata/internal/vec"
)

// These tests pin down representation hazards the recommendation relies
// on. Each asserts the hazardous behaviour actually happens, so a
// "passing" test here means "the hazard is real".

// A legitimate result can equal the sentinel, and is then read as NoData.
func TestHazardSentinelCollisionFromArithmetic(t *testing.T) {
	a := []float32{-10000}
	b := []float32{1}
	dst := make([]float32, 1)
	AddSentinelBranchy(dst, a, b, Sentinel)
	if dst[0] != Sentinel {
		t.Fatalf("expected the valid sum to collide with the sentinel, got %v", dst[0])
	}
}

// A zero sentinel also captures -0, because -0 == +0.
func TestHazardSentinelZeroMatchesNegativeZero(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	if !(negZero == 0) {
		t.Fatal("expected -0 == 0")
	}
}

// Fill values arrive as text/float64 metadata (GDAL_NODATA is an ASCII
// tag; Zarr fill_value is JSON). A value that is not exactly
// representable in float32 never compares equal unless the adapter
// rounds it through float32 exactly the way the writer did.
func TestHazardSentinelFloat64Metadata(t *testing.T) {
	meta, _ := strconv.ParseFloat("1e-9", 64)
	stored := float32(meta) // what a float32 writer put in the pixels
	if float64(stored) == meta {
		t.Fatal("expected 1e-9 to be inexact in float32")
	}
	// FLT_MAX-style fills (ESRI) survive, but only via float32 rounding.
	fltMax, _ := strconv.ParseFloat("-3.4028234663852886e+38", 64)
	if float32(fltMax) != -math.MaxFloat32 {
		t.Fatal("expected -FLT_MAX to round-trip through float32")
	}
}

// A NaN fill value cannot be tested with ==, so sentinel code written as
// v == nd never matches when nd is NaN. Here the only NoData cell is the
// Horn centre, which the stencil never reads, so the output is a
// valid-looking number.
func TestHazardSentinelNaNNeverMatches(t *testing.T) {
	src := make([]float32, 9)
	for i := range src {
		src[i] = 100
	}
	src[4] = NaN32
	dst := make([]float32, 9)
	SlopeSentinelBranchy(dst, src, 3, 3, 10, NaN32)
	if isNaN32(dst[4]) {
		t.Fatal("expected the NaN centre to go undetected")
	}
}

// Computation can create NaN (0/0, sqrt(-1), Inf-Inf). With NaN as
// NoData those results become indistinguishable from missing input.
func TestHazardNaNLegitimateResult(t *testing.T) {
	dst := make([]float32, 2)
	vec.Div(dst, []float32{0, 1}, []float32{0, 1})
	vec.Sqrt(dst[1:], []float32{-1})
	if !isNaN32(dst[0]) || !isNaN32(dst[1]) {
		t.Fatalf("expected computed NaNs, got %v", dst)
	}
}

// NaN propagates only through operations that read the NaN cell. Horn
// ignores the centre cell (see hornNaN), and comparisons, thresholds and
// classification turn NaN into a valid-looking false.
func TestHazardNaNDoesNotPropagateEverywhere(t *testing.T) {
	var z float32 = 100
	// horn has no z5 parameter at all; hornNaN must add a z5*0 term.
	if !isNaN32(hornNaN(z, z, z, z, NaN32, z, z, z, z, 0.1, 0.1)) {
		t.Fatal("hornNaN must propagate a NaN centre")
	}

	// Threshold "steep = slope > 0.5" classifies NoData as not steep.
	steep := b2u(NaN32 > 0.5)
	if steep != 0 {
		t.Fatal("expected NaN > 0.5 to be false")
	}
}

// Integer source rasters (uint8 land cover, int16 SRTM) have no NaN.
// Converting NaN to an integer type in Go is implementation-specific;
// on amd64 it yields the "integer indefinite" value, a valid-looking int.
func TestHazardNaNToInteger(t *testing.T) {
	v := NaN32
	got := int32(v)
	t.Logf("int32(NaN) on this platform = %d", got)
	if runtime.GOARCH == "amd64" && got != math.MinInt32 {
		t.Fatalf("expected integer indefinite on amd64, got %d", got)
	}
}

// vec.Min/Max/Clamp propagate NaN (DESIGN choice mirrored from Go's
// builtin min/max), so with NaN-as-NoData "max(x, 0)" keeps NoData, but
// a user expecting the IEEE fmax "ignore the missing operand" rule gets
// NoData instead. With a mask the data path choice does not matter.
func TestHazardNaNMinMaxSemantics(t *testing.T) {
	dst := make([]float32, 1)
	vec.Max(dst, []float32{NaN32}, []float32{0})
	if !isNaN32(dst[0]) {
		t.Fatalf("expected vec.Max to propagate NaN, got %v", dst[0])
	}
}

// A mask never collides with data: every float32 bit pattern, including
// NaN, -0 and the old sentinel, remains a valid value.
func TestMaskAcceptsEveryValue(t *testing.T) {
	a := []float32{NaN32, float32(math.Copysign(0, -1)), Sentinel, -10000}
	b := []float32{1, 1, 1, 1}
	valid := []uint64{0b1111}
	dst := make([]float32, 4)
	dstValid := make([]uint64, 1)
	AddMaskScalar(dst, a, b, dstValid, valid, valid)
	if dstValid[0] != 0b1111 {
		t.Fatalf("mask changed: %#b", dstValid[0])
	}
	if dst[3] != Sentinel {
		t.Fatalf("expected -10000+1 = -9999 as a valid value, got %v", dst[3])
	}
}
