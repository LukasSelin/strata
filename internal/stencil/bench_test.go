package stencil

import (
	"math"
	"math/rand/v2"
	"testing"
)

// BenchmarkSlopeRow compares ways to get slope in degrees for one
// 4096-cell row (in cache, compute-bound), to justify the vector
// arctangent:
//
//   - magnitude: Horn magnitude only (percent), the lower bound;
//   - atan: magnitude and Atan32 in one pass, as Slope does;
//   - magnitude+mathatan: the backend's magnitude, then a scalar pass of
//     float32(math.Atan(float64(m)));
//   - magnitude+atan32: the backend's magnitude, then a scalar Atan32 pass.
//
// Run it both ways to compare backends:
//
//	go test -run - -bench SlopeRow ./internal/stencil
//	GOEXPERIMENT=simd go test -run - -bench SlopeRow ./internal/stencil
func BenchmarkSlopeRow(b *testing.B) {
	const n = 4096
	rng := rand.New(rand.NewPCG(7, 8))
	rows := make([][]float32, 3)
	for k := range rows {
		rows[k] = make([]float32, n+2)
		for i := range rows[k] {
			rows[k][i] = float32(800 + 20*rng.NormFloat64())
		}
	}
	dst := make([]float32, n)
	kx, ky := HornScales(10, 10, 1)
	deg := float32(180 / math.Pi)
	report := func(b *testing.B) {
		b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(n)/float64(b.N), "ns/cell")
	}
	b.Logf("backend: %s", Backend())
	b.Run("magnitude", func(b *testing.B) {
		for b.Loop() {
			HornSlopeRow(dst, rows[0], rows[1], rows[2], kx, ky, 100, false)
		}
		report(b)
	})
	b.Run("atan", func(b *testing.B) {
		for b.Loop() {
			HornSlopeRow(dst, rows[0], rows[1], rows[2], kx, ky, deg, true)
		}
		report(b)
	})
	b.Run("magnitude+mathatan", func(b *testing.B) {
		for b.Loop() {
			HornSlopeRow(dst, rows[0], rows[1], rows[2], kx, ky, 1, false)
			for i, m := range dst {
				dst[i] = float32(float32(math.Atan(float64(m))) * deg)
			}
		}
		report(b)
	})
	b.Run("magnitude+atan32", func(b *testing.B) {
		for b.Loop() {
			HornSlopeRow(dst, rows[0], rows[1], rows[2], kx, ky, 1, false)
			for i, m := range dst {
				dst[i] = float32(Atan32(m) * deg)
			}
		}
		report(b)
	})
}
