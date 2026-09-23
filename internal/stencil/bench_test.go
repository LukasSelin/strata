package stencil

import (
	"fmt"
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

// BenchmarkRowWidth times one call of each row kernel at widths from a
// lane to a 4096-cell raster's interior, to separate the per-call cost
// from the per-cell cost. The engine calls a row kernel once per output
// row of a tile, so a kernel with a large per-call cost is slower in
// narrow tiles. ns/call - width·ns/cell(4094) estimates the fixed cost.
//
//	GOEXPERIMENT=simd go test -run - -bench RowWidth ./internal/stencil
func BenchmarkRowWidth(b *testing.B) {
	const maxN = 4094
	rng := rand.New(rand.NewPCG(9, 10))
	rows := make([][]float32, 3)
	for k := range rows {
		rows[k] = make([]float32, maxN+2)
		for i := range rows[k] {
			rows[k][i] = float32(800 + 20*rng.NormFloat64())
		}
	}
	dst, dst2 := make([]float32, maxN), make([]float32, maxN)
	kx, ky := HornScales(10, 10, 1)
	deg := float32(180 / math.Pi)
	kp, kq, kr, kt, ks := ZTScales(10, 10, 1)
	ops := []struct {
		name string
		row  func(n int)
	}{
		{"slope-percent", func(n int) { HornSlopeRow(dst[:n], rows[0], rows[1], rows[2], kx, ky, 100, false) }},
		{"slope-degrees", func(n int) { HornSlopeRow(dst[:n], rows[0], rows[1], rows[2], kx, ky, deg, true) }},
		{"aspect", func(n int) { HornAspectRow(dst[:n], rows[0], rows[1], rows[2], kx, ky, -1, false) }},
		{"hillshade", func(n int) { HornHillshadeRow(dst[:n], rows[0], rows[1], rows[2], kx, ky, 180, -127, 127) }},
		{"gradient", func(n int) { HornGradientRow(dst[:n], dst2[:n], rows[0], rows[1], rows[2], kx, ky) }},
		{"curvature-profile", func(n int) { ZTCurvatureRow(dst[:n], rows[0], rows[1], rows[2], kp, kq, kr, kt, ks, CurvProfile) }},
		{"curvature-plan", func(n int) { ZTCurvatureRow(dst[:n], rows[0], rows[1], rows[2], kp, kq, kr, kt, ks, CurvPlan) }},
		{"curvature-mean", func(n int) { ZTCurvatureRow(dst[:n], rows[0], rows[1], rows[2], kp, kq, kr, kt, ks, CurvMean) }},
		{"tri-riley", func(n int) { RuggednessRow(dst[:n], rows[0], rows[1], rows[2], RugTRIRiley) }},
		{"tri-wilson", func(n int) { RuggednessRow(dst[:n], rows[0], rows[1], rows[2], RugTRIWilson) }},
		{"tpi", func(n int) { RuggednessRow(dst[:n], rows[0], rows[1], rows[2], RugTPI) }},
		{"roughness", func(n int) { RuggednessRow(dst[:n], rows[0], rows[1], rows[2], RugRoughness) }},
	}
	b.Logf("backend: %s", Backend())
	for _, op := range ops {
		for _, n := range []int{8, 12, 16, 64, 254, 255, 256, 1022, 4094} {
			b.Run(fmt.Sprintf("op=%s/width=%d", op.name, n), func(b *testing.B) {
				for b.Loop() {
					op.row(n)
				}
				b.ReportMetric(b.Elapsed().Seconds()*1e9/float64(n)/float64(b.N), "ns/cell")
			})
		}
	}
}
