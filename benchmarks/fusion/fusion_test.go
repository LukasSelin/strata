package fusion_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/LukasSelin/strata/benchmarks/internal/suite"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// TestFormsAgree checks that the three forms compute the same raster, bit
// for bit, Data and validity, on every backend, tile shape and worker
// count the benchmarks use, on a size that is not a multiple of the SIMD
// width or of the tiles. A fused form that does not equal its unfused
// form is a bug, whatever its speed (DESIGN.md §52).
func TestFormsAgree(t *testing.T) {
	defer kernels.UseScalar(false)
	const size = 301
	f := newFixture(size)
	// Some factors that break naive reassociation: products that round
	// differently depending on the order, and a NaN.
	f.in[2][0], f.in[3][1] = math.MaxFloat32, float32(math.NaN())
	f.in[0][2], f.in[1][2], f.in[2][2] = 1+1.0/(1<<23), 1+1.0/(1<<22), 1-1.0/(1<<24)

	for _, masked := range []bool{false, true} {
		kernels.UseScalar(true)
		if err := chained(context.Background(), f, masked, engine.Options{}); err != nil {
			t.Fatal(err)
		}
		want := snapshot(f, masked)

		for _, backend := range suite.Backends {
			kernels.UseScalar(backend == suite.Scalar)
			if backend == suite.SIMD && vec.Backend() == suite.Scalar {
				continue
			}
			forms := map[string]form{"chained": chained, "pipeline": pipelined, "fused": fused}
			for name, run := range forms {
				for _, workers := range []int{1, 4} {
					for _, tile := range [][2]int{{0, 0}, {256, 256}, {37, 11}} {
						id := fmt.Sprintf("%s masked=%v backend=%s workers=%d tile=%v",
							name, masked, backend, workers, tile)
						clear(f.dst)
						o := engine.Options{Workers: workers, TileWidth: tile[0], TileHeight: tile[1]}
						if err := run(context.Background(), f, masked, o); err != nil {
							t.Fatalf("%s: %v", id, err)
						}
						same(t, id, snapshot(f, masked), want)
					}
				}
			}
		}
	}
}

type result struct {
	data  []float32
	valid []bool
}

func snapshot(f *fixture, masked bool) result {
	dst, _ := f.operands(masked)
	r := result{data: append([]float32(nil), dst.Data...)}
	if masked {
		r.valid = make([]bool, len(dst.Data))
		for i := range r.valid {
			r.valid[i] = raster.MaskGet(dst.Valid, i)
		}
	}
	return r
}

func same(t *testing.T, id string, got, want result) {
	t.Helper()
	for i := range want.data {
		g, w := got.data[i], want.data[i]
		if math.Float32bits(g) != math.Float32bits(w) && !(g != g && w != w) {
			t.Fatalf("%s: Data[%d] = %v, want %v", id, i, g, w)
		}
		if want.valid != nil && got.valid[i] != want.valid[i] {
			t.Fatalf("%s: validity[%d] = %v, want %v", id, i, got.valid[i], want.valid[i])
		}
	}
}

// TestCounter checks bytesPerCell against engine.Stats for the float32
// traffic it counts: 60 B/cell for the chain, 28 for the pipeline and the
// fused kernel.
func TestCounter(t *testing.T) {
	f := newFixture(300)
	for name, run := range map[string]form{"chained": chained, "pipeline": pipelined, "fused": fused} {
		var s engine.Stats
		if err := run(context.Background(), f, false, engine.Options{Stats: &s}); err != nil {
			t.Fatal(err)
		}
		// The chain's five calls each count their own cells.
		cells := float64(f.size * f.size)
		if got, want := float64(s.Total())/cells, bytesPerCell(name, false); got != want {
			t.Errorf("%s: %v B/cell counted, bytesPerCell says %v", name, got, want)
		}
	}
}

// TestAllocs checks that no form allocates per tile or per band: a bound
// that does not grow with the raster, as in the engine category.
func TestAllocs(t *testing.T) {
	defer kernels.UseScalar(false)
	f := newFixture(300)
	for name, run := range map[string]form{"chained": chained, "pipeline": pipelined, "fused": fused} {
		for _, masked := range []bool{false, true} {
			for _, workers := range []int{1, 4} {
				c := suite.Case{Size: f.size, Masked: masked, Backend: suite.SIMD, Workers: workers, Tiles: tiles256}
				w := workload(name, run)(f, c)
				calls := 1.0
				if name == "chained" {
					calls = 5
				}
				limit := calls * float64(16+2*workers)
				if allocs := testing.AllocsPerRun(5, w.Run); allocs > limit {
					t.Errorf("%s/%s: %v allocs/op, want at most %v", name, c.Name(), allocs, limit)
				}
			}
		}
	}
}
