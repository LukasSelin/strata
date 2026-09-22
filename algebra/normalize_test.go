package algebra_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/transfer"
)

// refNormalize is Normalize written from its definition: the extremes of
// the valid cells with Go's min and max, then (v - lo) / (hi - lo) per
// cell in float32.
func refNormalize(cells []cell) (want []cell, lo, hi float32) {
	lo, hi = float32(math.Inf(1)), float32(math.Inf(-1))
	n := 0
	for _, c := range cells {
		if c.valid {
			lo, hi = min(lo, c.v), max(hi, c.v)
			n++
		}
	}
	if n == 0 {
		lo, hi = float32(math.NaN()), float32(math.NaN())
	}
	want = make([]cell, len(cells))
	for i, c := range cells {
		want[i] = cell{(c.v - lo) / (hi - lo), c.valid}
	}
	return want, lo, hi
}

// finiteOperand is newOperand with elevation-like data: finite, and far
// enough from zero that a reciprocal-based map would miss its endpoints.
func finiteOperand(rng *rand.Rand, w, h int, lay layout, masked bool) operand {
	op := newOperand(rng, w, h, lay, masked)
	for i := range op.root.Data {
		op.root.Data[i] = float32(1000 + rng.NormFloat64()*50)
	}
	return op
}

func TestNormalizeMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	for _, gen := range []struct {
		name string
		new  func(*rand.Rand, int, int, layout, bool) operand
	}{{"specials", newOperand}, {"finite", finiteOperand}} {
		for _, sz := range sizes {
			w, h := sz[0], sz[1]
			for _, ls := range layouts {
				for _, ld := range layouts {
					for _, masked := range []bool{false, true} {
						for _, inPlace := range []bool{false, true} {
							src := gen.new(rng, w, h, ls, masked)
							dst := src
							if !inPlace {
								dst = newOperand(rng, w, h, ld, masked || rng.IntN(2) == 0)
							}
							want, wlo, whi := refNormalize(snapshot(src.r))
							f := freeze(dst)
							lo, hi := algebra.Normalize(dst.r, src.r)
							func() {
								defer func() {
									if t.Failed() {
										t.Logf("case %s %dx%d src=%v masked=%v dst=%v inPlace=%v",
											gen.name, w, h, ls, masked, ld, inPlace)
									}
								}()
								if !sameFloat(lo, wlo) || !sameFloat(hi, whi) {
									t.Fatalf("bounds = (%v, %v), want (%v, %v)", lo, hi, wlo, whi)
								}
								checkResult(t, dst.r, want)
								f.checkUntouched(t)
							}()
						}
					}
				}
			}
		}
	}
}

// TestNormalizeEndpoints is the property Normalize exists for: whatever
// the offset and spread of the data, the smallest cell maps to +0, the
// largest to exactly 1, and every cell into [0, 1] in the source's order.
func TestNormalizeEndpoints(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 14))
	for i := range 2000 {
		offset := float32(math.Ldexp(rng.NormFloat64(), rng.IntN(60)-20))
		scale := float32(math.Ldexp(1+rng.Float64(), rng.IntN(60)-30))
		n := 1 + rng.IntN(300)
		data := make([]float32, n)
		for j := range data {
			data[j] = offset + scale*float32(rng.NormFloat64())
		}
		src := raster.NewFloat32(n, 1, data)
		dst := raster.NewFloat32Like(src)
		lo, hi := algebra.Normalize(dst, src)
		if lo == hi {
			continue // every cell rounded to one value: the 0/0 case
		}
		order := make([]int, n)
		for j := range order {
			order[j] = j
		}
		slices.SortFunc(order, func(a, b int) int { return cmpFloat(data[a], data[b]) })
		first, last := dst.Data[order[0]], dst.Data[order[n-1]]
		if math.Float32bits(first) != 0 || last != 1 {
			t.Fatalf("draw %d (offset %v, scale %v): min -> %v, max -> %v, want +0 and 1", i, offset, scale, first, last)
		}
		prev := float32(0)
		for _, j := range order {
			v := dst.Data[j]
			if !(v >= prev && v <= 1) {
				t.Fatalf("draw %d: cell %v -> %v after %v; want nondecreasing within [0, 1]", i, data[j], v, prev)
			}
			prev = v
		}
	}
}

func cmpFloat(a, b float32) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	}
	return 0
}

// TestNormalizeBeatsRescaleRange records why Normalize does not write
// through transfer.RescaleRange: on elevations, the affine map can send
// the maximum past 1. Here it lands 32 ulps above.
func TestNormalizeBeatsRescaleRange(t *testing.T) {
	data := []float32{1000.25, 1003.5, 1011.75, 1016}
	src := raster.NewFloat32(len(data), 1, data)
	dst := raster.NewFloat32Like(src)
	lo, hi := algebra.Normalize(dst, src)
	if dst.Data[3] != 1 {
		t.Fatalf("Normalize(max) = %v, want 1", dst.Data[3])
	}
	transfer.RescaleRange(dst, src, lo, hi, 0, 1)
	got := dst.Data[3]
	ulps := int64(math.Float32bits(got)) - int64(math.Float32bits(1))
	t.Logf("RescaleRange(max) = %v, %d ulps from 1", got, ulps)
	if ulps <= 0 {
		t.Errorf("RescaleRange(max) = %v, not above 1; the comment on Normalize overstates the difference", got)
	}
}

func TestNormalizePathsAgree(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewPCG(15, 16))
	var tilings []engine.Options
	for _, tw := range []int{0, 1, 7, 64} {
		for _, th := range []int{0, 1, 5} {
			for _, wk := range []int{1, 3, runtime.GOMAXPROCS(0)} {
				tilings = append(tilings, engine.Options{TileWidth: tw, TileHeight: th, Workers: wk})
			}
		}
	}
	for _, size := range [][2]int{{1, 1}, {9, 4}, {65, 3}, {200, 7}} {
		for _, masked := range []bool{false, true} {
			w, h := size[0], size[1]
			src := finiteOperand(rng, w, h, nestedLayout, masked)
			want := newOperand(rng, w, h, compactLayout, masked)
			wlo, whi := algebra.Normalize(want.r, src.r)
			wantCells := snapshot(want.r)

			for _, opts := range tilings {
				id := fmt.Sprintf("%dx%d masked=%v %+v", w, h, masked, opts)
				got := newOperand(rng, w, h, stridedLayout, masked)
				lo, hi, err := algebra.NormalizeTiled(ctx, got.r, src.r, opts)
				if err != nil {
					t.Fatalf("Tiled %s: %v", id, err)
				}
				if lo != wlo || hi != whi {
					t.Fatalf("Tiled %s: bounds (%v, %v), want (%v, %v)", id, lo, hi, wlo, whi)
				}
				checkResult(t, got.r, wantCells)

				sink := engine.NewMemorySink(newOperand(rng, w, h, compactLayout, masked).r)
				lo, hi, err = algebra.NormalizeChunked(ctx, sink, engine.NewMemorySource(src.r), opts)
				if err != nil {
					t.Fatalf("Chunked %s: %v", id, err)
				}
				if lo != wlo || hi != whi {
					t.Fatalf("Chunked %s: bounds (%v, %v), want (%v, %v)", id, lo, hi, wlo, whi)
				}
				checkResult(t, sink.Raster(), wantCells)
			}
		}
	}
}

func TestNormalizeDegenerate(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	isNaN := func(v float32) bool { return v != v }
	run := func(data []float32, valid []bool) (raster.Float32Raster, float32, float32) {
		src := raster.NewFloat32(len(data), 1, data)
		if valid != nil {
			src.Valid = raster.NewMask(len(data))
			for i, ok := range valid {
				raster.MaskSet(src.Valid, i, ok)
			}
		}
		dst := raster.NewFloat32Like(src)
		lo, hi := algebra.Normalize(dst, src)
		return dst, lo, hi
	}

	dst, lo, hi := run([]float32{4, 4, 4}, nil)
	if lo != 4 || hi != 4 || !isNaN(dst.Data[0]) {
		t.Errorf("constant: bounds (%v, %v), cell %v; want (4, 4) and NaN", lo, hi, dst.Data[0])
	}

	dst, lo, hi = run([]float32{1, 2, 3}, []bool{false, false, false})
	if !isNaN(lo) || !isNaN(hi) || dst.IsValid(0, 0) || dst.IsValid(1, 0) || dst.IsValid(2, 0) {
		t.Errorf("no valid cells: bounds (%v, %v), validity %v %v %v; want NaN, NaN and none valid",
			lo, hi, dst.IsValid(0, 0), dst.IsValid(1, 0), dst.IsValid(2, 0))
	}

	dst, lo, hi = run([]float32{1, nan, 3}, nil)
	if !isNaN(lo) || !isNaN(hi) || !isNaN(dst.Data[0]) || !isNaN(dst.Data[2]) {
		t.Errorf("valid NaN: bounds (%v, %v), cells %v; want every value NaN", lo, hi, dst.Data)
	}

	// An invalid NaN is NoData, not a value: it must not reach the bounds.
	dst, lo, hi = run([]float32{1, nan, 3}, []bool{true, false, true})
	if lo != 1 || hi != 3 || dst.Data[0] != 0 || dst.Data[2] != 1 {
		t.Errorf("invalid NaN: bounds (%v, %v), cells %v; want (1, 3), 0 and 1", lo, hi, dst.Data)
	}

	dst, lo, hi = run([]float32{1, 2, inf}, nil)
	if lo != 1 || hi != inf || dst.Data[0] != 0 || dst.Data[1] != 0 || !isNaN(dst.Data[2]) {
		t.Errorf("+Inf cell: bounds (%v, %v), cells %v; want (1, +Inf), 0, 0, NaN", lo, hi, dst.Data)
	}

	dst, _, _ = run([]float32{-math.MaxFloat32, 0, math.MaxFloat32}, nil)
	if dst.Data[0] != 0 || dst.Data[1] != 0 || !isNaN(dst.Data[2]) {
		t.Errorf("overflowing span: cells %v; want 0, 0, NaN", dst.Data)
	}
}

func TestNormalizePanicsBeforeReducing(t *testing.T) {
	r := func(w, h int) raster.Float32Raster { return raster.NewFloat32(w, h, make([]float32, w*h)) }
	masked := r(4, 3)
	masked.Valid = raster.NewMask(12)
	mustPanic(t, "algebra.Normalize: src dimensions differ from dst", func() { algebra.Normalize(r(4, 3), r(3, 4)) })
	mustPanic(t, "dst.Valid is nil", func() { algebra.Normalize(r(4, 3), masked) })
	big := r(10, 10)
	mustPanic(t, "dst overlaps src at a different offset", func() {
		algebra.Normalize(big.Window(1, 0, 4, 3), big.Window(0, 0, 4, 3))
	})
	mustPanic(t, "algebra.NormalizeTiled: src dimensions differ from dst", func() {
		_, _, _ = algebra.NormalizeTiled(context.Background(), r(4, 3), r(3, 4), engine.Options{})
	})
}

func TestNormalizeCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	src := raster.NewFloat32(64, 64, make([]float32, 64*64))
	dst := raster.NewFloat32Like(src)
	lo, hi, err := algebra.NormalizeTiled(ctx, dst, src, engine.Options{TileHeight: 1})
	if !errors.Is(err, context.Canceled) || lo != 0 || hi != 0 {
		t.Errorf("NormalizeTiled = (%v, %v, %v), want (0, 0, context.Canceled)", lo, hi, err)
	}
	sink := engine.NewMemorySink(dst)
	lo, hi, err = algebra.NormalizeChunked(ctx, sink, engine.NewMemorySource(src), engine.Options{TileHeight: 1})
	if !errors.Is(err, context.Canceled) || lo != 0 || hi != 0 {
		t.Errorf("NormalizeChunked = (%v, %v, %v), want (0, 0, context.Canceled)", lo, hi, err)
	}
}

func ExampleNormalize() {
	// Elevations with one NoData cell, which takes no part in the range.
	src := raster.NewFloat32(5, 1, []float32{1200, 1250, -9999, 1300, 1225})
	src.Valid = raster.NewMask(5)
	for x := range 5 {
		src.SetValid(x, 0, x != 2)
	}
	dst := raster.NewFloat32Like(src)
	lo, hi := algebra.Normalize(dst, src)
	fmt.Println(lo, hi)
	for x := range 5 {
		if dst.IsValid(x, 0) {
			fmt.Print(dst.Data[x], " ")
		} else {
			fmt.Print("nodata ")
		}
	}
	fmt.Println()
	// Output:
	// 1200 1300
	// 0 0.5 nodata 1 0.25
}
