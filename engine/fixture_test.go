package engine_test

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"strata/engine"
	"strata/raster"
)

// operand is a raster under test and the root that owns its memory, so
// tests can check that nothing outside the operand changed.
type operand struct {
	r, root raster.Float32Raster
	x, y    int // r's position in root
}

var hazards = []float32{
	float32(math.NaN()), math.Float32frombits(0xffc0_0001), float32(math.Inf(1)), float32(math.Inf(-1)),
	0, float32(math.Copysign(0, -1)), 3e38,
}

// newOperand returns a w×h raster: compact with no offset, or a window
// with Stride > Width and a mask offset into a larger root. Values are
// DEM-like with some hazards; masks have about 5% of cells invalid, and
// root bits outside the window are random.
func newOperand(rng *rand.Rand, w, h int, windowed, masked bool) operand {
	rootW, rootH, stride, x, y, validOffset := w, h, w, 0, 0, 0
	if windowed {
		x, y = 1+rng.IntN(3), 1+rng.IntN(2)
		rootW, rootH = w+x+2, h+y+2
		stride = rootW + rng.IntN(70)
		validOffset = rng.IntN(100)
	}
	n := (rootH-1)*stride + rootW
	root := raster.NewFloat32Stride(rootW, rootH, stride, make([]float32, n))
	for i := range root.Data {
		root.Data[i] = float32(1000 + 100*rng.NormFloat64())
		if rng.IntN(50) == 0 {
			root.Data[i] = hazards[rng.IntN(len(hazards))]
		}
	}
	if masked {
		root.Valid = make([]uint64, raster.MaskWords(validOffset+n)+1)
		root.ValidOffset = validOffset
		for k := range root.Valid {
			root.Valid[k] = rng.Uint64()
		}
		for i := range n {
			raster.MaskSet(root.Valid, validOffset+i, rng.IntN(20) != 0)
		}
	}
	return operand{r: root.Window(x, y, w, h), root: root, x: x, y: y}
}

// clone deep-copies an operand's root and returns the same window of it.
func (o operand) clone() operand {
	root := o.root
	root.Data = append([]float32(nil), root.Data...)
	if root.Valid != nil {
		root.Valid = append([]uint64(nil), root.Valid...)
	}
	return operand{r: root.Window(o.x, o.y, o.r.Width, o.r.Height), root: root, x: o.x, y: o.y}
}

func sameFloat(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

// requireSameRoots fails unless two roots hold the same bits: identical
// validity words, and Data equal (any NaN matching any NaN) wherever the
// cell is valid or there is no mask. Cells outside the operand, including
// row padding and invalid cells, are compared the same way, so an
// operation that wrote outside its window would be caught when the other
// did not.
func requireSameRoots(t *testing.T, id string, got, want raster.Float32Raster) {
	t.Helper()
	for k := range want.Valid {
		if got.Valid[k] != want.Valid[k] {
			t.Fatalf("%s: mask word %d = %#016x, want %#016x", id, k, got.Valid[k], want.Valid[k])
		}
	}
	for i := range want.Data {
		if want.Valid != nil && !raster.MaskGet(want.Valid, want.ValidOffset+i) {
			continue
		}
		if !sameFloat(got.Data[i], want.Data[i]) {
			t.Fatalf("%s: root cell %d = %g (%#x), want %g (%#x)", id, i,
				got.Data[i], math.Float32bits(got.Data[i]), want.Data[i], math.Float32bits(want.Data[i]))
		}
	}
}

// layout says which operands are windows (Stride > Width, mask offset)
// rather than compact rasters.
type layout struct{ in, out bool }

var layouts = []layout{{false, false}, {true, true}, {false, true}, {true, false}}

func (l layout) String() string { return fmt.Sprintf("windowedIn=%v windowedOut=%v", l.in, l.out) }

// engineRun is one way to run the engine. Every one must give the same
// bits as a whole-raster run.
type engineRun struct {
	opts     engine.Options
	oneRow   bool // one-row bands
	cancelOK bool // a cancellable context that is never cancelled
}

func (run engineRun) String() string {
	return fmt.Sprintf("opts=%+v oneRow=%v cancellable=%v", run.opts, run.oneRow, run.cancelOK)
}

var engineRuns = []engineRun{
	{engine.Options{}, false, false},
	{engine.Options{}, true, true},
	{engine.Options{TileWidth: 1, TileHeight: 1}, false, false},
	{engine.Options{TileWidth: 2, TileHeight: 3, Workers: 4}, true, false},
	{engine.Options{TileWidth: 3, TileHeight: 2}, false, true},
	{engine.Options{TileWidth: 5, TileHeight: 4}, true, false},
	{engine.Options{TileWidth: 64, TileHeight: 1}, false, false},
	{engine.Options{TileWidth: 7, TileHeight: 1000, Workers: 1}, false, false},
}

// process runs ProcessN the way run says and fails the test on an error.
func process(t *testing.T, run engineRun, dst, src []raster.Float32Raster, k engine.Kernel) {
	t.Helper()
	if run.oneRow {
		defer engine.SetBandCells(1)()
	}
	ctx := context.Background()
	if run.cancelOK {
		c, cancel := context.WithCancel(ctx)
		defer cancel()
		ctx = c
	}
	if err := engine.ProcessN(ctx, dst, src, k, run.opts); err != nil {
		t.Fatalf("ProcessN: %v", err)
	}
}
