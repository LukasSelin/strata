//go:build goexperiment.simd && amd64

package curve

import "simd/archsimd"

// This file is the AVX2 backend, written with simd/archsimd
// (docs/adr/0001-simd-backend.md). It only exists in builds with
// GOEXPERIMENT=simd; other builds keep the scalar function variables from
// dispatch.go. Every kernel agrees bit for bit with its scalar
// counterpart in scalar.go, and returns a NaN cell's own bits.
//
// The scalar kernels search each cell's place in the table with a scan
// that stops early. Lanes cannot stop early independently, so the vector
// kernels walk the whole table for eight cells at once and select with
// blends: a comparison per table entry gives a lane mask, and the entry's
// data is blended into the lanes the mask holds. Because the x side of a
// table increases strictly, the last entry at or below a cell is the last
// blend that takes it, which is exactly the entry the scalar scan stops
// after. There are no gathers, which are slow on Zen 2, and no table
// index is ever formed.
//
// The table is expanded once per call into one 32-byte vector per entry,
// in a stack array the wrapper declares, so the loop loads each entry as
// a single aligned VEX load instead of rebroadcasting a float32, which
// archsimd does through an XMM insert. The expansion is scalar code and
// runs before the first 256-bit instruction.
//
// Each kernel is split in two, as in internal/vec and internal/stencil: a
// *Lanes function runs the vector loop, clears the upper AVX bits and
// returns how many cells it wrote, and the wrapper runs the scalar tail.
// Loops use array-pointer loads over slices that shrink by one lane per
// iteration, so the loads carry no bounds checks, and walk the table with
// range over the expanded entries, so neither do the table loads.
//
// A table longer than reclassVecMax breaks or lookupVecMax knots goes to
// the scalar kernel whole: the blend walk costs one step per entry for
// every cell, where the scan stops at the cell's own entry, so past some
// size the scan wins. That size is measured (BenchmarkTableSize), not
// assumed; see DESIGN.md §50.

const lane = 8

// reclassVecMax and lookupVecMax are the longest tables the vector
// kernels take. They bound the stack arrays below, so each is also the
// per-call zeroing cost of its kernel.
const (
	reclassVecMax = 64
	lookupVecMax  = 64
)

func init() {
	// AVX2, not just AVX: IfElse needs AVX2 instructions.
	if !archsimd.X86.AVX2() {
		return
	}
	simdKernels = &kernelSet{
		reclass: reclassFloat32AVX2,
		lookup:  lookupFloat32AVX2,
	}
	UseScalar(false)
}

// vec8 is one table value broadcast to every lane.
type vec8 = [lane]float32

func splat(x float32) vec8 {
	return vec8{x, x, x, x, x, x, x, x}
}

func load8(s []float32) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Array((*[lane]float32)(s))
}

func store8(v archsimd.Float32x8, s []float32) {
	v.StoreArray((*[lane]float32)(s))
}

// reclassStep is one break and the value that applies at or above it.
type reclassStep struct {
	b, v vec8
}

// reclassFloat32AVX2 blends values[j+1] into every lane at or above
// breaks[j], starting from values[0]. A NaN lane is at or above no break
// and would end on values[0], so it takes the cell back at the end, which
// is the scalar kernel's NaN test done last instead of first.
//
// Reclass does no arithmetic, so its result is identical to the scalar
// kernel's by construction; the comparisons treat -0 and +0 as equal,
// as the scalar v < b does.
func reclassFloat32AVX2(dst, src, breaks, values []float32) {
	if len(src) < lane || len(breaks) > reclassVecMax {
		scalarReclassFloat32(dst, src, breaks, values)
		return
	}
	var table [reclassVecMax]reclassStep
	steps := table[:len(breaks)]
	above := values[1:][:len(breaks)] // above[j] applies at or above breaks[j]
	for j, b := range breaks {
		steps[j] = reclassStep{splat(b), splat(above[j])}
	}
	first := splat(values[0])
	i := reclassLanes(dst, src, &first, steps)
	scalarReclassFloat32(dst[i:], src[i:], breaks, values)
}

func reclassLanes(dst, src []float32, first *vec8, steps []reclassStep) int {
	v0 := archsimd.LoadFloat32x8Array(first)
	n := len(dst)
	src = src[:n]
	for len(dst) >= lane && len(src) >= lane {
		v := load8(src)
		r := v0
		for k := range steps {
			s := &steps[k]
			r = archsimd.LoadFloat32x8Array(&s.v).IfElse(v.GreaterEqual(archsimd.LoadFloat32x8Array(&s.b)), r)
		}
		store8(v.IfElse(v.IsNaN(), r), dst)
		dst, src = dst[lane:], src[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

// lookupKnot is one knot and the segment that starts at it: x0 and y0,
// and the run and rise to the next knot. The last knot has no segment;
// its dx and dy are never used, because a lane at or above it takes y0.
type lookupKnot struct {
	x, y, dx, dy vec8
}

// lookupFloat32AVX2 blends each lane's segment — x0, y0, and the run and
// rise to the next knot — out of the knot table, then evaluates the
// scalar kernel's segment value in the scalar kernel's operation order:
//
//	t = (v - x0) / dx,  r = y0 + t·dy
//
// as VSUBPS, VDIVPS, VMULPS, VADDPS, never a fused VFMADD. dx and dy are
// xs[k]-x0 and ys[k]-y0, the same float32 subtractions the scalar kernel
// does per cell, done once per call; addition commutes bit for bit, so
// y0 + p and p + y0 agree. Every lane computes the segment value; lanes
// the scalar kernel would not interpolate — below the first knot, at or
// above the last, or exactly on a knot — then take y0 instead, whatever
// the arithmetic gave, and NaN lanes take the cell.
func lookupFloat32AVX2(dst, src, xs, ys []float32) {
	if len(src) < lane || len(xs) > lookupVecMax {
		scalarLookupFloat32(dst, src, xs, ys)
		return
	}
	var table [lookupVecMax]lookupKnot
	knots := expandLookup(&table, xs, ys)
	i := lookupLanes(dst, src, knots)
	scalarLookupFloat32(dst[i:], src[i:], xs, ys)
}

// expandLookup fills table with one lookupKnot per knot and returns the
// filled part. xs must be non-empty and no longer than the table.
func expandLookup(table *[lookupVecMax]lookupKnot, xs, ys []float32) []lookupKnot {
	knots := table[:len(xs)]
	last := len(knots) - 1
	// Each segment as four slices of one length, so the loop proves its
	// indexes without a check.
	x0s := xs[:last]
	y0s, x1s, y1s, segs := ys[:len(x0s)], xs[1:][:len(x0s)], ys[1:][:len(x0s)], knots[:len(x0s)]
	for k, x0 := range x0s {
		segs[k] = lookupKnot{splat(x0), splat(y0s[k]), splat(x1s[k] - x0), splat(y1s[k] - y0s[k])}
	}
	knots[last] = lookupKnot{x: splat(xs[last]), y: splat(ys[last]), dx: splat(1)}
	return knots
}

func lookupLanes(dst, src []float32, knots []lookupKnot) int {
	if len(knots) == 0 {
		return 0
	}
	k0 := &knots[0]
	lo := archsimd.LoadFloat32x8Array(&k0.x)
	hi := archsimd.LoadFloat32x8Array(&knots[len(knots)-1].x)
	rest := knots[1:]
	n := len(dst)
	src = src[:n]
	for len(dst) >= lane && len(src) >= lane {
		v := load8(src)
		x0, y0 := lo, archsimd.LoadFloat32x8Array(&k0.y)
		dx, dy := archsimd.LoadFloat32x8Array(&k0.dx), archsimd.LoadFloat32x8Array(&k0.dy)
		for k := range rest {
			kn := &rest[k]
			m := v.GreaterEqual(archsimd.LoadFloat32x8Array(&kn.x))
			x0 = archsimd.LoadFloat32x8Array(&kn.x).IfElse(m, x0)
			y0 = archsimd.LoadFloat32x8Array(&kn.y).IfElse(m, y0)
			dx = archsimd.LoadFloat32x8Array(&kn.dx).IfElse(m, dx)
			dy = archsimd.LoadFloat32x8Array(&kn.dy).IfElse(m, dy)
		}
		t := v.Sub(x0).Div(dx)
		r := y0.Add(t.Mul(dy))
		keep := v.Equal(x0).Or(v.Less(lo)).Or(v.GreaterEqual(hi))
		r = y0.IfElse(keep, r)
		store8(v.IfElse(v.IsNaN(), r), dst)
		dst, src = dst[lane:], src[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}
