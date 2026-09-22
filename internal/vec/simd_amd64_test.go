//go:build goexperiment.simd && amd64

package vec

import (
	"simd/archsimd"
	"testing"
)

// simdTestName and simdTestKernels are what simd_test.go expects this
// architecture's SIMD backend to be.
const simdTestName = "avx2"

var simdTestKernels = kernelSet{
	add: addFloat32AVX2, sub: subFloat32AVX2, mul: mulFloat32AVX2, div: divFloat32AVX2,
	addScalar: addScalarFloat32AVX2, mulScalar: mulScalarFloat32AVX2,
	affine: affineFloat32AVX2, subDiv: subDivFloat32AVX2,
	min: minFloat32AVX2, max: maxFloat32AVX2, clamp: clampFloat32AVX2,
	abs: absFloat32AVX2, sqrt: sqrtFloat32AVX2,
	reduceMin: reduceMinFloat32AVX2, reduceMax: reduceMaxFloat32AVX2,
}

func requireSIMD(tb testing.TB) {
	tb.Helper()
	if !archsimd.X86.AVX2() {
		tb.Skip("AVX2 not available on this CPU")
	}
}

// foldMin8 and foldMax8 fold rows of src into four accumulators the way a
// focal kernel does (internal/focalrow's extremeLanes): the accumulators
// live in registers across the loop and every other operand is a fresh
// load. The compiler treats archsimd's Min and Max as commutative, and
// under this register pressure it emits some VMINPS/VMAXPS with the
// loaded row as the second source, so min8/max8 must not rely on which
// operand VMINPS returns when they are unordered. A row is 4 vectors.
//
//go:noinline
func foldMin8(src []float32) [4 * avxLane]float32 {
	a0, a1, a2, a3 := load8(src), load8(src[avxLane:]), load8(src[2*avxLane:]), load8(src[3*avxLane:])
	for src = src[4*avxLane:]; len(src) >= 4*avxLane; src = src[4*avxLane:] {
		a0, a1 = min8(a0, load8(src)), min8(a1, load8(src[avxLane:]))
		a2, a3 = min8(a2, load8(src[2*avxLane:])), min8(a3, load8(src[3*avxLane:]))
	}
	var out [4 * avxLane]float32
	a0.StoreArray((*[avxLane]float32)(out[:]))
	a1.StoreArray((*[avxLane]float32)(out[avxLane:]))
	a2.StoreArray((*[avxLane]float32)(out[2*avxLane:]))
	a3.StoreArray((*[avxLane]float32)(out[3*avxLane:]))
	archsimd.ClearAVXUpperBits()
	return out
}

//go:noinline
func foldMax8(src []float32) [4 * avxLane]float32 {
	a0, a1, a2, a3 := load8(src), load8(src[avxLane:]), load8(src[2*avxLane:]), load8(src[3*avxLane:])
	for src = src[4*avxLane:]; len(src) >= 4*avxLane; src = src[4*avxLane:] {
		a0, a1 = max8(a0, load8(src)), max8(a1, load8(src[avxLane:]))
		a2, a3 = max8(a2, load8(src[2*avxLane:])), max8(a3, load8(src[3*avxLane:]))
	}
	var out [4 * avxLane]float32
	a0.StoreArray((*[avxLane]float32)(out[:]))
	a1.StoreArray((*[avxLane]float32)(out[avxLane:]))
	a2.StoreArray((*[avxLane]float32)(out[2*avxLane:]))
	a3.StoreArray((*[avxLane]float32)(out[3*avxLane:]))
	archsimd.ClearAVXUpperBits()
	return out
}

// TestMinMax8FoldNaNSecondOperand puts a single NaN in a loaded (second)
// operand of a register-accumulator fold, in every lane of every row after
// the first, with each special value as the first row. Go's min and max
// return NaN if either operand is NaN, so the lane that saw it must end as
// NaN whatever operand order the compiler gave VMINPS.
func TestMinMax8FoldNaNSecondOperand(t *testing.T) {
	requireSIMD(t)
	const width = 4 * avxLane
	for _, rows := range []int{2, 3} {
		for _, first := range []float32{inf, ninf, 0, negZero, 1} {
			for row := 1; row < rows; row++ {
				for lane := range width {
					src := make([]float32, rows*width)
					for i := range src {
						src[i] = float32(i%5) - 2
					}
					for i := range width {
						src[i] = first
					}
					src[row*width+lane] = nan
					gotMin, gotMax := foldMin8(src), foldMax8(src)
					for l := range width {
						wantMin, wantMax := src[l], src[l]
						for r := 1; r < rows; r++ {
							wantMin = min(wantMin, src[r*width+l])
							wantMax = max(wantMax, src[r*width+l])
						}
						if !sameBits(gotMin[l], wantMin) {
							t.Errorf("min rows=%d first=%v nan at row %d lane %d: lane %d = %v, want %v",
								rows, first, row, lane, l, gotMin[l], wantMin)
						}
						if !sameBits(gotMax[l], wantMax) {
							t.Errorf("max rows=%d first=%v nan at row %d lane %d: lane %d = %v, want %v",
								rows, first, row, lane, l, gotMax[l], wantMax)
						}
					}
				}
			}
		}
	}
}
