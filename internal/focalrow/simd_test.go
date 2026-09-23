//go:build goexperiment.simd && (amd64 || arm64)

package focalrow

import (
	"math/rand/v2"
	"testing"
)

// repoison overwrites every cell of src that is not an input cell.
func repoison(kind, n, k, stride int, src []float32, poison float32) {
	rows, cells, _ := shape(kind, n, k)
	if stride == 0 {
		for i := range src {
			src[i] = poison
		}
		return
	}
	for i := range src {
		j, c := i/stride, i%stride
		if j >= rows || c >= cells {
			src[i] = poison
		}
	}
}

func TestBackendSelection(t *testing.T) {
	requireSIMD(t)
	if got := Backend(); got != simdTestName {
		t.Fatalf("Backend() = %q, want %q", got, simdTestName)
	}
	UseScalar(true)
	if got := Backend(); got != "scalar" {
		t.Fatalf("after UseScalar(true), Backend() = %q", got)
	}
	UseScalar(false)
	if got := Backend(); got != simdTestName {
		t.Fatalf("after UseScalar(false), Backend() = %q", got)
	}
}

// TestSIMDRowsMatchScalar runs every kernel on both backends for every
// output length up to 120 (whole blocks, whole vectors and every tail),
// every neighbourhood up to 17 cells, and hazard values, and requires
// identical bits. The cells a kernel must not read are poisoned
// differently in the two runs, and the cells after dst must stay
// untouched.
func TestSIMDRowsMatchScalar(t *testing.T) {
	requireSIMD(t)
	defer UseScalar(false)
	rng := rand.New(rand.NewPCG(5, 6))
	for kind := range numKernels {
		for k := 1; k <= 17; k += 2 {
			for n := 0; n <= 120; n++ {
				special := []float64{0, 0.1, 0.5}[n%3]
				src, stride, w := operands(rng, kind, n, k, n%4, 3, special, 0)
				want := make([]float32, n+2)
				want[n], want[n+1] = 7, 7
				UseScalar(true)
				call(kind, want[:n], src, stride, w, k)
				repoison(kind, n, k, stride, src, 1e30)
				got := make([]float32, n+2)
				got[n], got[n+1] = 7, 7
				UseScalar(false)
				call(kind, got[:n], src, stride, w, k)
				for i := range want {
					if !sameBits(got[i], want[i]) {
						t.Fatalf("%s n=%d k=%d: cell %d = %v on %s, want %v", kernelNames[kind], n, k, i, got[i], Backend(), want[i])
					}
				}
			}
		}
	}
}

// TestSIMDMinMaxEdgePairs checks the min and max lanes on every ordered
// pair of hazards, where VMINPS and VMAXPS differ from Go's builtins.
func TestSIMDMinMaxEdgePairs(t *testing.T) {
	requireSIMD(t)
	defer UseScalar(false)
	for _, kind := range []int{kColumnMin, kColumnMax, kRowMin, kRowMax} {
		for _, a := range hazards {
			for _, b := range hazards {
				// 40 outputs cover a block, a vector and a tail. Values
				// alternate along rows and down columns, so every output
				// folds a and b, in either order.
				src := make([]float32, 3*64)
				for i := range src {
					if (i/64+i%64)%2 == 0 {
						src[i] = a
					} else {
						src[i] = b
					}
				}
				want, got := make([]float32, 40), make([]float32, 40)
				UseScalar(true)
				call(kind, want, src, 64, nil, 3)
				UseScalar(false)
				call(kind, got, src, 64, nil, 3)
				for i := range want {
					if !sameBits(got[i], want[i]) {
						t.Fatalf("%s(%v, %v): cell %d = %v, want %v", kernelNames[kind], a, b, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestSIMDFoldedRowsMatchScalar is TestSIMDRowsMatchScalar at strides
// whose rows collide in Zen 2's L2 (16384 cells, and 16400, a cache line
// apart modulo 64 KiB), where the AVX2 column passes over more than
// seven rows fold them a group at a time over chunks of 4096 cells: at
// lengths short of a chunk, on its boundary, over several with a partial
// last chunk, and with a tail after it. On NEON it is one more set of
// lengths and strides.
func TestSIMDFoldedRowsMatchScalar(t *testing.T) {
	requireSIMD(t)
	defer UseScalar(false)
	rng := rand.New(rand.NewPCG(7, 8))
	for _, kind := range []int{kCorrelateRow, kColumnCorrelate, kColumnSum, kColumnMin, kColumnMax} {
		for k := 7; k <= 17; k += 2 {
			for _, n := range []int{1000, 4095, 4096, 4097, 4096 + 40, 2*4096 + 1007} {
				for _, stride := range []int{16384, 16400} {
					_, cells, _ := shape(kind, n, k)
					src, stride, w := operands(rng, kind, n, k, stride-cells, 3, 0.05, 0)
					want := make([]float32, n+2)
					want[n], want[n+1] = 7, 7
					UseScalar(true)
					call(kind, want[:n], src, stride, w, k)
					repoison(kind, n, k, stride, src, 1e30)
					got := make([]float32, n+2)
					got[n], got[n+1] = 7, 7
					UseScalar(false)
					call(kind, got[:n], src, stride, w, k)
					for i := range want {
						if !sameBits(got[i], want[i]) {
							t.Fatalf("%s n=%d k=%d stride=%d: cell %d = %v on %s, want %v",
								kernelNames[kind], n, k, stride, i, got[i], Backend(), want[i])
						}
					}
				}
			}
		}
	}
}
