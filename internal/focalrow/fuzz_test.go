package focalrow

import (
	"testing"

	"github.com/LukasSelin/strata/internal/fuzzdata"
)

// FuzzFocalRows runs a kernel on the current backend over arbitrary
// float bits, lengths, neighbourhoods and strides, and requires the bits
// of the per-cell definition (any NaN matching any NaN), no reads outside
// the input cells (the rest are poisoned) and no writes past dst. In a
// SIMD build that is the SIMD kernels against the definition; in the
// others, the scalar ones.
func FuzzFocalRows(f *testing.F) {
	f.Add([]byte{0, 33, 1, 2})
	f.Add([]byte{1, 40, 2, 0})
	f.Add([]byte{6, 17, 3, 1})
	f.Add([]byte{8, 9, 0, 3})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		kind, n, k, pad := d.IntN(numKernels), d.Range(0, 80), 2*d.Range(0, 8)+1, d.IntN(5)
		rows, cells, nw := shape(kind, n, k)
		stride := cells + pad
		src := make([]float32, (rows-1)*stride+cells+3)
		for i := range src {
			src[i] = 1e30
		}
		for j := range rows {
			for c := range cells {
				src[j*stride+c] = d.Float32()
			}
		}
		w := make([]float32, nw)
		for i := range w {
			if d.Bool() {
				w[i] = d.Float32()
			} else {
				w[i] = float32(int8(d.Byte())) / 16
			}
		}
		want := make([]float32, n)
		naive(kind, want, src, stride, w, k)
		got := make([]float32, n+3)
		for i := n; i < len(got); i++ {
			got[i] = 7
		}
		call(kind, got[:n], src, stride, w, k)
		for i := range want {
			if !sameBits(got[i], want[i]) {
				t.Fatalf("%s n=%d k=%d stride=%d: cell %d = %v on %s, want %v",
					kernelNames[kind], n, k, stride, i, got[i], Backend(), want[i])
			}
		}
		for i := n; i < len(got); i++ {
			if got[i] != 7 {
				t.Fatalf("%s n=%d k=%d: wrote past dst at %d", kernelNames[kind], n, k, i)
			}
		}
	})
}
