//go:build goexperiment.simd && amd64

package fusion

import "simd/archsimd"

// The AVX2 fused loop, written with simd/archsimd like internal/vec
// (docs/adr/0001-simd-backend.md): array-pointer loads over slices that
// shrink by one lane per iteration, so the compiler drops the per-load
// bounds checks, and VZEROUPPER before the scalar tail. It exists only in
// GOEXPERIMENT=simd builds; other builds get simd_other.go.

// haveSIMD reports whether mul6SIMD can run on this machine, and
// simdBackend is the vec.Backend name it goes with.
var haveSIMD = archsimd.X86.AVX2()

const simdBackend = "avx2"

const lane = 8

func load8(s []float32) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Array((*[lane]float32)(s))
}

// mul6SIMD is mul6Scalar eight cells at a time. VMULPS rounds each
// product, so the lanes multiply in the chain's order and agree with it
// bit for bit.
func mul6SIMD(dst, a, b, c, d, e, f []float32) {
	a, b, c = a[:len(dst)], b[:len(dst)], c[:len(dst)]
	d, e, f = d[:len(dst)], e[:len(dst)], f[:len(dst)]
	for len(dst) >= lane && len(a) >= lane && len(b) >= lane && len(c) >= lane &&
		len(d) >= lane && len(e) >= lane && len(f) >= lane {
		v := load8(a).Mul(load8(b)).Mul(load8(c)).Mul(load8(d)).Mul(load8(e)).Mul(load8(f))
		v.StoreArray((*[lane]float32)(dst))
		dst, a, b, c = dst[lane:], a[lane:], b[lane:], c[lane:]
		d, e, f = d[lane:], e[lane:], f[lane:]
	}
	archsimd.ClearAVXUpperBits()
	mul6Scalar(dst, a, b, c, d, e, f)
}
