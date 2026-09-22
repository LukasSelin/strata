//go:build goexperiment.simd && arm64

package fusion

import "simd/archsimd"

// The NEON fused loop, simd_amd64.go four lanes at a time, with the same
// array-pointer loads over shrinking slices. arm64 has no SSE/AVX
// transition, so nothing is cleared before the scalar tail. It exists
// only in GOEXPERIMENT=simd builds; other builds get simd_other.go.

// haveSIMD is always true: NEON is part of the arm64 baseline.
const haveSIMD = true

const simdBackend = "neon"

const lane = 4

func load4(s []float32) archsimd.Float32x4 {
	return archsimd.LoadFloat32x4Array((*[lane]float32)(s))
}

// mul6SIMD is mul6Scalar four cells at a time. FMUL rounds each product,
// so the lanes multiply in the chain's order and agree with it bit for
// bit.
func mul6SIMD(dst, a, b, c, d, e, f []float32) {
	a, b, c = a[:len(dst)], b[:len(dst)], c[:len(dst)]
	d, e, f = d[:len(dst)], e[:len(dst)], f[:len(dst)]
	for len(dst) >= lane && len(a) >= lane && len(b) >= lane && len(c) >= lane &&
		len(d) >= lane && len(e) >= lane && len(f) >= lane {
		v := load4(a).Mul(load4(b)).Mul(load4(c)).Mul(load4(d)).Mul(load4(e)).Mul(load4(f))
		v.StoreArray((*[lane]float32)(dst))
		dst, a, b, c = dst[lane:], a[lane:], b[lane:], c[lane:]
		d, e, f = d[lane:], e[lane:], f[lane:]
	}
	mul6Scalar(dst, a, b, c, d, e, f)
}
