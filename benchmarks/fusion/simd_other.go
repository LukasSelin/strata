//go:build !(goexperiment.simd && amd64)

package fusion

// Without GOEXPERIMENT=simd on amd64 there is no AVX2 loop, and NewFused
// always returns the scalar one.

const haveAVX2 = false

func mul6AVX2(dst, a, b, c, d, e, f []float32) { mul6Scalar(dst, a, b, c, d, e, f) }
