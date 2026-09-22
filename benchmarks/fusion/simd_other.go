//go:build !(goexperiment.simd && (amd64 || arm64))

package fusion

// Without GOEXPERIMENT=simd on amd64 or arm64 there is no vector loop, and
// NewFused always returns the scalar one.

const haveSIMD = false

const simdBackend = ""

func mul6SIMD(dst, a, b, c, d, e, f []float32) { mul6Scalar(dst, a, b, c, d, e, f) }
