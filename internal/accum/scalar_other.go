//go:build !(goexperiment.simd && amd64)

package accum

// useScalar is a no-op where there is no vector backend.
func useScalar(bool) {}
