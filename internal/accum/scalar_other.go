//go:build !(goexperiment.simd && (amd64 || arm64))

package accum

// UseScalar switches Sum and Moments to the scalar loops (true) or back to
// the best available ones (false). This build has only the scalar loops,
// so it does nothing.
func UseScalar(bool) {}
