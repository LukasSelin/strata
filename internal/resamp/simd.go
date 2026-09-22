//go:build goexperiment.simd && (amd64 || arm64)

package resamp

// span returns the source range [lo, hi) the taps of output columns
// [c0, c1) of a cover.
func span(a *Axis, c0, c1 int) (lo, hi int) {
	first := a.First[c0:c1]
	taps := a.Taps[c0:c1][:len(first)]
	lo = int(first[0])
	for i, f := range first {
		lo = min(lo, int(f))
		hi = max(hi, int(f+taps[i]))
	}
	return lo, hi
}
