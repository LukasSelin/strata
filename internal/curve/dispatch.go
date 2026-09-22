package curve

// Backend function variables. A SIMD build swaps these in an init()
// rather than branching per call inside the exported functions, so the
// kernels never pay a dispatch cost. This is internal/vec's machinery,
// with one kernelSet of its own so that Backend answers for exactly the
// kernels this package runs.
var (
	reclassFloat32 = scalarReclassFloat32
	lookupFloat32  = scalarLookupFloat32
)

// kernelSet is one backend's kernels, one field per function variable.
type kernelSet struct {
	reclass func(dst, src, breaks, values []float32)
	lookup  func(dst, src, xs, ys []float32)
}

var scalarKernels = kernelSet{
	reclass: scalarReclassFloat32,
	lookup:  scalarLookupFloat32,
}

// simdKernels is the SIMD set, or nil when this build or CPU has none.
var simdKernels *kernelSet

func (k *kernelSet) install() {
	reclassFloat32, lookupFloat32 = k.reclass, k.lookup
}

// Backend names the kernels currently in use: "avx2" or "scalar".
//
// The AVX2 kernels hand a table longer than their measured crossover to
// the scalar scan themselves (see simd_amd64.go), so "avx2" means the
// vector kernels are installed, not that every table runs in lanes.
func Backend() string {
	if simdKernels != nil && !usingScalar {
		return "avx2"
	}
	return "scalar"
}

var usingScalar bool

// UseScalar forces the scalar kernels (true) or restores the best
// available backend (false). It exists for equivalence tests and
// scalar-vs-SIMD benchmarks, and must not be called while kernels run.
func UseScalar(scalar bool) {
	usingScalar = scalar
	if scalar || simdKernels == nil {
		scalarKernels.install()
		return
	}
	simdKernels.install()
}
