package resamp

// Backend function variables (DESIGN.md §17): scalar by default, swapped
// for SIMD kernels by a simd_<arch>.go init when the build and CPU have
// them.
var (
	hRows = scalarHRows
	vRow  = scalarVRow
)

// kernelSet is one backend's kernels.
type kernelSet struct {
	hRows func(t []float32, tStride int, src []float32, sStride, rows int, a *Axis, c0, c1, sx0 int, scratch []float32)
	vRow  func(dst, t []float32, tStride int, w []float32)
}

var scalarKernels = kernelSet{hRows: scalarHRows, vRow: scalarVRow}

// simdKernels is the SIMD set, or nil when this build or CPU has none;
// simdName is what Backend reports for it.
var (
	simdKernels *kernelSet
	simdName    string
	usingScalar bool
)

func (k *kernelSet) install() { hRows, vRow = k.hRows, k.vRow }

// Backend names the kernels in use: "avx2", "neon" or "scalar".
func Backend() string {
	if simdKernels != nil && !usingScalar {
		return simdName
	}
	return "scalar"
}

// UseScalar forces the scalar kernels (true) or restores the best
// available backend (false). It is for equivalence tests and
// benchmarks, and must not be called while kernels run.
func UseScalar(scalar bool) {
	usingScalar = scalar
	if scalar || simdKernels == nil {
		scalarKernels.install()
		return
	}
	simdKernels.install()
}

// scratchLanes is the widest backend's lane count; HScratch sizes the
// horizontal pass's scratch for it.
const scratchLanes = 8

// HScratch is the scratch length HRows needs for a source footprint
// fpW cells wide.
func HScratch(fpW int) int { return scratchLanes * (fpW + 2*scratchLanes) }

// HRows runs the horizontal pass: for each of rows source rows j and
// output columns c in [c0, c1) of a, all covered,
//
//	t[j*tStride + c-c0] = Σ_k a.W[a.Off[c]+k] · src[j*sStride + a.First[c]-sx0 + k]
//
// src must hold every tap: row j's cells are src[j*sStride+i] for i in
// the footprint of [c0, c1) less sx0. scratch must be at least
// HScratch(footprint width) long; its contents are clobbered. The result
// is the same bits on every backend.
func HRows(t []float32, tStride int, src []float32, sStride, rows int, a *Axis, c0, c1, sx0 int, scratch []float32) {
	if rows <= 0 || c1 <= c0 {
		return
	}
	hRows(t, tStride, src, sStride, rows, a, c0, c1, sx0, scratch)
}

// VRow runs the vertical pass for one output row: with n = len(w) taps,
//
//	dst[c] = Σ_k w[k] · t[k*tStride + c]
//
// It panics if w is empty or t is too short.
func VRow(dst, t []float32, tStride int, w []float32) {
	if len(w) == 0 {
		panic("resamp: VRow needs a tap")
	}
	if len(dst) == 0 {
		return
	}
	if need := (len(w)-1)*tStride + len(dst); len(t) < need {
		panic("resamp: VRow intermediate too short")
	}
	vRow(dst, t, tStride, w)
}
