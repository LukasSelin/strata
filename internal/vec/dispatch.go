package vec

// Backend function variables. A SIMD build swaps these in an init()
// (or via build-tag-gated files) rather than branching per call inside
// the exported functions below, so hot loops never pay a dispatch cost.
var (
	addFloat32       = scalarAddFloat32
	subFloat32       = scalarSubFloat32
	mulFloat32       = scalarMulFloat32
	divFloat32       = scalarDivFloat32
	addScalarFloat32 = scalarAddScalarFloat32
	mulScalarFloat32 = scalarMulScalarFloat32
	minFloat32       = scalarMinFloat32
	maxFloat32       = scalarMaxFloat32
	clampFloat32     = scalarClampFloat32
	absFloat32       = scalarAbsFloat32
	sqrtFloat32      = scalarSqrtFloat32
)

// kernelSet is one backend's kernels, one field per function variable.
type kernelSet struct {
	add, sub, mul, div func(dst, a, b []float32)
	addScalar          func(dst, src []float32, value float32)
	mulScalar          func(dst, src []float32, value float32)
	min, max           func(dst, a, b []float32)
	clamp              func(dst, src []float32, lo, hi float32)
	abs, sqrt          func(dst, src []float32)
}

var scalarKernels = kernelSet{
	add:       scalarAddFloat32,
	sub:       scalarSubFloat32,
	mul:       scalarMulFloat32,
	div:       scalarDivFloat32,
	addScalar: scalarAddScalarFloat32,
	mulScalar: scalarMulScalarFloat32,
	min:       scalarMinFloat32,
	max:       scalarMaxFloat32,
	clamp:     scalarClampFloat32,
	abs:       scalarAbsFloat32,
	sqrt:      scalarSqrtFloat32,
}

// simdKernels is the SIMD set, or nil when this build or CPU has none.
var simdKernels *kernelSet

func (k *kernelSet) install() {
	addFloat32, subFloat32, mulFloat32, divFloat32 = k.add, k.sub, k.mul, k.div
	addScalarFloat32, mulScalarFloat32 = k.addScalar, k.mulScalar
	minFloat32, maxFloat32, clampFloat32 = k.min, k.max, k.clamp
	absFloat32, sqrtFloat32 = k.abs, k.sqrt
}

// Backend names the kernels currently in use: "avx2" or "scalar".
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

func requireEqualLen2(dst, src []float32) {
	if len(src) != len(dst) {
		panic("vec: dst and src must have equal length")
	}
}

func requireEqualLen3(dst, a, b []float32) {
	if len(a) != len(dst) || len(b) != len(dst) {
		panic("vec: dst, a, and b must have equal length")
	}
}

// Add computes dst[i] = a[i] + b[i].
func Add(dst, a, b []float32) {
	requireEqualLen3(dst, a, b)
	addFloat32(dst, a, b)
}

// Sub computes dst[i] = a[i] - b[i].
func Sub(dst, a, b []float32) {
	requireEqualLen3(dst, a, b)
	subFloat32(dst, a, b)
}

// Mul computes dst[i] = a[i] * b[i].
func Mul(dst, a, b []float32) {
	requireEqualLen3(dst, a, b)
	mulFloat32(dst, a, b)
}

// Div computes dst[i] = a[i] / b[i].
func Div(dst, a, b []float32) {
	requireEqualLen3(dst, a, b)
	divFloat32(dst, a, b)
}

// AddScalar computes dst[i] = src[i] + value.
func AddScalar(dst, src []float32, value float32) {
	requireEqualLen2(dst, src)
	addScalarFloat32(dst, src, value)
}

// MulScalar computes dst[i] = src[i] * value.
func MulScalar(dst, src []float32, value float32) {
	requireEqualLen2(dst, src)
	mulScalarFloat32(dst, src, value)
}

// Min computes dst[i] = min(a[i], b[i]).
func Min(dst, a, b []float32) {
	requireEqualLen3(dst, a, b)
	minFloat32(dst, a, b)
}

// Max computes dst[i] = max(a[i], b[i]).
func Max(dst, a, b []float32) {
	requireEqualLen3(dst, a, b)
	maxFloat32(dst, a, b)
}

// Clamp computes dst[i] = clamp(src[i], lo, hi).
func Clamp(dst, src []float32, lo, hi float32) {
	requireEqualLen2(dst, src)
	clampFloat32(dst, src, lo, hi)
}

// Abs computes dst[i] = abs(src[i]).
func Abs(dst, src []float32) {
	requireEqualLen2(dst, src)
	absFloat32(dst, src)
}

// Sqrt computes dst[i] = sqrt(src[i]).
func Sqrt(dst, src []float32) {
	requireEqualLen2(dst, src)
	sqrtFloat32(dst, src)
}
