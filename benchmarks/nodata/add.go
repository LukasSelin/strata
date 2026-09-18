package nodata

import "github.com/LukasSelin/strata/internal/vec"

// Element-wise Add, dst = a + b, in every representation and form.
// All slices have equal length; masks have MaskWords(len) words.

// AddSentinelBranchy is the straightforward per-cell check.
func AddSentinelBranchy(dst, a, b []float32, nd float32) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		if a[i] == nd || b[i] == nd {
			dst[i] = nd
		} else {
			dst[i] = a[i] + b[i]
		}
	}
}

// AddSentinelSelect computes unconditionally and then selects, with no
// short-circuit, so the only control flow left is a select the compiler
// may lower to a conditional move.
func AddSentinelSelect(dst, a, b []float32, nd float32) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		av, bv := a[i], b[i]
		s := av + bv
		bad := b2u(av == nd) | b2u(bv == nd)
		if bad != 0 {
			s = nd
		}
		dst[i] = s
	}
}

// AddSentinelVecFixup uses the SIMD vec.Add, then a scalar fix-up pass
// that rewrites cells where either input was the sentinel.
func AddSentinelVecFixup(dst, a, b []float32, nd float32) {
	vec.Add(dst, a, b)
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		if a[i] == nd || b[i] == nd {
			dst[i] = nd
		}
	}
}

// AddNaNScalar is the plain loop; NaN propagates through +.
func AddNaNScalar(dst, a, b []float32) {
	a = a[:len(dst)]
	b = b[:len(dst)]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

// AddNaNVec is vec.Add unchanged.
func AddNaNVec(dst, a, b []float32) {
	vec.Add(dst, a, b)
}

// AddMaskScalar is the plain loop plus a word-level AND of the masks.
func AddMaskScalar(dst, a, b []float32, dstValid, aValid, bValid []uint64) {
	AddNaNScalar(dst, a, b)
	AndMasks(dstValid, aValid, bValid)
}

// AddMaskVec is vec.Add plus a word-level AND of the masks.
func AddMaskVec(dst, a, b []float32, dstValid, aValid, bValid []uint64) {
	vec.Add(dst, a, b)
	AndMasks(dstValid, aValid, bValid)
}

// AddMaskVecFill is AddMaskVec plus writing fill under invalid cells, for
// consumers that need defined values there.
func AddMaskVecFill(dst, a, b []float32, dstValid, aValid, bValid []uint64, fill float32) {
	AddMaskVec(dst, a, b, dstValid, aValid, bValid)
	FillInvalid(dst, dstValid, fill)
}
