//go:build goexperiment.simd && (amd64 || arm64)

package accum

// This file is the architecture-independent half of the vector backends
// of Sum and Moments (simd_amd64.go, simd_arm64.go). They add 64-cell
// blocks in registers instead of in bins.
//
// Within a block, every value is shifted onto the block's base exponent
// field and added to a 64-bit lane. A significand is below 2^24, so with
// the base at most sumWindow fields below the block's largest a shifted
// one is below 2^56, and 64 of them stay below 2^62. The block's sum is
// then added to three bins in 24-bit pieces: the same integer the scalar
// loop would have added cell by cell. So the bins, and every result, are
// bit for bit the scalar backend's by construction rather than by
// matching an evaluation order.
//
// Squares go the same way. A squared significand is below 2^48 and is
// split into two 24-bit halves as in addFinite; each is shifted by twice
// the value's distance from the base, so the base may only be
// squareWindow fields below the largest.
//
// A block holding a NaN or an infinity, or a non-zero value further below
// its largest than the window, goes to the scalar loop whole. So does the
// tail after the last whole block. Zeros shift harmlessly wherever they
// are.
//
// Each architecture file supplies blockBase and blockSums, which do the
// vector work, plus vectorBackend, haveVector and clearUpper.

const (
	block        = 64
	sumWindow    = 32
	squareWindow = 16
	pieceMask    = 1<<24 - 1
)

func init() { UseScalar(false) }

// UseScalar switches Sum and Moments to the scalar loops (true) or back to
// the best available ones (false), for tests and benchmarks that compare
// the two in one binary. The results are the same bits either way. It is
// not safe to call while an accumulator is being added to.
func UseScalar(scalar bool) {
	if scalar || !haveVector() {
		sumKernel, momentsKernel, backend = addSums, addMoments, "scalar"
	} else {
		sumKernel, momentsKernel, backend = addSumsBlocks, addMomentsBlocks, vectorBackend
	}
}

func addSumsBlocks(bins *[lanes][sumBins]int64, sp *specials, xs []float32) uint32 {
	var nzAll uint32
	for ; len(xs) >= block; xs = xs[block:] {
		blk := (*[block]float32)(xs)
		base, nz, ok := blockBase(blk, sumWindow)
		nzAll |= nz
		if !ok {
			addSums(bins, sp, blk[:])
			continue
		}
		s, _, _ := blockSums(blk, base, false)
		addSumPieces(&bins[0], uint8(base), s) // #nosec G115 -- blockBase's base is in [1, 222]
	}
	clearUpper()
	return nzAll | addSums(bins, sp, xs)
}

func addMomentsBlocks(bins *[lanes][sumBins]int64, sq *[lanes][sqBins]int64, sp *specials, xs []float32) uint32 {
	var nzAll uint32
	for ; len(xs) >= block; xs = xs[block:] {
		blk := (*[block]float32)(xs)
		base, nz, ok := blockBase(blk, squareWindow)
		nzAll |= nz
		if !ok {
			addMoments(bins, sq, sp, blk[:])
			continue
		}
		s, lo, hi := blockSums(blk, base, true)
		addSumPieces(&bins[0], uint8(base), s)       // #nosec G115 -- blockBase's base is in [1, 238]
		addSquarePieces(&sq[0], uint8(base), lo, hi) // #nosec G115 -- as above
	}
	clearUpper()
	return nzAll | addMoments(bins, sq, sp, xs)
}

// addSumPieces adds a, a multiple of the weight of sum bin base with
// |a| < 2^62, to bins base, base+24 and base+48 in 24-bit pieces, so that
// no bin takes more than a cell would add. base is a uint8 so that every
// index is provably in range.
func addSumPieces(b *[sumBins]int64, base uint8, a int64) {
	p0, p1, p2 := pieces(a)
	k := int(base)
	b[k] += p0
	b[k+24] += p1
	b[k+48] += p2
}

// addSquarePieces adds lo and hi, the block's sums of square halves, as
// multiples of the weights of square bins 2·base and 2·base+24.
func addSquarePieces(q *[sqBins]int64, base uint8, lo, hi int64) {
	l0, l1, l2 := pieces(lo)
	h0, h1, h2 := pieces(hi)
	j := 2 * int(base)
	q[j] += l0
	q[j+24] += l1 + h0
	q[j+48] += l2 + h1
	q[j+72] += h2
}

// pieces splits a, |a| < 2^62, into three signed 24-bit pieces with
// a = p0 + p1·2^24 + p2·2^48.
func pieces(a int64) (p0, p1, p2 int64) {
	neg := a >> 63
	m := (a ^ neg) - neg
	return (m&pieceMask ^ neg) - neg, ((m>>24)&pieceMask ^ neg) - neg, (m>>48 ^ neg) - neg
}
