package reduce

import (
	"context"
	"math"
	"math/bits"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// Count returns the number of valid cells of src. A raster without a
// validity mask has every cell valid, so Count is then its cell count.
func Count(src raster.Float32Raster) int64 {
	n, err := CountTiled(context.Background(), src, plain)
	must(err)
	return n
}

// MinMax returns the smallest and largest of src's valid cells, with Go's
// builtin min and max, and how many cells took part. With no valid cells
// it returns NaN, NaN and 0.
//
// A NaN in a valid cell is an ordinary value, not NoData, so it makes
// both results NaN. See the package documentation for -0, +0 and which
// NaN comes back.
func MinMax(src raster.Float32Raster) (mn, mx float32, count int64) {
	mn, mx, count, err := MinMaxTiled(context.Background(), src, plain)
	must(err)
	return mn, mx, count
}

// plain is how the entry points without a context run their reduction:
// one worker and one tile, so runWorkers takes its no-goroutine path and
// package reduce, like algebra and terrain, starts none itself
// (DESIGN.md §26). The plain functions go through the engine so that they
// cannot drift from the Tiled and Chunked ones; unlike algebra's they
// allocate a few small slices per call, as terrain's do.
var plain = engine.Options{Workers: 1}

// must turns the error of a reduction that cannot fail into a panic. A
// call with a background context and one worker never cancels and reads
// nothing, so only a bug could produce an error here.
func must(err error) {
	if err != nil {
		panic("reduce: " + err.Error())
	}
}

// CountTiled is Count run by the engine.
func CountTiled(ctx context.Context, src raster.Float32Raster, opts engine.Options) (int64, error) {
	return exec.Reduce(ctx, []raster.Float32Raster{src}, countOp{}, opts)
}

// CountChunked is Count run by the engine over a source, with bounded
// memory. A source that is not Masked has every cell valid.
func CountChunked(ctx context.Context, src engine.RasterSource, opts engine.Options) (int64, error) {
	return exec.ReduceChunked(ctx, []engine.RasterSource{src}, countOp{}, opts)
}

// MinMaxTiled is MinMax run by the engine.
func MinMaxTiled(ctx context.Context, src raster.Float32Raster, opts engine.Options) (mn, mx float32, count int64, err error) {
	e, err := exec.Reduce(ctx, []raster.Float32Raster{src}, minMaxOp{}, opts)
	if err != nil {
		return 0, 0, 0, err
	}
	mn, mx, count = e.result()
	return mn, mx, count, nil
}

// MinMaxChunked is MinMax run by the engine over a source, with bounded
// memory.
func MinMaxChunked(ctx context.Context, src engine.RasterSource, opts engine.Options) (mn, mx float32, count int64, err error) {
	e, err := exec.ReduceChunked(ctx, []engine.RasterSource{src}, minMaxOp{}, opts)
	if err != nil {
		return 0, 0, 0, err
	}
	mn, mx, count = e.result()
	return mn, mx, count, nil
}

// countOp counts valid cells. Its partial is the count itself, whose
// zero value is the identity and whose combination is addition, so it is
// order-independent without any argument.
type countOp struct{}

func (countOp) Inputs() int { return 1 }

func (countOp) Combine(a *int64, b int64) { *a += b }

func (countOp) Fold(p *int64, c exec.Cells) {
	if !c.Masked {
		*p += int64(c.Width) * int64(c.Height)
		return
	}
	n := 0
	for y := range c.Height {
		for x := 0; x < c.Width; x += wordBits {
			n += bits.OnesCount64(c.ValidBits(x, y, min(wordBits, c.Width-x)))
		}
	}
	*p += int64(n)
}

// extent is the partial of a MinMax reduction: the running minimum and
// maximum of the valid cells seen, and how many there were. The zero
// value is the identity, which is why count is the empty marker rather
// than a sentinel in mn or mx — no float32 is outside the range a raster
// may hold.
type extent struct {
	mn, mx float32
	count  int64
}

// result turns a finished partial into the values MinMax returns: NaN and
// 0 for an empty reduction, and otherwise the extremes, with any NaN
// replaced by the canonical one.
func (e extent) result() (mn, mx float32, count int64) {
	if e.count == 0 {
		return nan, nan, 0
	}
	return canonicalNaN(e.mn), canonicalNaN(e.mx), e.count
}

var (
	nan    = float32(math.NaN())
	posInf = float32(math.Inf(1))
	negInf = float32(math.Inf(-1))
)

// canonicalNaN replaces any NaN with the canonical quiet NaN.
//
// Go's min and max return a NaN when one of their operands is NaN, but
// not which one's payload, and neither do the vector kernels: folding
// eight lanes and folding one cell at a time can carry different payloads
// out of a raster that holds more than one. Since the payload would
// otherwise depend on the tiling, the worker count and the backend — the
// three things a reduction promises not to depend on — a NaN result is
// canonical instead (DESIGN.md §49). The value is still NaN either way.
func canonicalNaN(v float32) float32 {
	if v != v {
		return nan
	}
	return v
}

// wordBits is how many cells one validity word covers.
const wordBits = 64

// minMaxOp folds the valid cells with Go's builtin min and max, which
// makes NaN absorbing and orders -0 below +0. Both are associative and
// commutative under those semantics, so this reduction meets §49's
// guarantee with no accumulator machinery: tiles, workers and vector
// lanes may split and combine the cells however they like.
type minMaxOp struct{}

func (minMaxOp) Inputs() int { return 1 }

func (minMaxOp) Combine(a *extent, b extent) {
	if b.count == 0 {
		return
	}
	if a.count == 0 {
		*a = b
		return
	}
	a.mn, a.mx = min(a.mn, b.mn), max(a.mx, b.mx)
	a.count += b.count
}

// Fold folds into a local accumulator started at the identity and merges
// it once, so the hot loops never test whether the partial is still
// empty.
func (o minMaxOp) Fold(p *extent, c exec.Cells) {
	acc := extent{mn: posInf, mx: negInf}
	s := c.Src[0]
	switch {
	case c.Masked:
		for y := range c.Height {
			foldMaskedRow(&acc, s.Row(y), c, y)
		}
	case s.Stride == s.Width:
		// One contiguous run of cells: one vector call for the whole
		// rectangle, as package algebra's compact path does.
		n := s.Width * s.Height
		acc.mn = vec.ReduceMin(acc.mn, s.Data[:n])
		acc.mx = vec.ReduceMax(acc.mx, s.Data[:n])
		acc.count = int64(n)
	default:
		for y := range c.Height {
			row := s.Row(y)
			acc.mn = vec.ReduceMin(acc.mn, row)
			acc.mx = vec.ReduceMax(acc.mx, row)
		}
		acc.count = int64(c.Width) * int64(c.Height)
	}
	o.Combine(p, acc)
}

// foldMaskedRow folds the valid cells of one row into acc, a validity
// word at a time: a word with every cell valid goes through the vector
// kernels whole, a word with none is skipped, and the rest are walked bit
// by bit. Data under an invalid cell is unspecified and is never read
// (DESIGN.md §31).
func foldMaskedRow(acc *extent, row []float32, c exec.Cells, y int) {
	for x := 0; x < len(row); x += wordBits {
		k := min(wordBits, len(row)-x)
		m := c.ValidBits(x, y, k)
		if m == 0 {
			continue
		}
		cells := row[x : x+k]
		if m == ^uint64(0)>>uint(wordBits-k) {
			acc.mn = vec.ReduceMin(acc.mn, cells)
			acc.mx = vec.ReduceMax(acc.mx, cells)
			acc.count += int64(k)
			continue
		}
		for m != 0 {
			v := cells[bits.TrailingZeros64(m)]
			acc.mn, acc.mx = min(acc.mn, v), max(acc.mx, v)
			acc.count++
			m &= m - 1
		}
	}
}
