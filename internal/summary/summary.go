// Package summary is the reducer behind reduce.Stats: one fold that
// gathers the count, smallest and largest valid cell, and the exact sum
// and sum of squares, so that a mean and a standard deviation come out of
// the same pass as the range (DESIGN.md §49).
//
// It is internal, and separate from package reduce, so that other
// packages can fold something other than a raster with it — package
// terrain's statistics fold a kernel's output band by band, without
// writing it out — and still return reduce's Summary: the two Summary
// types have the same fields, so one converts to the other.
package summary

import (
	"math"
	"math/bits"

	"github.com/LukasSelin/strata/internal/accum"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/vec"
)

// Summary is what a reduction of the valid cells gives. See
// reduce.Summary for what each field holds.
type Summary struct {
	Count  int64
	Sum    float64
	Mean   float64
	StdDev float64
	Min    float32
	Max    float32
}

// Partial is the running state of a Summary reduction. The zero value is
// empty, and the identity of Reducer.Combine.
type Partial struct {
	m accum.Moments
	// mn and mx are the extremes of the cells folded so far, meaningful
	// only once m holds any: no float32 is outside what a raster may
	// hold, so the empty marker is the count rather than a sentinel.
	mn, mx float32
	runs   Runs
}

// Summary returns the finished reduction. With no cells, Count and Sum
// are 0 and the rest NaN. Every NaN is the canonical quiet NaN.
func (p *Partial) Summary() Summary {
	n := p.m.Count()
	s := Summary{Count: n, Sum: p.m.Sum(), Mean: p.m.Mean(), StdDev: p.m.StdDev(), Min: nan, Max: nan}
	if n > 0 {
		s.Min, s.Max = canonicalNaN(p.mn), canonicalNaN(p.mx)
	}
	return s
}

// Reducer folds one raster into a Partial. It is an exec.Reducer.
type Reducer struct{}

// Inputs is 1.
func (Reducer) Inputs() int { return 1 }

// Combine folds b into a. The count, extremes and exact moments are each
// associative and commutative, so the result does not depend on the
// order the engine combines partials in.
func (Reducer) Combine(a *Partial, b Partial) {
	if b.m.Count() == 0 {
		return
	}
	if a.m.Count() == 0 {
		*a = b
		return
	}
	a.mn, a.mx = min(a.mn, b.mn), max(a.mx, b.mx)
	a.m.Combine(&b.m)
}

// Fold folds the valid cells of c into p: the moments straight into p,
// the extremes into locals merged once, so the hot loops never ask
// whether p is still empty. Data under an invalid cell is never read.
func (Reducer) Fold(p *Partial, c exec.Cells) {
	before := p.m.Count()
	mn, mx := posInf, negInf
	p.runs.Each(c, func(cells []float32) {
		p.m.Add(cells)
		mn, mx = vec.ReduceMin(mn, cells), vec.ReduceMax(mx, cells)
	})
	switch {
	case p.m.Count() == before:
		// nothing valid in this band
	case before == 0:
		p.mn, p.mx = mn, mx
	default:
		p.mn, p.mx = min(p.mn, mn), max(p.mx, mx)
	}
}

// Runs hands a fold the valid cells of a band in runs long enough for
// the vector kernels. Its buffer belongs to a partial, so that a band,
// which the engine folds on the partial's own worker, allocates nothing.
type Runs struct {
	buf [runCells]float32
}

// runCells is the buffer's size: several 64-cell blocks of accum's vector
// backend, so that cells packed from partly valid words mostly reach it
// in whole blocks.
const runCells = 256

// Each calls add with every valid cell of c's first input exactly once,
// in runs, in no particular order: the whole rectangle when it is compact
// and unmasked, a row when it is not compact, and with a mask each
// stretch of wholly valid 64-cell words as it lies in the row. The valid
// cells of partly valid words are packed into the buffer and handed over
// when it fills, and at the end. Data under an invalid cell is never
// read. add must not keep the slice.
func (r *Runs) Each(c exec.Cells, add func(cells []float32)) {
	s := c.Src[0]
	switch {
	case !c.Masked && s.Stride == s.Width:
		add(s.Data[:s.Width*s.Height])
		return
	case !c.Masked:
		for y := range c.Height {
			add(s.Row(y))
		}
		return
	}
	n := 0
	for y := range c.Height {
		row := s.Row(y)
		start := -1 // first cell of the current stretch of whole words
		for x := 0; x < len(row); x += wordBits {
			k := min(wordBits, len(row)-x)
			m := c.ValidBits(x, y, k)
			if m == ^uint64(0)>>uint(wordBits-k) {
				if start < 0 {
					start = x
				}
				continue
			}
			if start >= 0 {
				add(row[start:x])
				start = -1
			}
			cells := row[x : x+k]
			for ; m != 0; m &= m - 1 {
				r.buf[n] = cells[bits.TrailingZeros64(m)&(wordBits-1)]
				n++
			}
			if n > runCells-wordBits {
				add(r.buf[:n])
				n = 0
			}
		}
		if start >= 0 {
			add(row[start:])
		}
	}
	if n > 0 {
		add(r.buf[:n])
	}
}

// wordBits is how many cells one validity word covers.
const wordBits = 64

var (
	nan    = float32(math.NaN())
	posInf = float32(math.Inf(1))
	negInf = float32(math.Inf(-1))
)

// canonicalNaN replaces any NaN with the canonical quiet NaN, since which
// payload survives min and max depends on the order they were folded in
// (DESIGN.md §49).
func canonicalNaN(v float32) float32 {
	if v != v {
		return nan
	}
	return v
}
