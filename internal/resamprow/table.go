// Package resamprow holds the tap tables and row kernels behind package
// resample (DESIGN.md §54): per-axis tables that map each output column
// or row to a run of source cells and their weights, and the two passes
// of the separable filter that consume them, a horizontal pass over the
// source rows of a footprint and a vertical pass down its columns.
//
// It is a kernel package (§12, §14): it takes spans and scalars, and
// knows nothing of rasters, masks or workers. Package resamp drives it
// over a raster band.
//
//strata:kernel
package resamprow

import (
	"fmt"
	"math"
)

// Method is a resampling method.
type Method uint8

// The methods, in the order package resample exports them.
const (
	Nearest Method = iota
	Bilinear
	Cubic
	Lanczos
	Average
)

func (m Method) String() string {
	switch m {
	case Nearest:
		return "Nearest"
	case Bilinear:
		return "Bilinear"
	case Cubic:
		return "Cubic"
	case Lanczos:
		return "Lanczos"
	case Average:
		return "Average"
	}
	return fmt.Sprintf("Method(%d)", uint8(m))
}

// Spec is one axis of a resampling: n output cells of resolution Res from
// Origin, over SrcN source cells of resolution SrcRes from SrcOrigin.
type Spec struct {
	N           int
	Origin, Res float64

	SrcN              int
	SrcOrigin, SrcRes float64
}

// Axis is the tap table of one axis. For output index c, the taps are the
// source indices First[c] … First[c]+Taps[c]-1, in increasing order, with
// weights W[Off[c] : Off[c]+Taps[c]].
//
// Only [Lo, Hi) is covered: the output cells whose centre lies inside the
// source (Nearest, Bilinear, Cubic, Lanczos) or that overlap it
// (Average). Outside it an output cell is invalid on this axis, and its
// entries are zero. Taps[c] >= 1 inside it.
type Axis struct {
	Lo, Hi int

	First []int32
	Taps  []int32
	Off   []int32
	W     []float32

	// Centre is floor(u), the source cell under output cell c's centre,
	// for c in [Lo, Hi). It is -1 elsewhere, and for Average, whose
	// validity does not depend on a centre cell.
	Centre []int32
	// Clipped reports that output cell c lost a tap of non-zero weight
	// to the edge of the source.
	Clipped []bool
	// WinFirst and WinN are the source cells the kernel reaches, zero
	// weights included, clipped to the source: the cells gdalwarp's
	// half-valid rule for a widened Lanczos counts (see Plan.HalfValid).
	WinFirst, WinN []int32

	// Exact reports that output cell c's centre lies exactly on a source
	// cell's centre, so its kernel reads that cell alone.
	Exact []bool

	// Widened reports that the kernel was widened for downsampling.
	Widened bool
	// MaxTaps is the largest Taps[c], MaxWin the largest WinN[c].
	MaxTaps, MaxWin int
}

// Kernel supports: the half-width of each method's kernel in source
// cells before widening.
const (
	bilinearSupport = 1
	cubicSupport    = 2
	lanczosSupport  = 3
)

// widenThreshold is gdalwarp's: bilinear and cubic widen once the output
// has fewer than 0.95 output cells per source cell, that is when s >
// 1/0.95 (measured, DESIGN.md §54). Lanczos widens for any s > 1.
const widenThreshold = 0.95

// maxAxisCells bounds the table sizes an axis may produce, so that a
// pathological grid panics instead of exhausting memory.
const maxAxisCells = math.MaxInt32

// NewAxis builds the table of one axis. It panics on a non-positive cell
// count, or on a resolution or origin that is zero (resolutions only) or
// not finite.
func NewAxis(m Method, sp Spec) Axis {
	return newAxis(m, sp, false)
}

// Axes is the two axes of a resampling. The row kernels take one axis
// at a time; the direct reference and the footprint take the pair, and
// package resamp's Plan is this plus the method's rules.
type Axes struct {
	X, Y Axis
}

// NewAxes builds both axes of a resampling.
func NewAxes(m Method, x, y Spec) Axes {
	return Axes{X: NewAxis(m, x), Y: NewAxis(m, y)}
}

// Footprint returns the source window [fx0, fx1) x [fy0, fy1) that output
// cells [x0, x1) x [y0, y1) read, empty if none of them is covered. With
// win it also covers the cells the half-valid rule counts.
func (a *Axes) Footprint(x0, y0, x1, y1 int, win bool) (fx0, fy0, fx1, fy1 int) {
	fx0, fx1 = a.X.Footprint(x0, x1, win)
	fy0, fy1 = a.Y.Footprint(y0, y1, win)
	if fx0 >= fx1 || fy0 >= fy1 {
		return 0, 0, 0, 0
	}
	return fx0, fy0, fx1, fy1
}

// NewBilinear4 is the bilinear table gdalwarp's four-sample cubic falls
// back to: never widened, whatever the scale. Package resamp builds it
// alongside a cubic plan's own axes.
func NewBilinear4(sp Spec) Axis { return newAxis(Bilinear, sp, true) }

func newAxis(m Method, sp Spec, noWiden bool) Axis {
	checkSpec(sp)
	a := Axis{
		First:    make([]int32, sp.N),
		Taps:     make([]int32, sp.N),
		Off:      make([]int32, sp.N),
		Centre:   make([]int32, sp.N),
		Clipped:  make([]bool, sp.N),
		WinFirst: make([]int32, sp.N),
		WinN:     make([]int32, sp.N),
		Exact:    make([]bool, sp.N),
		Lo:       -1,
	}
	inv1 := 1 / sp.SrcRes
	inv0 := -sp.SrcOrigin * inv1
	// scale is source cells per output cell along this axis.
	scale := math.Abs(sp.Res / sp.SrcRes)
	s := 1.0
	switch m {
	case Bilinear, Cubic:
		if 1/scale < widenThreshold {
			s = scale
		}
	case Lanczos:
		if scale > 1 {
			s = scale
		}
	}
	if noWiden {
		s = 1
	}
	a.Widened = s > 1
	support := 0.0
	var kernel func(float64) float64
	switch m {
	case Bilinear:
		support, kernel = bilinearSupport, tent
	case Cubic:
		support, kernel = cubicSupport, keys
	case Lanczos:
		support, kernel = lanczosSupport, lanczos3
	}
	// Presize W for the most taps a cell can have, so building it does
	// not grow it by appending.
	// Computed in float64 and capped by the source, so a huge scale
	// neither overflows nor reserves more than the source holds.
	perCell := 1.0
	switch m {
	case Average:
		perCell = math.Ceil(scale) + 1
	case Bilinear, Cubic, Lanczos:
		perCell = math.Ceil(2*support*s) + 1
	}
	perCell = min(perCell, float64(sp.SrcN+1), float64(maxAxisCells/sp.N))
	a.W = make([]float32, 0, sp.N*int(perCell))
	var w64 []float64
	for c := range sp.N {
		a.Centre[c] = -1
		// Products go through mul, so no compiler fuses them into a
		// multiply-add and the tables, ties included, are the same bits on
		// every architecture.
		x := sp.Origin + mul(float64(c)+0.5, sp.Res)
		u := inv0 + mul(x, inv1)
		var first, taps int
		switch m {
		case Nearest:
			ci, ok := cellOf(u, sp.SrcN)
			if !ok {
				continue
			}
			a.Centre[c] = int32(ci) // #nosec G115 -- ci < SrcN < 2³¹ (checkSpec)
			first, taps = ci, 1
			w64 = append(w64[:0], 1)
		case Average:
			// The output cell's edges in source pixel coordinates.
			e0 := inv0 + mul(sp.Origin+mul(float64(c), sp.Res), inv1)
			e1 := inv0 + mul(sp.Origin+mul(float64(c+1), sp.Res), inv1)
			lo, hi := math.Min(e0, e1), math.Max(e0, e1)
			if sl, sh := snap(lo, 0), snap(hi, 0); sh > sl {
				lo, hi = sl, sh
			}
			if hi > lo {
				first, taps, w64 = overlaps(lo, hi, sp.SrcN, w64[:0])
			} else {
				// A cell too small for float64 to separate its edges is a
				// point: it takes the source cell it lies in.
				ci, ok := cellOf(lo, sp.SrcN)
				first, taps, w64 = ci, 0, append(w64[:0], 1)
				if ok {
					taps = 1
				}
			}
			if taps == 0 {
				continue
			}
		default:
			ci, ok := cellOf(u, sp.SrcN)
			if !ok {
				continue
			}
			a.Centre[c] = int32(ci) // #nosec G115 -- ci < SrcN < 2³¹ (checkSpec)
			var win winRange
			us := snap(u, 0.5)
			a.Exact[c] = us-0.5 == math.Floor(us)
			first, taps, w64, a.Clipped[c], win = kernelTaps(kernel, support, s, us, sp.SrcN, w64[:0])
			a.WinFirst[c], a.WinN[c] = win.first, win.inside
		}
		if a.Lo < 0 {
			a.Lo = c
		}
		a.Hi = c + 1
		a.First[c] = int32(first)  // #nosec G115 -- first < SrcN < 2³¹ (checkSpec)
		a.Taps[c] = int32(taps)    // #nosec G115 -- taps <= SrcN
		a.Off[c] = int32(len(a.W)) // #nosec G115 -- len(W) <= maxAxisCells, checked below
		a.W = appendNormalised(a.W, w64)
		a.MaxTaps = max(a.MaxTaps, taps)
		a.MaxWin = max(a.MaxWin, int(a.WinN[c]))
		if len(a.W) > maxAxisCells {
			panic(fmt.Sprintf("resamp: axis needs more than %d taps", maxAxisCells))
		}
	}
	if a.Lo < 0 {
		a.Lo, a.Hi = 0, 0
	}
	return a
}

func checkSpec(sp Spec) {
	if sp.N <= 0 || sp.SrcN <= 0 || sp.N > math.MaxInt32 || sp.SrcN > math.MaxInt32 {
		panic(fmt.Sprintf("resamp: axis sizes must be in [1, 2³¹), got %d from %d", sp.N, sp.SrcN))
	}
	for _, v := range []float64{sp.Origin, sp.Res, sp.SrcOrigin, sp.SrcRes} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			panic(fmt.Sprintf("resamp: axis %+v is not finite", sp))
		}
	}
	if sp.Res == 0 || sp.SrcRes == 0 {
		panic(fmt.Sprintf("resamp: axis %+v has a zero resolution", sp))
	}
}

// snapTolerance is how close, in source cells, a coordinate must be to a
// cell centre or edge to be taken as exactly on it when computing
// weights. It removes the rounding noise of u, such as the 8e-16 that
// would give an identity grid of resolution 0.3 a second tap, and moves
// no weight by more than 1e-9.
const snapTolerance = 1e-9

// snap returns v moved onto the nearest k+frac if it is within
// snapTolerance of it. Weights use the snapped coordinate; the centre
// cell, which decides validity and Nearest, uses u as computed, so that
// its ties fall the way gdalwarp's do.
func snap(v, frac float64) float64 {
	r := math.Round(v-frac) + frac
	if math.Abs(v-r) <= snapTolerance {
		return r
	}
	return v
}

// cellOf returns floor(u) and whether it is a source cell.
func cellOf(u float64, n int) (int, bool) {
	f := math.Floor(u)
	if !(f >= 0 && f < float64(n)) {
		return 0, false
	}
	return int(f), true
}

// winRange is the cells a kernel reaches inside the source: inside of
// them from first.
type winRange struct{ first, inside int32 }

// kernelTaps returns the in-source taps of kernel k, of half-width
// support, widened by s and centred at u: their first index, count and
// unnormalised weights, whether a tap of non-zero weight fell outside the
// source, and the cells the kernel reaches.
func kernelTaps(k func(float64) float64, support, s, u float64, n int, w []float64) (first, taps int, _ []float64, clipped bool, win winRange) {
	reach := mul(support, s)
	// Cells beyond the source contribute only to clipped, which only the
	// unstretched four-sample Cubic uses (and whose reach is 2), so a
	// stretched kernel visits the source and one cell either side, not a
	// reach that a huge scale makes astronomically wide.
	lof := math.Floor(u - 0.5 - reach)
	hif := math.Ceil(u - 0.5 + reach)
	if s > 1 {
		lof, hif = math.Max(lof, -1), math.Min(hif, float64(n))
	}
	lo, hi := int(lof), int(hif)
	first = -1
	for i := lo; i <= hi; i++ {
		d := (float64(i) + 0.5 - u) / s
		if !(math.Abs(d) < support) {
			continue
		}
		if i >= 0 && i < n {
			if win.inside == 0 {
				win.first = int32(i) // #nosec G115 -- 0 <= i < SrcN < 2³¹
			}
			win.inside++
		}
		wt := k(d)
		if wt == 0 {
			continue
		}
		if i < 0 || i >= n {
			clipped = true
			continue
		}
		if first < 0 {
			first = i
		}
		// Zero-weight cells inside a run are kept as taps, so a run is
		// contiguous; only its ends are trimmed.
		for len(w) < i-first {
			w = append(w, 0)
		}
		w = append(w, wt)
	}
	return first, len(w), w, clipped, win
}

// overlaps returns the source cells overlapping [lo, hi) in pixel
// coordinates, clipped to the source, and their overlap lengths.
func overlaps(lo, hi float64, n int, w []float64) (first, taps int, _ []float64) {
	i0 := max(0, int(math.Floor(lo)))
	i1 := min(n, int(math.Ceil(hi)))
	first = -1
	for i := i0; i < i1; i++ {
		o := math.Min(hi, float64(i+1)) - math.Max(lo, float64(i))
		if !(o > 0) {
			if first < 0 {
				continue
			}
			o = 0
		}
		if first < 0 {
			first = i
		}
		w = append(w, o)
	}
	for len(w) > 0 && w[len(w)-1] == 0 {
		w = w[:len(w)-1]
	}
	if len(w) == 0 {
		return 0, 0, w
	}
	return first, len(w), w
}

// appendNormalised appends w divided by its sum, rounded to float32.
func appendNormalised(dst []float32, w []float64) []float32 {
	sum := 0.0
	for _, v := range w {
		sum += v
	}
	for _, v := range w {
		dst = append(dst, float32(v/sum))
	}
	return dst
}

// mul is a*b, rounded: a call the compiler does not inline, so the
// product can never be fused with an addition into a multiply-add, which
// arm64 and amd64 with FMA would otherwise do in different places. A
// float64(...) conversion does not prevent it for float64 operands.
//
//go:noinline
func mul(a, b float64) float64 { return a * b }

// tent is the bilinear kernel.
func tent(x float64) float64 {
	return 1 - math.Abs(x)
}

// keys is Keys' cubic convolution kernel with a = -0.5, GDAL's cubic.
func keys(x float64) float64 {
	const a = -0.5
	x = math.Abs(x)
	if x < 1 {
		return mul(mul(mul(a+2, x)-(a+3), x), x) + 1
	}
	return mul(mul(mul(a, x)-5*a, x)+8*a, x) - 4*a
}

// lanczos3 is the Lanczos kernel with a = 3. It is exactly zero at the
// non-zero integers, where sin(πx) would otherwise leave rounding noise.
func lanczos3(x float64) float64 {
	if x == 0 {
		return 1
	}
	if x == math.Trunc(x) {
		return 0
	}
	px := math.Pi * x
	return mul(3*math.Sin(px), math.Sin(px/3)) / mul(px, px)
}

// Footprint returns the source range [first, end) that output indices
// [lo, hi) of a read, or (0, 0) if none of them is covered. With win, it
// also covers the cells the half-valid rule counts.
func (a *Axis) Footprint(lo, hi int, win bool) (first, end int) {
	lo, hi = max(lo, a.Lo), min(hi, a.Hi)
	if lo >= hi {
		return 0, 0
	}
	first, end = math.MaxInt, 0
	for c := lo; c < hi; c++ {
		f := int(a.First[c])
		first = min(first, f)
		end = max(end, f+int(a.Taps[c]))
		if win {
			f = int(a.WinFirst[c])
			first = min(first, f)
			end = max(end, f+int(a.WinN[c]))
		}
	}
	return first, end
}
