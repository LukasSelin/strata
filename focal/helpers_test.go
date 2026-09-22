package focal_test

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/internal/fuzzdata"
	"github.com/LukasSelin/strata/internal/rastertest"
	"github.com/LukasSelin/strata/raster"
)

// The operations under test, by number.
const (
	kCorrelate = iota
	kConvolve
	kSeparable
	kMean
	kMin
	kMax
	numKinds
)

var kindNames = [numKinds]string{"Correlate", "Convolve", "CorrelateSeparable", "Mean", "Min", "Max"}

// spec is one operation with its options.
type spec struct {
	kind     int
	r        int
	w        []float32 // Correlate and Convolve
	row, col []float32 // CorrelateSeparable
}

func (s spec) String() string {
	switch s.kind {
	case kCorrelate, kConvolve:
		return fmt.Sprintf("%s r=%d w=%v", kindNames[s.kind], s.r, s.w)
	case kSeparable:
		return fmt.Sprintf("%s r=%d row=%v col=%v", kindNames[s.kind], s.r, s.row, s.col)
	}
	return fmt.Sprintf("%s r=%d", kindNames[s.kind], s.r)
}

func (s spec) size() int { return 2*s.r + 1 }

// weightValues says how newSpec draws weights: integers from -3 to 3,
// which keep sums of small integers exact, or finite floats.
type weightValues int

const (
	intWeights weightValues = iota
	floatWeights
)

// newSpec draws the weights of an operation of the given kind and radius.
func newSpec(d fuzzdata.Source, kind, r int, wv weightValues) spec {
	s := spec{kind: kind, r: r}
	k := s.size()
	draw := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			if wv == intWeights {
				v[i] = float32(d.Range(-3, 3))
				continue
			}
			x := d.Float32()
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				x = float32(d.Range(-100, 100)) / 8
			}
			v[i] = x
		}
		return v
	}
	switch kind {
	case kCorrelate, kConvolve:
		s.w = draw(k * k)
	case kSeparable:
		s.row, s.col = draw(k), draw(k)
	}
	return s
}

// run runs s from src into dst through the plain function (path 0), the
// Tiled one (1) or the Chunked one over memory sources and sinks (2).
func (s spec) run(path int, eopts engine.Options, dst, src raster.Float32Raster) error {
	ctx := context.Background()
	wo := focal.WeightsOptions{Radius: s.r, Weights: s.w}
	so := focal.SeparableOptions{Radius: s.r, Row: s.row, Col: s.col}
	bo := focal.BoxOptions{Radius: s.r}
	if path == 2 {
		sink, source := engine.NewMemorySink(dst), engine.NewMemorySource(src)
		switch s.kind {
		case kCorrelate:
			return focal.CorrelateChunked(ctx, sink, source, wo, eopts)
		case kConvolve:
			return focal.ConvolveChunked(ctx, sink, source, wo, eopts)
		case kSeparable:
			return focal.CorrelateSeparableChunked(ctx, sink, source, so, eopts)
		case kMean:
			return focal.MeanChunked(ctx, sink, source, bo, eopts)
		case kMin:
			return focal.MinChunked(ctx, sink, source, bo, eopts)
		default:
			return focal.MaxChunked(ctx, sink, source, bo, eopts)
		}
	}
	if path == 1 {
		switch s.kind {
		case kCorrelate:
			return focal.CorrelateTiled(ctx, dst, src, wo, eopts)
		case kConvolve:
			return focal.ConvolveTiled(ctx, dst, src, wo, eopts)
		case kSeparable:
			return focal.CorrelateSeparableTiled(ctx, dst, src, so, eopts)
		case kMean:
			return focal.MeanTiled(ctx, dst, src, bo, eopts)
		case kMin:
			return focal.MinTiled(ctx, dst, src, bo, eopts)
		default:
			return focal.MaxTiled(ctx, dst, src, bo, eopts)
		}
	}
	switch s.kind {
	case kCorrelate:
		focal.Correlate(dst, src, wo)
	case kConvolve:
		focal.Convolve(dst, src, wo)
	case kSeparable:
		focal.CorrelateSeparable(dst, src, so)
	case kMean:
		focal.Mean(dst, src, bo)
	case kMin:
		focal.Min(dst, src, bo)
	default:
		focal.Max(dst, src, bo)
	}
	return nil
}

// naiveCell computes s at interior cell (x, y) of src one term at a
// time, in the documented order, rounding every operation to float32. It
// shares no code with the kernels.
func (s spec) naiveCell(src raster.Float32Raster, x, y int) float32 {
	k, r := s.size(), s.r
	v := func(c, j int) float32 { return src.Data[src.Index(x+c-r, y+j-r)] }
	switch s.kind {
	case kCorrelate, kConvolve:
		w := s.w
		if s.kind == kConvolve {
			w = rot180(w)
		}
		acc := float32(w[0] * v(0, 0))
		for t := 1; t < k*k; t++ {
			p := float32(w[t] * v(t%k, t/k))
			acc = float32(acc + p)
		}
		return acc
	case kSeparable:
		var acc float32
		for c := range k {
			col := float32(s.col[0] * v(c, 0))
			for j := 1; j < k; j++ {
				p := float32(s.col[j] * v(c, j))
				col = float32(col + p)
			}
			p := float32(s.row[c] * col)
			if c == 0 {
				acc = p
			} else {
				acc = float32(acc + p)
			}
		}
		return acc
	case kMean:
		var acc float32
		for c := range k {
			col := v(c, 0)
			for j := 1; j < k; j++ {
				col = float32(col + v(c, j))
			}
			if c == 0 {
				acc = col
			} else {
				acc = float32(acc + col)
			}
		}
		return float32(acc / float32(k*k))
	default:
		acc := v(0, 0)
		for j := range k {
			for c := range k {
				if s.kind == kMin {
					acc = min(acc, v(c, j))
				} else {
					acc = max(acc, v(c, j))
				}
			}
		}
		return acc
	}
}

func rot180(w []float32) []float32 {
	out := make([]float32, len(w))
	for i, x := range w {
		out[len(w)-1-i] = x
	}
	return out
}

// interior reports whether (x, y) has a whole neighbourhood of radius r
// in a w×h raster.
func interior(x, y, w, h, r int) bool {
	return x >= r && y >= r && x < w-r && y < h-r
}

// naiveValid is the documented validity of output cell (x, y): interior,
// and every cell of the neighbourhood valid.
func naiveValid(src raster.Float32Raster, x, y, r int) bool {
	if !interior(x, y, src.Width, src.Height, r) {
		return false
	}
	if src.Valid == nil {
		return true
	}
	for j := -r; j <= r; j++ {
		for c := -r; c <= r; c++ {
			if !src.IsValid(x+c, y+j) {
				return false
			}
		}
	}
	return true
}

// requireNaive checks got, the output of s on src, against the
// definition: NaN and invalid at the edges, the documented validity, and
// the naive value, bit for bit, in every valid cell (every interior cell
// when neither has a mask).
func requireNaive(t rastertest.TB, id string, s spec, got, src raster.Float32Raster) {
	t.Helper()
	w, h := src.Width, src.Height
	for y := range h {
		for x := range w {
			g := got.Data[got.Index(x, y)]
			valid := naiveValid(src, x, y, s.r)
			if got.Valid != nil && got.IsValid(x, y) != valid {
				t.Fatalf("%s: %v at (%d, %d): valid = %v, want %v", id, s, x, y, got.IsValid(x, y), valid)
			}
			if !interior(x, y, w, h, s.r) {
				if g == g {
					t.Fatalf("%s: %v at edge cell (%d, %d): %v, want NaN", id, s, x, y, g)
				}
				continue
			}
			if !valid {
				continue
			}
			if want := s.naiveCell(src, x, y); !rastertest.SameFloat(g, want) {
				t.Fatalf("%s: %v at (%d, %d) of %d×%d: %v (%#x), want %v (%#x)",
					id, s, x, y, w, h, g, math.Float32bits(g), want, math.Float32bits(want))
			}
		}
	}
}

// requireSame checks that got and want hold the same validity and the
// same bits in every valid or edge cell.
func requireSame(t rastertest.TB, id string, got, want raster.Float32Raster) {
	t.Helper()
	for y := range want.Height {
		for x := range want.Width {
			gv, wv := got.Valid == nil || got.IsValid(x, y), want.Valid == nil || want.IsValid(x, y)
			if gv != wv {
				t.Fatalf("%s: (%d, %d) valid = %v, want %v", id, x, y, gv, wv)
			}
			g, w := got.Data[got.Index(x, y)], want.Data[want.Index(x, y)]
			if wv && !rastertest.SameFloat(g, w) {
				t.Fatalf("%s: (%d, %d) = %v (%#x), want %v (%#x)", id, x, y, g, math.Float32bits(g), w, math.Float32bits(w))
			}
		}
	}
}

// cellValues says how newSrc draws cells.
type cellValues int

const (
	anyValues     cellValues = iota // any float32 bits: NaN, ±Inf, ±0, subnormals
	smoothValues                    // terrain-like, finite
	integerValues                   // small integers, so sums stay exact
)

// newSrc returns a w×h input with cells drawn as vals and, if masked,
// about one cell in ten invalid, laid out as rastertest.Place decodes.
func newSrc(d fuzzdata.Source, w, h int, vals cellValues, masked bool) raster.Float32Raster {
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	for i := range r.Data {
		switch vals {
		case anyValues:
			if d.IntN(3) == 0 {
				r.Data[i] = d.Float32()
			} else {
				r.Data[i] = float32(d.Range(-4000, 4000)) / 7
			}
		case smoothValues:
			x, y := float64(i%w), float64(i/w)
			r.Data[i] = float32(500 + 40*math.Sin(x/5) + 30*math.Cos(y/3) + float64(d.Range(-100, 100))/10)
		default:
			r.Data[i] = float32(d.Range(-50, 50))
		}
	}
	if masked {
		r.Valid = raster.NewMask(w * h)
		for i := range w * h {
			if d.IntN(10) == 0 {
				raster.MaskSet(r.Valid, i, false)
			}
		}
	}
	return rastertest.Place(d, r)
}

// mustPanic runs f and requires a panic whose message starts with prefix.
func mustPanic(t rastertest.TB, id, prefix string, f func()) {
	t.Helper()
	defer func() {
		t.Helper()
		r := recover()
		s, ok := r.(string)
		if !ok {
			if e, isErr := r.(error); isErr {
				s, ok = e.Error(), true
			}
		}
		if !ok || !strings.HasPrefix(s, prefix) {
			t.Fatalf("%s: panic %v, want a %q message", id, r, prefix)
		}
	}()
	f()
}
