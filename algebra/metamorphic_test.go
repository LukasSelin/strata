package algebra_test

import (
	"context"
	"fmt"
	"math"
	"testing"

	"strata/algebra"
	"strata/engine"
	"strata/internal/fuzzdata"
	"strata/internal/rastertest"
	"strata/raster"
)

// relApply runs the named operation on inputs placed in fresh memory
// layouts, through the plain function, the Tiled one or the Chunked one
// as decoded from d, and returns a compact copy of the result. The output
// has a mask if extraMask is set or an input has one.
func relApply(d *fuzzdata.Reader, name string, lo, hi float32, extraMask bool, inputs ...raster.Float32Raster) raster.Float32Raster {
	masked := extraMask
	placed := make([]raster.Float32Raster, len(inputs))
	for i, in := range inputs {
		masked = masked || in.Valid != nil
		placed[i] = rastertest.Place(d, in)
	}
	w, h := inputs[0].Width, inputs[0].Height
	dst := rastertest.Output(d, w, h, masked, true)
	a := placed[0]
	b := a
	if len(placed) > 1 {
		b = placed[1]
	}
	path := d.IntN(3)
	opts := engine.Options{TileWidth: d.Range(0, w+2), TileHeight: d.Range(0, h+2), Workers: d.Range(0, 3)}
	ctx := context.Background()
	sink, sa, sb := engine.NewMemorySink(dst), engine.NewMemorySource(a), engine.NewMemorySource(b)
	var err error
	switch name {
	case "Add":
		err = pick(path, func() { algebra.Add(dst, a, b) }, func() error { return algebra.AddTiled(ctx, dst, a, b, opts) },
			func() error { return algebra.AddChunked(ctx, sink, sa, sb, opts) })
	case "Sub":
		err = pick(path, func() { algebra.Sub(dst, a, b) }, func() error { return algebra.SubTiled(ctx, dst, a, b, opts) },
			func() error { return algebra.SubChunked(ctx, sink, sa, sb, opts) })
	case "Mul":
		err = pick(path, func() { algebra.Mul(dst, a, b) }, func() error { return algebra.MulTiled(ctx, dst, a, b, opts) },
			func() error { return algebra.MulChunked(ctx, sink, sa, sb, opts) })
	case "Min":
		err = pick(path, func() { algebra.Min(dst, a, b) }, func() error { return algebra.MinTiled(ctx, dst, a, b, opts) },
			func() error { return algebra.MinChunked(ctx, sink, sa, sb, opts) })
	case "Max":
		err = pick(path, func() { algebra.Max(dst, a, b) }, func() error { return algebra.MaxTiled(ctx, dst, a, b, opts) },
			func() error { return algebra.MaxChunked(ctx, sink, sa, sb, opts) })
	case "Clamp":
		err = pick(path, func() { algebra.Clamp(dst, a, lo, hi) }, func() error { return algebra.ClampTiled(ctx, dst, a, lo, hi, opts) },
			func() error { return algebra.ClampChunked(ctx, sink, sa, lo, hi, opts) })
	default:
		panic("unknown operation " + name)
	}
	if err != nil {
		panic(err) // a background context never fails
	}
	return rastertest.Compact(dst)
}

func pick(path int, plain func(), tiled, chunked func() error) error {
	switch path {
	case 0:
		plain()
		return nil
	case 1:
		return tiled()
	default:
		return chunked()
	}
}

// negated returns a compact copy of r with every value negated (the sign
// bit flipped, NaN included).
func negated(r raster.Float32Raster) raster.Float32Raster {
	n := rastertest.Compact(r)
	for i := range n.Data {
		n.Data[i] = -n.Data[i]
	}
	return n
}

// constant returns an unmasked w×h raster of v.
func constant(w, h int, v float32) raster.Float32Raster {
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	for i := range r.Data {
		r.Data[i] = v
	}
	return r
}

// requireRelated fails unless got and want have the same validity in
// every cell and related Data (by same) in every valid cell.
func requireRelated(t *testing.T, id string, got, want raster.Float32Raster, same func(g, w float32) bool) {
	t.Helper()
	for y := range want.Height {
		for x := range want.Width {
			if gv, wv := got.IsValid(x, y), want.IsValid(x, y); gv != wv {
				t.Fatalf("%s: cell (%d, %d) valid = %v, want %v", id, x, y, gv, wv)
			}
			if !want.IsValid(x, y) {
				continue
			}
			if g, w := got.Data[got.Index(x, y)], want.Data[want.Index(x, y)]; !same(g, w) {
				t.Fatalf("%s: cell (%d, %d) = %v (%#08x), want %v (%#08x)", id, x, y, g, math.Float32bits(g), w, math.Float32bits(w))
			}
		}
	}
}

// FuzzAlgebraRelations checks algebraic relations between results, each
// side computed through its own execution path and memory layouts, over
// arbitrary values and masks: commutativity of Add, Mul, Min and Max,
// associativity of Min and Max, Sub(a, b) = -Sub(b, a), Min(a, b) =
// -Max(-a, -b), Add(a, a) = Mul(a, 2), Clamp = Min(Max(x, lo), hi) and
// its idempotence, all with identical validity; that Data under invalid
// cells never reaches a valid result; and that invalidating one input
// cell invalidates exactly that output cell.
func FuzzAlgebraRelations(f *testing.F) {
	for rel := range 9 {
		f.Add([]byte{byte(rel), 0, 33, 3, 1, 5, 6})
		f.Add([]byte{byte(rel), 3, 64, 2, 0, 1, 2, 3})
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		rel, choice := d.IntN(9), d.IntN(4)
		w, h := d.Range(1, 70), d.Range(1, 6)
		extraMask := d.Bool()
		operand := func() raster.Float32Raster {
			r := raster.NewFloat32(w, h, make([]float32, w*h))
			for i := range r.Data {
				r.Data[i] = d.Float32()
			}
			if d.IntN(3) != 0 {
				r.Valid = raster.NewMask(w * h)
				for i := range r.Data {
					raster.MaskSet(r.Valid, i, d.IntN(6) != 0)
				}
			}
			return r
		}
		a, b, c := operand(), operand(), operand()
		lo, hi := d.Float32(), d.Float32()
		apply := func(name string, inputs ...raster.Float32Raster) raster.Float32Raster {
			return relApply(d, name, lo, hi, extraMask, inputs...)
		}
		id := fmt.Sprintf("relation %d choice %d %d×%d", rel, choice, w, h)

		switch rel {
		case 0:
			op := []string{"Add", "Mul", "Min", "Max"}[choice]
			requireRelated(t, id+": "+op+" commutes", apply(op, b, a), apply(op, a, b), rastertest.SameFloat)
		case 1:
			op := []string{"Min", "Max"}[choice%2]
			requireRelated(t, id+": "+op+" associates", apply(op, a, apply(op, b, c)), apply(op, apply(op, a, b), c), rastertest.SameFloat)
		case 2:
			requireRelated(t, id+": Sub(a, b) = -Sub(b, a)", negated(apply("Sub", b, a)), apply("Sub", a, b), rastertest.SameValue)
		case 3:
			requireRelated(t, id+": Min(a, b) = -Max(-a, -b)", negated(apply("Max", negated(a), negated(b))), apply("Min", a, b), rastertest.SameFloat)
			requireRelated(t, id+": Max(a, b) = -Min(-a, -b)", negated(apply("Min", negated(a), negated(b))), apply("Max", a, b), rastertest.SameFloat)
		case 4:
			requireRelated(t, id+": Add(a, a) = Mul(a, 2)", apply("Mul", a, constant(w, h, 2)), apply("Add", a, a), rastertest.SameFloat)
		case 5:
			requireRelated(t, fmt.Sprintf("%s: Clamp(x, %v, %v) = Min(Max(x, lo), hi)", id, lo, hi),
				apply("Min", apply("Max", a, constant(w, h, lo)), constant(w, h, hi)), apply("Clamp", a), rastertest.SameFloat)
		case 6:
			once := apply("Clamp", a)
			requireRelated(t, fmt.Sprintf("%s: Clamp(Clamp(x, %v, %v)) = Clamp(x)", id, lo, hi), apply("Clamp", once), once, rastertest.SameFloat)
		case 7:
			op := []string{"Add", "Sub", "Mul", "Min", "Max", "Clamp"}[d.IntN(6)]
			sa, sb := rastertest.Compact(a), rastertest.Compact(b)
			rastertest.ScrambleInvalid(sa, d)
			rastertest.ScrambleInvalid(sb, d)
			requireRelated(t, id+": "+op+" with scrambled invalid cells", apply(op, sa, sb), apply(op, a, b), rastertest.SameFloat)
		case 8:
			op := []string{"Add", "Sub", "Mul", "Min", "Max", "Clamp"}[d.IntN(6)]
			px, py := d.IntN(w), d.IntN(h)
			holed := rastertest.Compact(a)
			if holed.Valid == nil {
				holed.Valid = raster.NewMask(w * h)
				a = rastertest.Compact(holed) // the same cells, all valid
			}
			holed.SetValid(px, py, false)
			before, after := apply(op, a, b), apply(op, holed, b)
			for y := range h {
				for x := range w {
					want := before.IsValid(x, y) && (x != px || y != py)
					if got := after.IsValid(x, y); got != want {
						t.Fatalf("%s: %s with input cell (%d, %d) invalid: cell (%d, %d) valid = %v, want %v", id, op, px, py, x, y, got, want)
					}
					if want && !rastertest.SameFloat(after.Data[after.Index(x, y)], before.Data[before.Index(x, y)]) {
						t.Fatalf("%s: %s with input cell (%d, %d) invalid: valid cell (%d, %d) changed", id, op, px, py, x, y)
					}
				}
			}
		}
	})
}
