package terrain

import (
	"context"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/fuzzdata"
	"github.com/LukasSelin/strata/internal/rastertest"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// The metamorphic tests run an operation on related DEMs and check that
// the results relate as the mathematics says. Horn's weighted differences
// have a fixed evaluation order, shared by the SIMD kernels, so most
// relations hold bit for bit:
//
//   - A mirror or transposition of the grid (with the cell sizes swapped)
//     negates or swaps dx and dy exactly: the sums read the same cells in
//     a commuted order, and B - A is exactly -(A - B). Slope, which squares
//     them, is bitwise invariant, and so is Hillshade under a horizontal
//     mirror with the azimuth negated (sin is odd and cos even, exactly).
//   - Adding an integer constant to an integer DEM small enough that
//     every sum is exact changes no difference, so no result.
//   - Scaling a DEM by a power of two and ZFactor (or dividing the cell
//     sizes) by the same power scales dx exactly and the factor exactly.
//   - Negating a DEM is negating ZFactor.
//   - A DEM whose rows are all the same has dy exactly +0, so CellSizeY
//     changes no result, and likewise CellSize for identical columns.
//     Symmetries cannot see a kernel that uses each axis's cell size for
//     the other, since transposing swaps the sizes back; this relation
//     can.
//   - Slope, Aspect and Hillshade are functions of Gradient's dx and dy,
//     and Curvature is its documented formula of the DEM's cells.
//   - Curvature's Zevenbergen–Thorne derivatives map like dx and dy
//     under the symmetries, so a mirror leaves it bitwise unchanged; a
//     transposition reorders its sums and holds up to rounding. Scaling
//     the cell sizes by a power of two scales it by the inverse power,
//     exactly.
//   - Ruggedness reads only the DEM's cells, and has no cell size or
//     ZFactor. Adding an integer constant to a small integer DEM changes
//     no result, TPI's included (every value it rounds is exact), and
//     scaling by a power of two scales every result by it exactly.
//     Negating the DEM leaves the TRIs and Roughness bitwise unchanged and
//     negates TPI. Roughness is bitwise invariant under every symmetry;
//     the others sum the neighbours in row-major order, so a symmetry
//     reorders the sum and holds up to its rounding.
//   - An output cell depends only on the DEM cells, Data and validity, in
//     its 3×3 neighbourhood, and on nothing under invalid cells.
//
// Aspect and Hillshade under the other symmetries are exact only up to
// their arctangent and float32 rounding, and are checked with tolerances
// from the documented error bounds.

// relOp is a terrain operation with its options.
type relOp struct {
	kind              int // 0 gradient, 1 slope, 2 aspect, 3 hillshade, 4 curvature, 5 ruggedness
	cs, csy, z        float64
	curv              CurvatureType
	rug               RuggednessType
	units             SlopeUnits
	zeroFlat, trig    bool
	azimuth, altitude float64
}

func (o relOp) String() string {
	name := [...]string{"gradient", "slope", "aspect", "hillshade", "curvature", "ruggedness"}[o.kind]
	return fmt.Sprintf("%s cs %v csy %v z %v units %d zeroFlat %v trig %v az %v alt %v curv %v rug %d",
		name, o.cs, o.csy, o.z, o.units, o.zeroFlat, o.trig, o.azimuth, o.altitude, o.curv, o.rug)
}

func (o relOp) outputs() int {
	if o.kind == 0 {
		return 2
	}
	return 1
}

// swapped returns o for a transposed DEM: the cell sizes change axes.
func (o relOp) swapped() relOp {
	csy := o.csy
	if csy == 0 {
		csy = o.cs
	}
	o.cs, o.csy = csy, o.cs
	return o
}

// flat returns the value Aspect writes for flat cells.
func (o relOp) flat() float32 {
	if o.zeroFlat {
		return 0
	}
	return AspectFlat
}

// resolvedAzimuth is the azimuth Hillshade uses.
func (o relOp) resolvedAzimuth() float64 {
	if o.azimuth == 0 {
		return 315
	}
	return o.azimuth
}

// run runs o on dem into fresh outputs, through the plain function, the
// Tiled one or the Chunked one (path 0, 1 or 2) with eopts, and returns
// compact copies of the outputs. Outputs have masks if masked, stale
// contents if stale, and layouts decoded from d.
func (o relOp) run(d fuzzdata.Source, path int, eopts engine.Options, dem raster.Float32Raster, masked, stale bool) []raster.Float32Raster {
	outs := make([]raster.Float32Raster, o.outputs())
	for i := range outs {
		outs[i] = rastertest.Output(d, dem.Width, dem.Height, masked, stale)
	}
	ctx := context.Background()
	var err error
	switch o.kind {
	case 0:
		opts := GradientOptions{CellSize: o.cs, CellSizeY: o.csy, ZFactor: o.z}
		switch path {
		case 0:
			Gradient(outs[0], outs[1], dem, opts)
		case 1:
			err = GradientTiled(ctx, outs[0], outs[1], dem, opts, eopts)
		default:
			err = GradientChunked(ctx, engine.NewMemorySink(outs[0]), engine.NewMemorySink(outs[1]), engine.NewMemorySource(dem), opts, eopts)
		}
	case 1:
		opts := SlopeOptions{CellSize: o.cs, CellSizeY: o.csy, ZFactor: o.z, Units: o.units}
		switch path {
		case 0:
			Slope(outs[0], dem, opts)
		case 1:
			err = SlopeTiled(ctx, outs[0], dem, opts, eopts)
		default:
			err = SlopeChunked(ctx, engine.NewMemorySink(outs[0]), engine.NewMemorySource(dem), opts, eopts)
		}
	case 2:
		opts := AspectOptions{CellSize: o.cs, CellSizeY: o.csy, ZFactor: o.z, ZeroForFlat: o.zeroFlat, Trigonometric: o.trig}
		switch path {
		case 0:
			Aspect(outs[0], dem, opts)
		case 1:
			err = AspectTiled(ctx, outs[0], dem, opts, eopts)
		default:
			err = AspectChunked(ctx, engine.NewMemorySink(outs[0]), engine.NewMemorySource(dem), opts, eopts)
		}
	case 3:
		opts := HillshadeOptions{CellSize: o.cs, CellSizeY: o.csy, ZFactor: o.z, Azimuth: o.azimuth, Altitude: o.altitude}
		switch path {
		case 0:
			Hillshade(outs[0], dem, opts)
		case 1:
			err = HillshadeTiled(ctx, outs[0], dem, opts, eopts)
		default:
			err = HillshadeChunked(ctx, engine.NewMemorySink(outs[0]), engine.NewMemorySource(dem), opts, eopts)
		}
	case 4:
		opts := CurvatureOptions{CellSize: o.cs, CellSizeY: o.csy, ZFactor: o.z, Type: o.curv}
		switch path {
		case 0:
			Curvature(outs[0], dem, opts)
		case 1:
			err = CurvatureTiled(ctx, outs[0], dem, opts, eopts)
		default:
			err = CurvatureChunked(ctx, engine.NewMemorySink(outs[0]), engine.NewMemorySource(dem), opts, eopts)
		}
	case 5:
		opts := RuggednessOptions{Type: o.rug}
		switch path {
		case 0:
			Ruggedness(outs[0], dem, opts)
		case 1:
			err = RuggednessTiled(ctx, outs[0], dem, opts, eopts)
		default:
			err = RuggednessChunked(ctx, engine.NewMemorySink(outs[0]), engine.NewMemorySource(dem), opts, eopts)
		}
	}
	if err != nil {
		panic(err) // a background context never fails
	}
	for i, out := range outs {
		outs[i] = rastertest.Compact(out)
	}
	return outs
}

// relDEM values: arbitrary float bits, small integers for which Horn's
// sums are exact, or moderate values for the approximate relations.
const (
	anyValues = iota
	integerValues
	moderateValues
)

func relDEM(d fuzzdata.Source, w, h, values int, masked bool) raster.Float32Raster {
	dem := raster.NewFloat32(w, h, make([]float32, w*h))
	for i := range dem.Data {
		switch values {
		case anyValues:
			dem.Data[i] = d.Float32()
		case integerValues:
			dem.Data[i] = float32(d.Range(-1<<19, 1<<19))
		case moderateValues:
			dem.Data[i] = float32(d.Range(-1e6, 1e6)) / float32(1+d.IntN(16))
		}
	}
	if masked {
		dem.Valid = raster.NewMask(w * h)
		for i := range dem.Data {
			if d.IntN(12) == 0 {
				raster.MaskSet(dem.Valid, i, false)
			}
		}
	}
	return dem
}

// FuzzTerrainRelations checks the relations above over fuzz inputs.
// TestTerrainRelations checks the same body with rapid instead.
func FuzzTerrainRelations(f *testing.F) {
	for rel := range 11 {
		for kind := range 6 {
			f.Add([]byte{byte(rel), byte(kind), 12, 7, 1, 2, 3, 4, 5})
		}
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		terrainRelations(t, fuzzdata.New(data))
	})
}

// terrainRelations decodes a relation, an operation and a DEM from d and
// checks that relation. Each side of it runs through its own execution
// path (plain, Tiled or Chunked, with its own tiles and workers) over its
// own memory layout, so a relation also catches a halo or tile that reads
// the wrong cells.
func terrainRelations(t rastertest.TB, d fuzzdata.Source) {
	rel, kind := d.IntN(11), d.IntN(6)
	o := relOp{
		kind:     kind,
		cs:       float64(d.Range(1, 200)) / 4,
		z:        float64(d.Range(-8, 8)) / 2,
		units:    SlopeUnits(d.IntN(3)),
		zeroFlat: d.Bool(), trig: d.Bool(),
		azimuth:  float64(d.Range(-720, 720)) / 2,
		altitude: float64(d.Range(0, 180)) / 2,
		curv:     CurvatureType(d.IntN(3)),
		rug:      RuggednessType(d.IntN(4)),
	}
	if d.Bool() {
		o.csy = float64(d.Range(1, 200)) / 4
	}
	w, h := d.Range(1, 40), d.Range(1, 9)
	masked := d.IntN(3) != 0
	opts := func() engine.Options {
		return engine.Options{TileWidth: d.Range(0, w+2), TileHeight: d.Range(0, h+2), Workers: d.Range(0, 3)}
	}
	run := func(o relOp, dem raster.Float32Raster) []raster.Float32Raster {
		return o.run(d, d.IntN(3), opts(), rastertest.Place(d, dem), masked, true)
	}
	id := fmt.Sprintf("relation %d %v %d×%d masked %v", rel, o, w, h, masked)

	switch rel {
	case 0:
		testDihedral(t, d, id, o, w, h, masked, run)
	case 1: // translation
		dem := relDEM(d, w, h, integerValues, masked)
		moved := rastertest.Compact(dem)
		c := float32(d.Range(-1<<19, 1<<19))
		for i := range moved.Data {
			moved.Data[i] += c
		}
		requireSameOutputs(t, fmt.Sprintf("%s: DEM + %v", id, c), run(o, moved), run(o, dem), true)
	case 2: // power-of-two scaling
		dem := relDEM(d, w, h, integerValues, masked)
		k := d.Range(-4, 4)
		s := math.Ldexp(1, k)
		scaled := rastertest.Compact(dem)
		for i := range scaled.Data {
			scaled.Data[i] *= float32(s)
		}
		so := o
		byZ := d.Bool()
		if byZ {
			z := o.z
			if z == 0 {
				z = 1
			}
			so.z = z / s
		} else {
			so.cs *= s
			so.csy *= s
		}
		if o.kind == 5 {
			so = o // Ruggedness has no ZFactor or cell size
		}
		want := run(o, dem)
		if o.kind == 5 {
			// Ruggedness is in elevation units, and every rounding it does
			// scales with a power of two (Riley's float64 sum by its
			// square, which the root halves).
			for _, out := range want {
				for i := range out.Data {
					out.Data[i] *= float32(s)
				}
			}
		}
		if o.kind == 4 && !byZ {
			// Curvature is in 1/length: p and q are unchanged and r, s
			// and t, so the result, scale by 1/s, exactly.
			for _, out := range want {
				for i := range out.Data {
					out.Data[i] *= float32(1 / s)
				}
			}
		}
		requireSameOutputs(t, fmt.Sprintf("%s: DEM·2^%d with %v", id, k, so), run(so, scaled), want, true)
	case 3: // negation
		dem := relDEM(d, w, h, anyValues, masked)
		neg := rastertest.Compact(dem)
		for i := range neg.Data {
			neg.Data[i] = -neg.Data[i]
		}
		if o.kind == 5 {
			// IEEE negation commutes with every operation Ruggedness does,
			// and Go's max and min mirror each other. TPI's +0 stays +0,
			// so it matches its negation by value, not by bits.
			want := run(o, dem)
			if o.rug == RuggednessTPI {
				for _, out := range want {
					for i := range out.Data {
						out.Data[i] = -out.Data[i]
					}
				}
			}
			requireSameOutputs(t, id+": -DEM", run(o, neg), want, o.rug != RuggednessTPI)
			break
		}
		no := o
		if no.z == 0 {
			no.z = 1
		}
		no.z = -no.z
		requireSameOutputs(t, fmt.Sprintf("%s: -DEM vs ZFactor %v", id, no.z), run(o, neg), run(no, dem), o.kind != 0)
	case 4:
		testDerived(t, id, o, relDEM(d, w, h, anyValues, masked), run)
	case 5: // crop
		dem := relDEM(d, w, h, anyValues, masked)
		ww, hh := d.Range(1, w), d.Range(1, h)
		x0, y0 := d.Range(0, w-ww), d.Range(0, h-hh)
		full, part := run(o, dem), run(o, dem.Window(x0, y0, ww, hh))
		for k := range full {
			for y := 1; y < hh-1; y++ {
				for x := 1; x < ww-1; x++ {
					requireSameCell(t, fmt.Sprintf("%s: output %d of the %d×%d window at (%d, %d)", id, k, ww, hh, x0, y0),
						part[k], x, y, full[k], x0+x, y0+y, true)
				}
			}
		}
	case 6: // Data under invalid cells
		dem := relDEM(d, w, h, anyValues, true)
		masked = true
		scrambled := rastertest.Compact(dem)
		rastertest.ScrambleInvalid(scrambled, d)
		requireSameOutputs(t, id+": scrambled invalid cells", run(o, scrambled), run(o, dem), true)
	case 7: // one more invalid cell
		dem := relDEM(d, w, h, anyValues, true)
		masked = true
		px, py := d.IntN(w), d.IntN(h)
		holed := rastertest.Compact(dem)
		holed.SetValid(px, py, false)
		a, b := run(o, dem), run(o, holed)
		for k := range a {
			for y := range h {
				for x := range w {
					cid := fmt.Sprintf("%s: output %d with DEM cell (%d, %d) invalid", id, k, px, py)
					want := a[k].IsValid(x, y) && !rastertest.Near(x, y, px, py, 1)
					if got := b[k].IsValid(x, y); got != want {
						t.Fatalf("%s: cell (%d, %d) valid = %v, want %v", cid, x, y, got, want)
					}
					if want {
						requireSameCell(t, cid, b[k], x, y, a[k], x, y, true)
					}
				}
			}
		}
	case 8: // one changed value
		dem := relDEM(d, w, h, anyValues, masked)
		px, py := d.IntN(w), d.IntN(h)
		changed := rastertest.Compact(dem)
		changed.Data[changed.Index(px, py)] = d.Float32()
		a, b := run(o, dem), run(o, changed)
		for k := range a {
			for y := range h {
				for x := range w {
					if !rastertest.Near(x, y, px, py, 1) {
						requireSameCell(t, fmt.Sprintf("%s: output %d with DEM cell (%d, %d) changed", id, k, px, py),
							b[k], x, y, a[k], x, y, true)
					}
				}
			}
		}
	case 9: // stale outputs
		dem := relDEM(d, w, h, anyValues, masked)
		pd := rastertest.Place(d, dem)
		fresh := o.run(d, d.IntN(3), opts(), pd, masked, false)
		stale := o.run(d, d.IntN(3), opts(), pd, masked, true)
		requireSameOutputs(t, id+": stale outputs", stale, fresh, true)
	case 10: // one axis's cell size cannot matter along the other
		dem := relDEM(d, w, h, anyValues, masked)
		byColumn := d.Bool()
		for y := range h {
			for x := range w {
				if byColumn {
					dem.Data[dem.Index(x, y)] = dem.Data[dem.Index(x, 0)] // rows identical: dy = 0
				} else {
					dem.Data[dem.Index(x, y)] = dem.Data[dem.Index(0, y)] // columns identical: dx = 0
				}
			}
		}
		other := o
		if other.csy == 0 {
			other.csy = other.cs // explicit, so only one axis's size changes
		}
		if byColumn {
			other.csy = float64(d.Range(1, 200)) / 4
		} else {
			other.cs = float64(d.Range(1, 200)) / 4
		}
		requireSameOutputs(t, fmt.Sprintf("%s: rows identical %v, %v", id, byColumn, other), run(other, dem), run(o, dem), true)
	}
}

// requireSameCell fails unless cell (x, y) of got and (gx, gy) of want
// have the same validity and, if valid, the same Data: the same bits (any
// NaN matching any NaN) with bits set, the same value (+0 matching -0)
// otherwise.
func requireSameCell(t rastertest.TB, id string, got raster.Float32Raster, x, y int, want raster.Float32Raster, wx, wy int, bits bool) {
	t.Helper()
	gv, wv := got.IsValid(x, y), want.IsValid(wx, wy)
	if gv != wv {
		t.Fatalf("%s: cell (%d, %d) valid = %v, want %v as cell (%d, %d)", id, x, y, gv, wv, wx, wy)
	}
	if !gv {
		return
	}
	g, w := got.Data[got.Index(x, y)], want.Data[want.Index(wx, wy)]
	same := rastertest.SameValue(g, w)
	if bits {
		same = rastertest.SameFloat(g, w)
	}
	if !same {
		t.Fatalf("%s: cell (%d, %d) = %v (%#08x), want %v (%#08x) as cell (%d, %d)",
			id, x, y, g, math.Float32bits(g), w, math.Float32bits(w), wx, wy)
	}
}

func requireSameOutputs(t rastertest.TB, id string, got, want []raster.Float32Raster, bits bool) {
	t.Helper()
	for k := range want {
		for y := range want[k].Height {
			for x := range want[k].Width {
				requireSameCell(t, fmt.Sprintf("%s: output %d", id, k), got[k], x, y, want[k], x, y, bits)
			}
		}
	}
}

// testDihedral checks op(T(dem)) against T(op(dem)) for a symmetry T:
// exactly for Gradient (dx and dy mapped by T), Slope and Hillshade under
// a horizontal mirror with the azimuth negated, over arbitrary values; up
// to rounding for Aspect (the angle mapped by T) and Hillshade under the
// other symmetries (the light mapped by T), over moderate values.
func testDihedral(t rastertest.TB, d fuzzdata.Source, id string, o relOp, w, h int, masked bool,
	run func(relOp, raster.Float32Raster) []raster.Float32Raster) {
	t.Helper()
	tr := rastertest.All()[d.Range(1, 7)]
	mirror := tr == rastertest.Dihedral{FlipX: true}
	exact := o.kind <= 1 || o.kind == 3 && mirror || o.kind == 4 && !tr.Swap || o.kind == 5 && o.rug == RuggednessRoughness
	values := moderateValues
	if exact {
		values = anyValues
	}
	dem := relDEM(d, w, h, values, masked)
	to := o
	if tr.Swap {
		to = o.swapped()
	}
	if o.kind == 3 {
		az := o.resolvedAzimuth()
		if mirror {
			to.azimuth = -az
		} else {
			// The light's horizontal direction, (-sin az, cos az) in the
			// terms Hillshade weights dx and dy by, maps like a gradient.
			bx, by := tr.Gradient(-math.Sin(az*math.Pi/180), math.Cos(az*math.Pi/180))
			to.azimuth = math.Atan2(-bx, by) * 180 / math.Pi
			if to.azimuth == 0 {
				to.azimuth = 360
			}
		}
	}
	id = fmt.Sprintf("%s: %+v, transformed %v", id, tr, to)
	a, b := run(o, dem), run(to, tr.Apply(dem))
	tw, th := tr.Size(w, h)
	flat := o.flat()
	for y := range th {
		for x := range tw {
			u, v := tr.Source(w, h, x, y)
			if b[0].IsValid(x, y) != a[0].IsValid(u, v) {
				t.Fatalf("%s: cell (%d, %d) valid = %v, want %v as cell (%d, %d)", id, x, y, b[0].IsValid(x, y), a[0].IsValid(u, v), u, v)
			}
			if !a[0].IsValid(u, v) {
				continue
			}
			av, bv := a[0].Data[a[0].Index(u, v)], b[0].Data[b[0].Index(x, y)]
			fail := func(want string) {
				t.Fatalf("%s: cell (%d, %d) = %v, want %s (cell (%d, %d) = %v)", id, x, y, bv, want, u, v, av)
			}
			switch {
			case o.kind == 0:
				dx, dy := tr.Gradient(float64(av), float64(a[1].Data[a[1].Index(u, v)]))
				if bdy := b[1].Data[b[1].Index(x, y)]; !rastertest.SameValue(bv, float32(dx)) || !rastertest.SameValue(bdy, float32(dy)) {
					t.Fatalf("%s: cell (%d, %d) gradient = (%v, %v), want (%v, %v)", id, x, y, bv, bdy, dx, dy)
				}
			case exact:
				if !rastertest.SameFloat(bv, av) {
					fail(fmt.Sprint(av))
				}
			case av != av || bv != bv:
				if av == av || bv == bv {
					fail("NaN iff NaN")
				}
			case o.kind == 2:
				// Flat cells stay flat. With ZeroForFlat, 0 is also the
				// angle north (or east), so only a pair of zeros is known
				// to be flat, and a single 0 must be an angle.
				if av == flat && bv == flat {
					continue
				}
				if flat != 0 && (av == flat || bv == flat) {
					fail("flat iff flat")
				}
				want := mapAspect(tr, float64(av), o.trig)
				if diff := math.Abs(float64(bv) - want); math.Min(diff, 360-diff) > 1e-4 {
					fail(fmt.Sprintf("%v within 1e-4°", want))
				}
			case o.kind == 5:
				if tol := ruggednessTol(o.rug, window(dem, u, v), av); math.Abs(float64(bv)-float64(av)) > tol {
					fail(fmt.Sprintf("%v within %.3g", av, tol))
				}
			case o.kind == 4:
				// The derivatives are the same float32 values, so only
				// the formula's rounding differs, on each side.
				der := derivs32(o, dem, u, v)
				if tol := 2 * curvatureTol(o.curv, der, [5]float64{}); math.Abs(float64(bv)-float64(av)) > tol {
					fail(fmt.Sprintf("%v within %.3g", av, tol))
				}
			default:
				if math.Abs(float64(bv)-float64(av)) > 1e-3 {
					fail(fmt.Sprintf("%v within 1e-3", av))
				}
			}
		}
	}
}

// mapAspect returns the aspect, in [0, 360), of the gradient an aspect of
// a (a compass bearing, or with trig the trigonometric angle) becomes
// under T.
func mapAspect(tr rastertest.Dihedral, a float64, trig bool) float64 {
	r := a * math.Pi / 180
	// Bearing: (-gx, gy) points at (sin a, cos a). Trig: (-gx, gy) points
	// at (cos a, sin a).
	gx, gy := -math.Sin(r), math.Cos(r)
	if trig {
		gx, gy = -math.Cos(r), math.Sin(r)
	}
	gx, gy = tr.Gradient(gx, gy)
	out := math.Atan2(-gx, gy)
	if trig {
		out = math.Atan2(gy, -gx)
	}
	out *= 180 / math.Pi
	if out < 0 {
		out += 360
	}
	return out
}

// testDerived checks that Slope, Aspect and Hillshade are the documented
// functions of Gradient's dx and dy, and Curvature and Ruggedness the
// documented functions of the DEM, bit for bit, cell by cell, including
// validity.
func testDerived(t rastertest.TB, id string, o relOp, dem raster.Float32Raster,
	run func(relOp, raster.Float32Raster) []raster.Float32Raster) {
	t.Helper()
	if o.kind >= 4 {
		testDerivedWindow(t, id, o, dem, run)
		return
	}
	g := o
	g.kind = 0
	grad := run(g, dem)
	if o.kind == 0 {
		requireSameOutputs(t, id+": gradient twice", run(o, dem), grad, true)
		return
	}
	out := run(o, dem)[0]
	dx, dy := grad[0], grad[1]
	for y := range dem.Height {
		for x := range dem.Width {
			if out.IsValid(x, y) != dx.IsValid(x, y) {
				t.Fatalf("%s: cell (%d, %d) valid = %v, gradient's %v", id, x, y, out.IsValid(x, y), dx.IsValid(x, y))
			}
			gx, gy := dx.Data[dx.Index(x, y)], dy.Data[dy.Index(x, y)]
			var want float32
			switch o.kind {
			case 1:
				want = derivedSlope(o.units, gx, gy)
			case 2:
				want = derivedAspect(gx, gy, o.flat(), o.trig)
			case 3:
				k := newHillshadeKernel(HillshadeOptions{CellSize: o.cs, CellSizeY: o.csy, ZFactor: o.z, Azimuth: o.azimuth, Altitude: o.altitude})
				want = derivedHillshade(gx, gy, k.c, k.bx, k.by)
			}
			if x == 0 || y == 0 || x == dem.Width-1 || y == dem.Height-1 {
				want = float32(math.NaN()) // the edge policy, not the formula
			}
			if got := out.Data[out.Index(x, y)]; out.IsValid(x, y) && !rastertest.SameFloat(got, want) {
				t.Fatalf("%s: cell (%d, %d) = %v (%#08x), from gradient (%v, %v) %v (%#08x)",
					id, x, y, got, math.Float32bits(got), gx, gy, want, math.Float32bits(want))
			}
		}
	}
}

// The derived formulas, written from the documentation with every product
// in an explicit float32 conversion (no fused multiply-adds).

func derivedSlope(units SlopeUnits, gx, gy float32) float32 {
	m := float32(math.Sqrt(float64(float32(gx*gx) + float32(gy*gy))))
	switch units {
	case SlopePercent:
		return float32(m * 100)
	case SlopeRadians:
		return float32(stencil.Atan32(m) * 1)
	default:
		return float32(stencil.Atan32(m) * float32(180/math.Pi))
	}
}

func derivedAspect(gx, gy, flat float32, trig bool) float32 {
	y, x := 0-gx, gy
	if trig {
		y, x = gy, 0-gx
	}
	deg := float32(stencil.Atan2F32(y, x) * float32(180/math.Pi))
	off := float32(0)
	if deg < 0 {
		off = 360
	}
	deg += off
	if deg >= 360 {
		deg = 0
	}
	if y == 0 && x == 0 {
		deg = flat
	}
	return deg
}

func derivedHillshade(gx, gy, c, bx, by float32) float32 {
	num := c + (float32(bx*gx) + float32(by*gy))
	den := float32(math.Sqrt(float64(1 + (float32(gx*gx) + float32(gy*gy)))))
	v := num / den
	if v < 0 {
		v = 0
	}
	if v > 255 {
		v = 255
	}
	return v
}

// testDerivedWindow is testDerived for Curvature and Ruggedness: validity
// is the 3×3 erosion of the DEM's, and valid cells are the bits of
// derivedCurvature or derivedRuggedness.
func testDerivedWindow(t rastertest.TB, id string, o relOp, dem raster.Float32Raster,
	run func(relOp, raster.Float32Raster) []raster.Float32Raster) {
	t.Helper()
	out := run(o, dem)[0]
	derive := func(z [9]float32) float32 { return derivedRuggedness(o.rug, z) }
	if o.kind == 4 {
		k := newCurvatureKernel(CurvatureOptions{CellSize: o.cs, CellSizeY: o.csy, ZFactor: o.z, Type: o.curv})
		derive = func(z [9]float32) float32 { return derivedCurvature(o.curv, z, k.kp, k.kq, k.kr, k.kt, k.ks) }
	}
	for y := range dem.Height {
		for x := range dem.Width {
			interior := x > 0 && y > 0 && x < dem.Width-1 && y < dem.Height-1
			valid := interior
			for j := -1; valid && j <= 1; j++ {
				for i := -1; valid && i <= 1; i++ {
					valid = dem.IsValid(x+i, y+j)
				}
			}
			if out.Valid != nil && out.IsValid(x, y) != valid {
				t.Fatalf("%s: cell (%d, %d) valid = %v, want %v", id, x, y, out.IsValid(x, y), valid)
			}
			got := out.Data[out.Index(x, y)]
			if !interior {
				if got == got && out.IsValid(x, y) {
					t.Fatalf("%s: border cell (%d, %d) = %v, want NaN or invalid", id, x, y, got)
				}
				continue
			}
			if !valid {
				continue
			}
			z := window(dem, x, y)
			want := derive(z)
			if !rastertest.SameFloat(got, want) {
				t.Fatalf("%s: cell (%d, %d) = %v (%#08x), from window %v %v (%#08x)",
					id, x, y, got, math.Float32bits(got), z, want, math.Float32bits(want))
			}
		}
	}
}

// derivs32 is the float32 Zevenbergen–Thorne derivatives of the window
// centred on (x, y), as Curvature computes them, widened to float64.
func derivs32(o relOp, dem raster.Float32Raster, x, y int) [5]float64 {
	k := newCurvatureKernel(CurvatureOptions{CellSize: o.cs, CellSizeY: o.csy, ZFactor: o.z, Type: o.curv})
	at := func(dx, dy int) float32 { return dem.Data[dem.Index(x+dx, y+dy)] }
	p, q, r, s, t := ztDerivs32(at(-1, -1), at(0, -1), at(1, -1), at(-1, 0), at(0, 0), at(1, 0), at(-1, 1), at(0, 1), at(1, 1),
		k.kp, k.kq, k.kr, k.kt, k.ks)
	return [5]float64{float64(p), float64(q), float64(r), float64(s), float64(t)}
}

func ztDerivs32(z1, z2, z3, z4, z5, z6, z7, z8, z9, kp, kq, kr, kt, ks float32) (p, q, r, s, t float32) {
	p = float32((z6 - z4) * kp)
	q = float32((z8 - z2) * kq)
	r = float32(((z4 + z6) - (z5 + z5)) * kr)
	t = float32(((z2 + z8) - (z5 + z5)) * kt)
	s = float32(((z1 + z9) - (z3 + z7)) * ks)
	return p, q, r, s, t
}

func derivedCurvature(c CurvatureType, z [9]float32, kp, kq, kr, kt, ks float32) float32 {
	p, q, r, s, t := ztDerivs32(z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7], z[8], kp, kq, kr, kt, ks)
	p2, q2 := float32(p*p), float32(q*q)
	pq2 := float32(p*q) + float32(p*q)
	g := p2 + q2
	w := 1 + g
	sw := float32(math.Sqrt(float64(w)))
	switch c {
	case CurvatureProfile:
		num := (float32(p2*r) + float32(pq2*s)) + float32(q2*t)
		if g == 0 {
			return num + 0
		}
		return 0 - num/g/float32(w*sw)
	case CurvaturePlan:
		num := (float32(q2*r) - float32(pq2*s)) + float32(p2*t)
		if g == 0 {
			return num + 0
		}
		return 0 - num/g/float32(math.Sqrt(float64(g)))
	default:
		num := (float32((1+q2)*r) - float32(pq2*s)) + float32((1+p2)*t)
		d := float32(w * sw)
		return 0 - num/(d+d)
	}
}

// window is the 3×3 window of dem centred on (x, y), in row-major order.
func window(dem raster.Float32Raster, x, y int) [9]float32 {
	var z [9]float32
	for j := range 3 {
		for i := range 3 {
			z[j*3+i] = dem.Data[dem.Index(x+i-1, y+j-1)]
		}
	}
	return z
}

// derivedRuggedness is Ruggedness's documented formula for window z,
// written from the documentation and gdaldem's source rather than shared
// with the kernels.
func derivedRuggedness(r RuggednessType, z [9]float32) float32 {
	c := z[4]
	nb := [8]float32{z[0], z[1], z[2], z[3], z[5], z[6], z[7], z[8]}
	switch r {
	case RuggednessTRI:
		var s float64
		for _, v := range nb {
			d := float64(v - c)
			s += float64(d * d)
		}
		return float32(math.Sqrt(s))
	case RuggednessTRIWilson:
		s := float32(math.Abs(float64(nb[0] - c)))
		for _, v := range nb[1:] {
			s += float32(math.Abs(float64(v - c)))
		}
		return float32(s * 0.125)
	case RuggednessTPI:
		s := nb[0]
		for _, v := range nb[1:] {
			s += v
		}
		return c - float32(s*0.125)
	default:
		return slices.Max(z[:]) - slices.Min(z[:])
	}
}

// ruggednessTol bounds how far a reordering of Ruggedness's sums can move
// its result a for window z. Riley's float64 sum moves it only through
// the final rounding to float32, by an ulp of a on each side. Wilson's
// and TPI's float32 sums of eight terms round seven times each, every
// time by at most half an ulp of a running total no larger than the sum
// of the terms' magnitudes, and the result is an eighth of the sum.
func ruggednessTol(r RuggednessType, z [9]float32, a float32) float64 {
	const eps = 0x1p-24
	abs := math.Abs(float64(a))
	ulp := float64(math.Nextafter32(float32(abs), float32(math.Inf(1)))) - abs
	if r == RuggednessTRI {
		return 2 * ulp
	}
	var mag float64
	for i, v := range z {
		switch {
		case i == 4:
		case r == RuggednessTRIWilson:
			mag += math.Abs(float64(v) - float64(z[4]))
		default:
			mag += math.Abs(float64(v))
		}
	}
	return 2*7*eps*mag/8 + 2*ulp
}
