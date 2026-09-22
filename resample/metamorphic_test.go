package resample_test

import (
	"math"
	"testing"

	"github.com/LukasSelin/strata/internal/fuzzdata"
	"github.com/LukasSelin/strata/internal/rastertest"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/resample"
)

// relationCase is one decoded case: a source on a dyadic grid, so that
// every coordinate the tables compute is exact and relations between
// grids can be held to the bit, and a destination grid over it.
type relationCase struct {
	m      resample.Method
	sg, dg raster.Grid
	src    raster.Float32Raster
}

func decodeRelationCase(d fuzzdata.Source) relationCase {
	c := relationCase{m: resample.Method(d.IntN(5))}
	c.sg = raster.Grid{Width: d.Range(2, 30), Height: d.Range(2, 30), ResolutionX: 1, ResolutionY: -1}
	c.sg.OriginY = float64(c.sg.Height)
	scales := []float64{0.25, 0.5, 1, 2, 4, 0.75, 1.5}
	sx, sy := scales[d.IntN(len(scales))], scales[d.IntN(len(scales))]
	c.dg = raster.Grid{
		Width: max(1, int(float64(c.sg.Width)/sx)), Height: max(1, int(float64(c.sg.Height)/sy)),
		ResolutionX: sx, ResolutionY: -sy,
		OriginX: float64(d.IntN(5)) / 4, OriginY: c.sg.OriginY - float64(d.IntN(5))/4,
	}
	c.src = raster.NewFloat32(c.sg.Width, c.sg.Height, make([]float32, c.sg.Width*c.sg.Height))
	for i := range c.src.Data {
		// Moderate values, so that scaling by powers of two stays exact.
		c.src.Data[i] = float32(d.Range(-1000, 1000)) / 8
	}
	if d.Bool() {
		c.src.Valid = raster.NewMask(len(c.src.Data))
		for i := range c.src.Data {
			raster.MaskSet(c.src.Valid, i, d.IntN(6) != 0)
		}
	}
	return c
}

func (c relationCase) run(src raster.Float32Raster, sg, dg raster.Grid) raster.Float32Raster {
	dst := raster.NewFloat32(dg.Width, dg.Height, make([]float32, dg.Width*dg.Height))
	dst.Valid = raster.NewMask(len(dst.Data))
	resample.Resample(raster.NewDataset(dg, dst), raster.NewDataset(sg, src), resample.Options{Method: c.m})
	return dst
}

func sameRasters(t rastertest.TB, what string, a, b raster.Float32Raster) {
	t.Helper()
	for y := range a.Height {
		for x := range a.Width {
			if a.IsValid(x, y) != b.IsValid(x, y) {
				t.Fatalf("%s: cell (%d, %d) validity %v vs %v", what, x, y, a.IsValid(x, y), b.IsValid(x, y))
			}
			if a.IsValid(x, y) && !rastertest.SameFloat(a.Data[a.Index(x, y)], b.Data[b.Index(x, y)]) {
				t.Fatalf("%s: cell (%d, %d) = %v vs %v", what, x, y, a.Data[a.Index(x, y)], b.Data[b.Index(x, y)])
			}
		}
	}
}

// resampleRelations holds one decoded case to the relations resampling
// must satisfy. Each side of a relation is a separate call, so a relation
// breaks if a table, a pass or the validity rule is wrong in a way that
// depends on where or what the data is.
func resampleRelations(t rastertest.TB, d fuzzdata.Source) {
	t.Helper()
	c := decodeRelationCase(d)
	base := c.run(c.src, c.sg, c.dg)

	// Identity: the source's own grid copies it.
	sameRasters(t, "identity", c.run(c.src, c.sg, c.sg), maskedCopy(c.src))

	// Translation: moving both grids by whole source cells changes nothing.
	kx, ky := float64(d.Range(-3, 3)), float64(d.Range(-3, 3))
	sg, dg := c.sg, c.dg
	sg.OriginX, sg.OriginY = sg.OriginX+kx, sg.OriginY+ky
	dg.OriginX, dg.OriginY = dg.OriginX+kx, dg.OriginY+ky
	sameRasters(t, "translation", c.run(c.src, sg, dg), base)

	// Scaling the values by a power of two scales the result exactly.
	scaled := rastertest.Compact(c.src)
	for i := range scaled.Data {
		scaled.Data[i] *= 4
	}
	got := c.run(scaled, c.sg, c.dg)
	for i := range got.Data {
		got.Data[i] /= 4
	}
	sameRasters(t, "power-of-two scaling", got, base)

	// An all-valid mask is no mask; Data under cleared bits is never read.
	if c.src.Valid == nil {
		all := rastertest.Compact(c.src)
		all.Valid = raster.NewMask(len(all.Data))
		sameRasters(t, "all-valid mask", c.run(all, c.sg, c.dg), base)
	} else {
		junk := rastertest.Compact(c.src)
		rastertest.ScrambleInvalid(junk, d)
		sameRasters(t, "data under cleared bits", c.run(junk, c.sg, c.dg), base)
	}

	// Locality: a source window holding every cell a destination cell
	// reads gives that cell the same bits.
	x0, y0 := d.IntN(c.sg.Width), d.IntN(c.sg.Height)
	w, h := d.Range(1, c.sg.Width-x0), d.Range(1, c.sg.Height-y0)
	win := raster.NewDataset(c.sg, c.src).Window(x0, y0, w, h)
	local := c.run(win.Raster, win.Grid, c.dg)
	reach := map[resample.Method]float64{resample.Nearest: 0, resample.Bilinear: 1, resample.Cubic: 2, resample.Lanczos: 3, resample.Average: 0}[c.m]
	for r := range c.dg.Height {
		for col := range c.dg.Width {
			// The source-pixel box this cell's kernel can reach.
			sx, sy := math.Abs(c.dg.ResolutionX), math.Abs(c.dg.ResolutionY)
			rx, ry := reach*math.Max(1, sx)+sx/2+1, reach*math.Max(1, sy)+sy/2+1
			u := c.dg.OriginX + (float64(col)+0.5)*c.dg.ResolutionX
			v := c.sg.OriginY - (c.dg.OriginY + (float64(r)+0.5)*c.dg.ResolutionY)
			if u-rx < float64(x0) || u+rx > float64(x0+w) || v-ry < float64(y0) || v+ry > float64(y0+h) {
				continue
			}
			if local.IsValid(col, r) != base.IsValid(col, r) ||
				base.IsValid(col, r) && !rastertest.SameFloat(local.Data[local.Index(col, r)], base.Data[base.Index(col, r)]) {
				t.Fatalf("%v locality: cell (%d, %d) inside window (%d, %d) %d×%d: %v/%v vs %v/%v", c.m, col, r, x0, y0, w, h,
					local.Data[local.Index(col, r)], local.IsValid(col, r), base.Data[base.Index(col, r)], base.IsValid(col, r))
			}
		}
	}

	// A constant field stays constant, to within the rounding of weights
	// that sum to one, and exactly under Nearest. Renormalising Cubic or
	// Lanczos over the valid cells divides two float32 sums whose negative
	// lobes can cancel most of the weight, which amplifies their rounding;
	// that needs a looser bound.
	k := float32(d.Range(-100, 100)) / 4
	flat := rastertest.Compact(c.src)
	for i := range flat.Data {
		flat.Data[i] = k
	}
	fr := c.run(flat, c.sg, c.dg)
	ones := rastertest.Compact(c.src)
	for i := range ones.Data {
		ones.Data[i] = 1
	}
	or := c.run(ones, c.sg, c.dg)
	for i, v := range fr.Data {
		if !raster.MaskGet(fr.Valid, i) {
			continue
		}
		tol := 1e-5 * math.Abs(float64(k))
		if c.src.Valid != nil && (c.m == resample.Cubic || c.m == resample.Lanczos) {
			// How far a field of ones strays measures the conditioning.
			tol = math.Max(2e-3, 16*math.Abs(float64(or.Data[i])-1)) * math.Abs(float64(k))
		}
		if c.m == resample.Nearest {
			tol = 0
		}
		if math.Abs(float64(v-k)) > tol {
			t.Fatalf("%v: a field of %v resamples to %v at cell %d", c.m, k, v, i)
		}
	}

	// Mirroring both grids mirrors the result: a flipped source axis with
	// the destination read in reverse. Sums run in the other order, so
	// only within rounding, except under Nearest; cells whose centre lies
	// on a source edge may round to the other side and are skipped.
	if c.m != resample.Nearest {
		mir := rastertest.Dihedral{FlipX: true}.Apply(c.src)
		msg, mdg := c.sg, c.dg
		mdg.OriginX = float64(c.sg.Width) - (c.dg.OriginX + float64(c.dg.Width)*c.dg.ResolutionX)
		mr := c.run(mir, msg, mdg)
		for r := range c.dg.Height {
			for col := range c.dg.Width {
				mc := c.dg.Width - 1 - col
				u := c.dg.OriginX + (float64(col)+0.5)*c.dg.ResolutionX
				if u == math.Trunc(u) || base.IsValid(col, r) != mr.IsValid(mc, r) {
					continue // a tie on a cell edge: the centre rule rounds the other way
				}
				if !base.IsValid(col, r) {
					continue
				}
				a, b := float64(base.Data[base.Index(col, r)]), float64(mr.Data[mr.Index(mc, r)])
				// A renormalised value far beyond the inputs (at most 125)
				// comes from a valid weight that nearly cancels, which
				// amplifies rounding by as much.
				amp := math.Max(1, math.Abs(a)/125)
				if math.Abs(a-b) > 1e-4*math.Max(1, math.Abs(a))*amp {
					t.Fatalf("%v mirror: cell (%d, %d) = %v, mirrored %v", c.m, col, r, a, b)
				}
			}
		}
	}

	// Nearest up ×2 and Average back down returns the source exactly.
	up := raster.Grid{Width: 2 * c.sg.Width, Height: 2 * c.sg.Height, ResolutionX: 0.5, ResolutionY: -0.5, OriginY: c.sg.OriginY}
	nc := c
	nc.m = resample.Nearest
	upr := nc.run(c.src, c.sg, up)
	if c.src.Valid == nil {
		upr.Valid = nil
		ac := c
		ac.m = resample.Average
		back := ac.run(upr, up, c.sg)
		sameRasters(t, "nearest up, average down", back, maskedCopy(c.src))
	}
}

// maskedCopy is r compact with a mask, all valid if r has none.
func maskedCopy(r raster.Float32Raster) raster.Float32Raster {
	c := rastertest.Compact(r)
	if c.Valid == nil {
		c.Valid = raster.NewMask(len(c.Data))
	}
	return c
}

// FuzzResampleRelations fuzzes the relations of resampleRelations.
func FuzzResampleRelations(f *testing.F) {
	f.Add([]byte{0, 5, 5, 1, 1})
	f.Add([]byte{3, 12, 9, 4, 2, 1, 1, 7})
	f.Add([]byte{4, 20, 3, 0, 5, 2, 3})
	f.Fuzz(func(t *testing.T, b []byte) {
		resampleRelations(t, fuzzdata.New(b))
	})
}
