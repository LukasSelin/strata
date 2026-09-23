package resample_test

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/resample"
)

var methods = []resample.Method{resample.Nearest, resample.Bilinear, resample.Cubic, resample.Lanczos, resample.Average}

// reference is an independent float64 resampler written from the package
// documentation, not from the tables: for each output cell it walks every
// source cell and weighs it by the method's kernel. It returns the value
// and validity of output cell (c, r).
type reference struct {
	m        resample.Method
	dg, sg   raster.Grid
	src      raster.Float32Raster
	sx, sy   float64 // source cells per output cell
	wx, wy   float64 // kernel stretch per axis
	cubic4   bool
	halfRule bool
}

func newReference(m resample.Method, dg raster.Grid, src raster.Dataset) *reference {
	r := &reference{m: m, dg: dg, sg: src.Grid, src: src.Raster}
	r.sx = math.Abs(dg.ResolutionX / src.Grid.ResolutionX)
	r.sy = math.Abs(dg.ResolutionY / src.Grid.ResolutionY)
	stretch := func(s float64) float64 {
		switch m {
		case resample.Bilinear, resample.Cubic:
			if 1/s < 0.95 {
				return s
			}
		case resample.Lanczos:
			if s > 1 {
				return s
			}
		}
		return 1
	}
	r.wx, r.wy = stretch(r.sx), stretch(r.sy)
	r.cubic4 = m == resample.Cubic && r.wx == 1 && r.wy == 1
	r.halfRule = m == resample.Lanczos
	return r
}

func refKernel(m resample.Method, d float64) float64 {
	ad := math.Abs(d)
	switch m {
	case resample.Bilinear:
		return math.Max(0, 1-ad)
	case resample.Cubic:
		const a = -0.5
		switch {
		case ad < 1:
			return (a+2)*ad*ad*ad - (a+3)*ad*ad + 1
		case ad < 2:
			return a*ad*ad*ad - 5*a*ad*ad + 8*a*ad - 4*a
		}
		return 0
	case resample.Lanczos:
		if ad >= 3 {
			return 0
		}
		if d == 0 {
			return 1
		}
		px := math.Pi * d
		return 3 * math.Sin(px) * math.Sin(px/3) / (px * px)
	}
	panic("no kernel")
}

func refSupport(m resample.Method) float64 {
	return map[resample.Method]float64{resample.Bilinear: 1, resample.Cubic: 2, resample.Lanczos: 3}[m]
}

// pixel maps output cell (c, r)'s centre to source pixel coordinates.
func (r *reference) pixel(c, row float64) (u, v float64) {
	x := r.dg.OriginX + c*r.dg.ResolutionX
	y := r.dg.OriginY + row*r.dg.ResolutionY
	return (x - r.sg.OriginX) / r.sg.ResolutionX, (y - r.sg.OriginY) / r.sg.ResolutionY
}

func (r *reference) valid(i, j int) bool {
	return r.src.Valid == nil || r.src.IsValid(i, j)
}

func (r *reference) at(c, row int) (float64, bool) {
	sw, sh := r.src.Width, r.src.Height
	u, v := r.pixel(float64(c)+0.5, float64(row)+0.5)
	ci, cj := int(math.Floor(u)), int(math.Floor(v))
	inside := u >= 0 && v >= 0 && ci < sw && cj < sh
	switch r.m {
	case resample.Nearest:
		if !inside || !r.valid(ci, cj) {
			return 0, false
		}
		return float64(r.src.Data[r.src.Index(ci, cj)]), true
	case resample.Average:
		u0, v0 := r.pixel(float64(c), float64(row))
		u1, v1 := r.pixel(float64(c)+1, float64(row)+1)
		u0, u1 = math.Min(u0, u1), math.Max(u0, u1)
		v0, v1 = math.Min(v0, v1), math.Max(v0, v1)
		var num, den float64
		for j := range sh {
			oy := math.Min(v1, float64(j+1)) - math.Max(v0, float64(j))
			if oy <= 0 {
				continue
			}
			for i := range sw {
				ox := math.Min(u1, float64(i+1)) - math.Max(u0, float64(i))
				if ox <= 0 || !r.valid(i, j) {
					continue
				}
				num += ox * oy * float64(r.src.Data[r.src.Index(i, j)])
				den += ox * oy
			}
		}
		if den <= 0 {
			return 0, false
		}
		return num / den, true
	}
	if !inside || !r.valid(ci, cj) {
		return 0, false
	}
	sum := func(m resample.Method, wx, wy float64) (num, den float64, full, clipped bool, nvalid, window int) {
		full = true
		sup := refSupport(m)
		for j := int(math.Floor(v - 0.5 - sup*wy)); j <= int(math.Ceil(v-0.5+sup*wy)); j++ {
			ky := refKernel(m, (float64(j)+0.5-v)/wy)
			if math.Abs((float64(j)+0.5-v)/wy) >= sup {
				continue
			}
			for i := int(math.Floor(u - 0.5 - sup*wx)); i <= int(math.Ceil(u-0.5+sup*wx)); i++ {
				dx := (float64(i) + 0.5 - u) / wx
				if math.Abs(dx) >= sup {
					continue
				}
				w := refKernel(m, dx) * ky
				if i < 0 || j < 0 || i >= sw || j >= sh {
					if w != 0 {
						clipped = true
					}
					continue
				}
				window++
				if !r.valid(i, j) {
					if w != 0 {
						full = false
					}
					continue
				}
				nvalid++
				num += w * float64(r.src.Data[r.src.Index(i, j)])
				den += w
			}
		}
		return num, den, full, clipped, nvalid, window
	}
	num, den, full, clipped, nvalid, window := sum(r.m, r.wx, r.wy)
	// A centre exactly on a source centre on both axes copies that cell,
	// and gdalwarp waives the half-valid rule for it.
	exact := math.Abs(u-0.5-math.Round(u-0.5)) < 1e-9 && math.Abs(v-0.5-math.Round(v-0.5)) < 1e-9
	if r.cubic4 && (!full || clipped) {
		num, den, _, _, _, _ = sum(resample.Bilinear, 1, 1)
	}
	if den <= 0 {
		return 0, false
	}
	if r.halfRule && !exact && r.src.Valid != nil && 2*nvalid < window {
		return 0, false
	}
	return num / den, true
}

// grids returns a random source grid and a destination grid over roughly
// the same area at a random scale, with random signs and offsets.
func grids(rng *rand.Rand) (dg, sg raster.Grid) {
	sg = raster.Grid{Width: 3 + rng.IntN(40), Height: 3 + rng.IntN(40), ResolutionX: 1, ResolutionY: -1}
	if rng.IntN(4) == 0 {
		sg.ResolutionY = 1
	}
	sg.OriginY = float64(sg.Height)
	if sg.ResolutionY > 0 {
		sg.OriginY = 0
	}
	scales := []float64{0.25, 0.5, 0.73, 1, 1.37, 2, 3.3, 4}
	sx, sy := scales[rng.IntN(len(scales))], scales[rng.IntN(len(scales))]
	if rng.IntN(3) == 0 {
		sy = sx
	}
	dg = raster.Grid{
		Width:       max(1, int(float64(sg.Width)/sx)+rng.IntN(3)-1),
		Height:      max(1, int(float64(sg.Height)/sy)+rng.IntN(3)-1),
		ResolutionX: sx,
		ResolutionY: -sy,
		OriginX:     rng.Float64()*2 - 1,
	}
	dg.OriginY = float64(sg.Height) + rng.Float64()*2 - 1
	if rng.IntN(2) == 0 {
		dg.OriginX, dg.OriginY = 0, float64(sg.Height)
	}
	return dg, sg
}

func randomRaster(rng *rand.Rand, w, h int, masked bool) raster.Float32Raster {
	r := raster.NewFloat32(w, h, make([]float32, w*h))
	for i := range r.Data {
		r.Data[i] = float32(rng.NormFloat64()*10 + 50)
	}
	if masked {
		r.Valid = raster.NewMask(w * h)
		for i := range r.Data {
			if rng.IntN(6) == 0 {
				raster.MaskSet(r.Valid, i, false)
				r.Data[i] = float32(math.Inf(1)) // must never leak into a valid cell
			}
		}
	}
	return r
}

func newDst(g raster.Grid) raster.Dataset {
	r := raster.NewFloat32(g.Width, g.Height, make([]float32, g.Width*g.Height))
	r.Valid = raster.NewMask(g.Width * g.Height)
	return raster.NewDataset(g, r)
}

// TestAgainstReference holds every method to the float64 reference over
// random grids, with and without masks: validity exactly, values within
// float32 rounding of the weighted sums.
func TestAgainstReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for iter := range 300 {
		dg, sg := grids(rng)
		masked := iter%2 == 1
		src := raster.NewDataset(sg, randomRaster(rng, sg.Width, sg.Height, masked))
		for _, m := range methods {
			dst := newDst(dg)
			resample.Resample(dst, src, resample.Options{Method: m})
			ref := newReference(m, dg, src)
			for r := range dg.Height {
				for c := range dg.Width {
					want, wantOK := ref.at(c, r)
					got, gotOK := dst.Raster.Data[dst.Raster.Index(c, r)], dst.Raster.IsValid(c, r)
					if gotOK != wantOK {
						t.Fatalf("%v masked=%v dst %+v src %+v: cell (%d, %d) valid %v, reference %v",
							m, masked, dg, sg, c, r, gotOK, wantOK)
					}
					if !gotOK {
						if !math.IsNaN(float64(got)) {
							t.Fatalf("%v: invalid cell (%d, %d) holds %v, want NaN", m, c, r, got)
						}
						continue
					}
					if math.Abs(float64(got)-want) > 2e-4*math.Max(1, math.Abs(want)) {
						t.Fatalf("%v masked=%v dst %+v src %+v: cell (%d, %d) = %v, reference %v",
							m, masked, dg, sg, c, r, got, want)
					}
				}
			}
		}
	}
}

// TestIdentityIsCopy: resampling onto the source's own grid copies it bit
// for bit under every method, NaN payloads, infinities and -0 included.
func TestIdentityIsCopy(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, res := range []float64{1, 0.3, 30} {
		g := raster.Grid{Width: 23, Height: 17, ResolutionX: res, ResolutionY: -res, OriginX: 100, OriginY: 500}
		sr := randomRaster(rng, g.Width, g.Height, false)
		sr.Data[5] = float32(math.Copysign(0, -1))
		sr.Data[6] = float32(math.Inf(-1))
		sr.Data[7] = math.Float32frombits(0x7fc01234)
		src := raster.NewDataset(g, sr)
		for _, m := range methods {
			dst := newDst(g)
			resample.Resample(dst, src, resample.Options{Method: m})
			for i, v := range sr.Data {
				got := dst.Raster.Data[i]
				if math.Float32bits(got) != math.Float32bits(v) && !(v != v && got != got && m != resample.Nearest) {
					t.Fatalf("%v res %v: cell %d = %v (%#x), source %v (%#x)", m, res, i, got, math.Float32bits(got), v, math.Float32bits(v))
				}
			}
		}
	}
}

// TestTiledAndChunkedMatch is the §23 contract for resampling: every tile
// size and worker count, Tiled and Chunked, gives the plain call's bits
// in Data and in validity.
func TestTiledAndChunkedMatch(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for iter := range 60 {
		dg, sg := grids(rng)
		masked := iter%2 == 0
		src := raster.NewDataset(sg, randomRaster(rng, sg.Width, sg.Height, masked))
		for _, m := range methods {
			want := newDst(dg)
			resample.Resample(want, src, resample.Options{Method: m})
			for _, tile := range [][2]int{{1, 1}, {7, 5}, {64, 64}, {0, 3}, {dg.Width + 5, dg.Height + 5}} {
				for _, workers := range []int{1, 2, 3, 0} {
					eo := engine.Options{TileWidth: tile[0], TileHeight: tile[1], Workers: workers}
					got := newDst(dg)
					if err := resample.ResampleTiled(t.Context(), got, src, resample.Options{Method: m}, eo); err != nil {
						t.Fatal(err)
					}
					sameBits(t, "Tiled", m, eo, want.Raster, got.Raster)
					got = newDst(dg)
					err := resample.ResampleChunked(t.Context(), engine.NewMemorySink(got.Raster), dg,
						engine.NewMemorySource(src.Raster), sg, resample.Options{Method: m}, eo)
					if err != nil {
						t.Fatal(err)
					}
					sameBits(t, "Chunked", m, eo, want.Raster, got.Raster)
				}
			}
		}
	}
}

func sameBits(t *testing.T, form string, m resample.Method, eo engine.Options, want, got raster.Float32Raster) {
	t.Helper()
	for y := range want.Height {
		for x := range want.Width {
			a, b := want.Data[want.Index(x, y)], got.Data[got.Index(x, y)]
			if want.IsValid(x, y) != got.IsValid(x, y) {
				t.Fatalf("%s %v %+v: cell (%d, %d) validity differs", form, m, eo, x, y)
			}
			if math.Float32bits(a) != math.Float32bits(b) && !(a != a && b != b) {
				t.Fatalf("%s %v %+v: cell (%d, %d) = %v, plain %v", form, m, eo, x, y, b, a)
			}
		}
	}
}

func TestPanics(t *testing.T) {
	g := raster.Grid{Width: 4, Height: 4, ResolutionX: 1, ResolutionY: -1, OriginY: 4}
	src := raster.NewDataset(g, raster.NewFloat32(4, 4, make([]float32, 16)))
	masked := raster.NewFloat32(4, 4, make([]float32, 16))
	masked.Valid = raster.NewMask(16)
	msrc := raster.NewDataset(g, masked)
	plainDst := func(g raster.Grid) raster.Dataset {
		return raster.NewDataset(g, raster.NewFloat32(g.Width, g.Height, make([]float32, g.Width*g.Height)))
	}
	shifted := g
	shifted.OriginX = 0.5
	crsA, crsB := g, g
	crsA.CRS.Code, crsB.CRS.Code = "EPSG:25833", "EPSG:4326"
	crsURN := g
	crsURN.CRS.Code = "urn:ogc:def:crs:EPSG::25833" // opaque, so not linked
	zero := g
	zero.ResolutionX = 0
	inf := g
	inf.OriginX = math.Inf(1)
	for _, c := range []struct {
		name string
		run  func()
		want string
	}{
		{"crs", func() { resample.Resample(newDst(crsA), raster.NewDataset(crsB, src.Raster), resample.Options{}) },
			`"EPSG:25833" (https://epsg.io/25833) differs from src CRS "EPSG:4326" (https://epsg.io/4326)`},
		{"crs unlinked", func() { resample.Resample(newDst(crsA), raster.NewDataset(crsURN, src.Raster), resample.Options{}) },
			`src CRS "urn:ogc:def:crs:EPSG::25833"; reprojection`},
		{"zero res", func() { resample.Resample(newDst(zero), src, resample.Options{}) }, "zero resolution"},
		{"inf origin", func() { resample.Resample(newDst(inf), src, resample.Options{}) }, "non-finite"},
		{"method", func() { resample.Resample(newDst(g), src, resample.Options{Method: 9}) }, "unknown"},
		{"masked src", func() { resample.Resample(plainDst(g), msrc, resample.Options{}) }, "validity mask"},
		{"uncovered", func() { resample.Resample(plainDst(shifted), src, resample.Options{Method: resample.Bilinear}) }, "beyond src"},
		{"same memory", func() { resample.Resample(raster.NewDataset(g, src.Raster), src, resample.Options{}) }, "share memory"},
		{"options", func() {
			_ = resample.ResampleTiled(t.Context(), newDst(g), src, resample.Options{}, engine.Options{Workers: -1})
		}, "negative"},
		{"grid size", func() {
			_ = resample.ResampleChunked(t.Context(), engine.NewMemorySink(newDst(g).Raster), shifted,
				engine.NewMemorySource(src.Raster), raster.Grid{Width: 5, Height: 4, ResolutionX: 1, ResolutionY: 1}, resample.Options{}, engine.Options{})
		}, "grid is"},
	} {
		func() {
			defer func() {
				v := recover()
				s, _ := v.(string)
				if !strings.HasPrefix(s, "resample: ") || !strings.Contains(s, c.want) {
					t.Errorf("%s: panic %v, want a resample: panic mentioning %q", c.name, v, c.want)
				}
			}()
			c.run()
		}()
	}
}
