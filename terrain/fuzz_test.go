package terrain

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/fuzzdata"
	"github.com/LukasSelin/strata/internal/stencil"
	"github.com/LukasSelin/strata/raster"
)

// fuzzRoot is a raster that owns its memory and the window of it an
// operand uses.
type fuzzRoot struct {
	r    raster.Float32Raster
	x, y int
}

func (o fuzzRoot) clone() fuzzRoot {
	o.r.Data = append([]float32(nil), o.r.Data...)
	if o.r.Valid != nil {
		o.r.Valid = append([]uint64(nil), o.r.Valid...)
	}
	return o
}

// terrainCall is one fuzzed operation: its plain, Tiled and Chunked forms
// over a DEM and one or two outputs.
type terrainCall struct {
	name    string
	outputs int
	plain   func(outs []raster.Float32Raster, dem raster.Float32Raster)
	tiled   func(ctx context.Context, outs []raster.Float32Raster, dem raster.Float32Raster, o engine.Options) error
	chunked func(ctx context.Context, outs []engine.RasterSink, dem engine.RasterSource, o engine.Options) error
	// inRange reports whether a valid interior value is possible.
	inRange func(v float32) bool
	// finite reports whether a cell whose neighbourhood z is finite and
	// moderate must have a value that is not NaN.
	finite func(z [9]float64) bool
}

// FuzzTerrain runs Gradient, Slope, Aspect, Hillshade, Curvature and
// Ruggedness with options decoded from the fuzz input, sane or arbitrary (NaN, infinities, huge
// and tiny values), over DEMs with arbitrary values, windows, strides and
// masks, and outputs that may lack a mask or overlap the DEM. Invalid
// options must panic with a "terrain:" message and invalid operands with
// an "engine:" one, before anything is written. Otherwise the plain,
// Tiled and Chunked functions must write the same bits for any tiles and
// workers, nothing outside the outputs' cells, NaN and invalid borders,
// the eroded validity of the DEM, the values of a scalar run on a compact
// copy of the DEM, only values in the documented range, and no NaN where
// every elevation around a cell is finite and moderate.
func FuzzTerrain(f *testing.F) {
	f.Add([]byte{1, 10, 5, 0})
	f.Add([]byte{3, 40, 4, 1, 1, 0, 0, 1, 2, 3})
	f.Add([]byte{2, 3, 3, 2, 0, 1, 7})
	f.Add([]byte{0, 66, 6, 0, 3, 3, 1, 1})
	f.Add([]byte{4, 20, 7, 1, 2, 2, 0, 1, 1, 0})
	f.Add([]byte{5, 12, 9, 2, 1, 3, 0, 2, 1, 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		defer stencil.UseScalar(false)
		d := fuzzdata.New(data)
		kind := d.IntN(6)
		// Options: mostly sane, sometimes arbitrary.
		opt := func(sane float64) float64 {
			if d.IntN(4) == 0 {
				return d.Float64()
			}
			return sane
		}
		cs := opt(float64(d.Range(1, 40)))
		csy := opt(float64(d.Range(0, 40)))
		z := opt(float64(d.Range(-4, 4)))
		units := SlopeUnits(d.IntN(3))
		if d.IntN(10) == 0 {
			units = SlopeUnits(d.Range(-1, 5))
		}
		az, alt := opt(float64(d.Range(0, 400))), opt(float64(d.Range(0, 90)))
		zeroFlat, trig := d.Bool(), d.Bool()
		curv := CurvatureType(d.IntN(3))
		if d.IntN(10) == 0 {
			curv = CurvatureType(d.Range(-1, 5))
		}
		rug := RuggednessType(d.IntN(4))
		if d.IntN(10) == 0 {
			rug = RuggednessType(d.Range(-1, 6))
		}

		// The documented option rules.
		csy0, z0, alt0 := csy, z, alt
		if csy0 == 0 {
			csy0 = cs
		}
		if z0 == 0 {
			z0 = 1
		}
		if alt0 == 0 {
			alt0 = 45
		}
		finiteF := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
		kx64, ky64 := z0/(8*cs), z0/(8*csy0)
		scaleOK := func(k float64) bool {
			k32 := float32(k)
			return k32 != 0 && !math.IsInf(float64(k32), 0)
		}
		optionsOK := cs > 0 && finiteF(cs) && csy0 > 0 && finiteF(csy0) && finiteF(z0) && scaleOK(kx64) && scaleOK(ky64)
		// Curvature has its own factors and does not need Horn's.
		cellsOK := cs > 0 && finiteF(cs) && csy0 > 0 && finiteF(csy0) && finiteF(z0)
		ztK := [5]float64{z0 / (2 * cs), z0 / (2 * csy0), z0 / (cs * cs), z0 / (4 * cs * csy0), z0 / (csy0 * csy0)}
		hornGrad := func(z [9]float64) (gx, gy float64) {
			gx = ((z[2] + z[8]) + 2*z[5] - (z[0] + z[6]) - 2*z[3]) * kx64
			gy = ((z[6] + z[8]) + 2*z[7] - (z[0] + z[2]) - 2*z[1]) * ky64
			return gx, gy
		}

		ctx := context.Background()
		flat := float32(AspectFlat)
		if zeroFlat {
			flat = 0
		}
		var call terrainCall
		switch kind {
		case 0:
			o := GradientOptions{cs, csy, z}
			call = terrainCall{"gradient", 2,
				func(outs []raster.Float32Raster, dem raster.Float32Raster) { Gradient(outs[0], outs[1], dem, o) },
				func(ctx context.Context, outs []raster.Float32Raster, dem raster.Float32Raster, e engine.Options) error {
					return GradientTiled(ctx, outs[0], outs[1], dem, o, e)
				},
				func(ctx context.Context, outs []engine.RasterSink, dem engine.RasterSource, e engine.Options) error {
					return GradientChunked(ctx, outs[0], outs[1], dem, o, e)
				},
				func(float32) bool { return true },
				func([9]float64) bool { return true },
			}
		case 1:
			o := SlopeOptions{cs, csy, z, units}
			optionsOK = optionsOK && units >= SlopeDegrees && units <= SlopeRadians
			top := float32(90)
			switch units {
			case SlopePercent:
				top = float32(math.Inf(1))
			case SlopeRadians:
				top = float32(math.Pi / 2)
			}
			call = terrainCall{"slope", 1,
				func(outs []raster.Float32Raster, dem raster.Float32Raster) { Slope(outs[0], dem, o) },
				func(ctx context.Context, outs []raster.Float32Raster, dem raster.Float32Raster, e engine.Options) error {
					return SlopeTiled(ctx, outs[0], dem, o, e)
				},
				func(ctx context.Context, outs []engine.RasterSink, dem engine.RasterSource, e engine.Options) error {
					return SlopeChunked(ctx, outs[0], dem, o, e)
				},
				func(v float32) bool { return v >= 0 && v <= top },
				func([9]float64) bool { return true },
			}
		case 2:
			o := AspectOptions{cs, csy, z, zeroFlat, trig}
			call = terrainCall{"aspect", 1,
				func(outs []raster.Float32Raster, dem raster.Float32Raster) { Aspect(outs[0], dem, o) },
				func(ctx context.Context, outs []raster.Float32Raster, dem raster.Float32Raster, e engine.Options) error {
					return AspectTiled(ctx, outs[0], dem, o, e)
				},
				func(ctx context.Context, outs []engine.RasterSink, dem engine.RasterSource, e engine.Options) error {
					return AspectChunked(ctx, outs[0], dem, o, e)
				},
				func(v float32) bool { return v == flat || v >= 0 && v < 360 },
				func([9]float64) bool { return true },
			}
		case 3:
			o := HillshadeOptions{cs, csy, z, az, alt}
			optionsOK = optionsOK && finiteF(az) && alt0 > 0 && alt0 <= 90
			call = terrainCall{"hillshade", 1,
				func(outs []raster.Float32Raster, dem raster.Float32Raster) { Hillshade(outs[0], dem, o) },
				func(ctx context.Context, outs []raster.Float32Raster, dem raster.Float32Raster, e engine.Options) error {
					return HillshadeTiled(ctx, outs[0], dem, o, e)
				},
				func(ctx context.Context, outs []engine.RasterSink, dem engine.RasterSource, e engine.Options) error {
					return HillshadeChunked(ctx, outs[0], dem, o, e)
				},
				func(v float32) bool { return v >= 0 && v <= 255 },
				// Squares of larger gradients overflow float32, and
				// Inf/Inf is NaN.
				func(z [9]float64) bool {
					gx, gy := hornGrad(z)
					return math.Abs(gx) < 1e15 && math.Abs(gy) < 1e15
				},
			}
		case 4:
			o := CurvatureOptions{cs, csy, z, curv}
			optionsOK = cellsOK && curv >= CurvatureProfile && curv <= CurvatureMean
			for _, k := range ztK {
				optionsOK = optionsOK && scaleOK(k)
			}
			call = terrainCall{"curvature", 1,
				func(outs []raster.Float32Raster, dem raster.Float32Raster) { Curvature(outs[0], dem, o) },
				func(ctx context.Context, outs []raster.Float32Raster, dem raster.Float32Raster, e engine.Options) error {
					return CurvatureTiled(ctx, outs[0], dem, o, e)
				},
				func(ctx context.Context, outs []engine.RasterSink, dem engine.RasterSource, e engine.Options) error {
					return CurvatureChunked(ctx, outs[0], dem, o, e)
				},
				// A zero result is +0.
				func(v float32) bool { return math.Float32bits(v) != 1<<31 },
				// With these bounds every product and sum of the formulas
				// is finite in float32, so only overflow of a quotient
				// (to ±Inf, never NaN) remains.
				func(z [9]float64) bool {
					p, q := (z[5]-z[3])*ztK[0], (z[7]-z[1])*ztK[1]
					r := (z[3] - 2*z[4] + z[5]) * ztK[2]
					s := (z[0] - z[2] - z[6] + z[8]) * ztK[3]
					t := (z[1] - 2*z[4] + z[7]) * ztK[4]
					return math.Abs(p) < 1e9 && math.Abs(q) < 1e9 &&
						math.Abs(r) < 1e18 && math.Abs(s) < 1e18 && math.Abs(t) < 1e18
				},
			}
		case 5:
			// Ruggedness takes no cell size, so none of those rules apply.
			o := RuggednessOptions{rug}
			optionsOK = rug >= RuggednessTRI && rug <= RuggednessRoughness
			call = terrainCall{"ruggedness", 1,
				func(outs []raster.Float32Raster, dem raster.Float32Raster) { Ruggedness(outs[0], dem, o) },
				func(ctx context.Context, outs []raster.Float32Raster, dem raster.Float32Raster, e engine.Options) error {
					return RuggednessTiled(ctx, outs[0], dem, o, e)
				},
				func(ctx context.Context, outs []engine.RasterSink, dem engine.RasterSource, e engine.Options) error {
					return RuggednessChunked(ctx, outs[0], dem, o, e)
				},
				// All but TPI are at least +0, never -0.
				func(v float32) bool { return rug == RuggednessTPI || v >= 0 && math.Float32bits(v) != 1<<31 },
				// Elevations within 1e37 keep every difference and sum
				// below float32's limit.
				func([9]float64) bool { return true },
			}
		}

		// Operands.
		w, h := d.Range(1, 40), d.Range(1, 7)
		newRoot := func(masked bool, values bool) fuzzRoot {
			rootW, rootH, x, y := w, h, 0, 0
			layout := d.IntN(3) // compact, strided, window
			if layout == 2 {
				x, y = d.IntN(w+2), d.IntN(h+3)
				rootW, rootH = w+x+d.IntN(4), h+y+d.IntN(h+3)
			}
			stride := rootW
			if layout > 0 {
				stride += d.IntN(70)
			}
			n := (rootH-1)*stride + rootW
			r := raster.NewFloat32Stride(rootW, rootH, stride, make([]float32, n))
			for i := range r.Data {
				if values {
					r.Data[i] = d.Float32()
				} else {
					r.Data[i] = float32(d.IntN(100))
				}
			}
			if masked {
				off := d.IntN(130)
				r.Valid = make([]uint64, raster.MaskWords(off+n)+d.IntN(2))
				r.ValidOffset = off
				for k := range r.Valid {
					r.Valid[k] = d.Dense(3)
				}
			}
			return fuzzRoot{r, x, y}
		}
		demMasked := d.Bool()
		roots := []fuzzRoot{newRoot(demMasked, true)}
		outRoot := make([]int, call.outputs)
		outPos := make([][2]int, call.outputs)
		for i := range call.outputs {
			masked := d.IntN(8) != 0
			if !demMasked {
				masked = d.Bool()
			}
			roots = append(roots, newRoot(masked, false))
			outRoot[i], outPos[i] = len(roots)-1, [2]int{roots[len(roots)-1].x, roots[len(roots)-1].y}
		}
		// Sometimes the first output is a window of the DEM's root.
		if dr := roots[0].r; d.IntN(8) == 0 {
			outRoot[0], outPos[0] = 0, [2]int{d.IntN(dr.Width - w + 1), d.IntN(dr.Height - h + 1)}
		}
		opts := engine.Options{TileWidth: d.Range(0, w+2), TileHeight: d.Range(0, h+2), Workers: d.Range(0, 4)}
		id := fmt.Sprintf("%s %d×%d cs %v csy %v z %v units %d az %v alt %v flat %v trig %v opts %+v",
			call.name, w, h, cs, csy, z, units, az, alt, zeroFlat, trig, opts)

		// Operand expectations.
		missingMask := false
		for i := range call.outputs {
			missingMask = missingMask || demMasked && roots[outRoot[i]].r.Valid == nil
		}
		demRoot := roots[0].r
		span := (h-1)*demRoot.Stride + w
		demStart := roots[0].y*demRoot.Stride + roots[0].x
		overlapping, wordsMeet := false, false
		if outRoot[0] == 0 {
			p := outPos[0][1]*demRoot.Stride + outPos[0][0]
			overlapping = p < demStart+span && demStart < p+span
			wp, wq := demRoot.ValidOffset+p, demRoot.ValidOffset+demStart
			wordsMeet = demRoot.Valid != nil && wp>>6 <= (wq+span-1)>>6 && wq>>6 <= (wp+span-1)>>6
		}

		views := func(rs []fuzzRoot) (dem raster.Float32Raster, outs []raster.Float32Raster) {
			dem = rs[0].r.Window(rs[0].x, rs[0].y, w, h)
			for i := range call.outputs {
				outs = append(outs, rs[outRoot[i]].r.Window(outPos[i][0], outPos[i][1], w, h))
			}
			return dem, outs
		}

		// Reference values: a scalar run on a compact, unmasked copy.
		var ref []raster.Float32Raster
		dem0, _ := views(roots)
		if optionsOK {
			compact := raster.NewFloat32(w, h, make([]float32, w*h))
			for y := range h {
				copy(compact.Row(y), dem0.Row(y))
			}
			for range call.outputs {
				ref = append(ref, raster.NewFloat32(w, h, make([]float32, w*h)))
			}
			stencil.UseScalar(true)
			call.plain(ref, compact)
			stencil.UseScalar(false)
		}

		type run struct {
			name string
			call func(dem raster.Float32Raster, outs []raster.Float32Raster)
			// panics: 1 must, -1 must not, 0 may.
			panics int
		}
		must := func(p bool) int {
			if p {
				return 1
			}
			return -1
		}
		noErr := func(err error) {
			if err != nil {
				t.Fatalf("%s: %v", id, err)
			}
		}
		chunkedPanics := must(!optionsOK || missingMask || overlapping)
		if chunkedPanics < 0 && wordsMeet {
			chunkedPanics = 0
		}
		runs := []run{
			{"plain", func(dem raster.Float32Raster, outs []raster.Float32Raster) { call.plain(outs, dem) },
				must(!optionsOK || missingMask || overlapping)},
			{"tiled", func(dem raster.Float32Raster, outs []raster.Float32Raster) { noErr(call.tiled(ctx, outs, dem, opts)) },
				must(!optionsOK || missingMask || overlapping)},
			{"chunked", func(dem raster.Float32Raster, outs []raster.Float32Raster) {
				sinks := make([]engine.RasterSink, len(outs))
				for i, o := range outs {
					sinks[i] = engine.NewMemorySink(o)
				}
				noErr(call.chunked(ctx, sinks, engine.NewMemorySource(dem), opts))
			}, chunkedPanics},
		}

		var first []fuzzRoot
		for _, rn := range runs {
			rs := make([]fuzzRoot, len(roots))
			for i := range roots {
				rs[i] = roots[i].clone()
			}
			dem, outs := views(rs)
			rid := id + " " + rn.name
			msg := func() (msg string) {
				defer func() {
					if v := recover(); v != nil {
						s, ok := v.(string)
						if !ok {
							t.Fatalf("%s: panic %v (%T), want a message", rid, v, v)
						}
						msg = s
					}
				}()
				rn.call(dem, outs)
				return ""
			}()
			if msg != "" {
				want := "engine: "
				if !optionsOK {
					want = "terrain: "
				}
				if !strings.HasPrefix(msg, want) {
					t.Fatalf("%s: panic %q, want prefix %q", rid, msg, want)
				}
			}
			if rn.panics != 0 && (msg != "") != (rn.panics > 0) {
				t.Fatalf("%s: panic %q, want panic %v (options ok %v, missing mask %v, overlapping %v)",
					rid, msg, rn.panics > 0, optionsOK, missingMask, overlapping)
			}

			// Nothing outside the outputs' cells changes, and nothing at
			// all if the call panicked.
			for ri := range roots {
				orig, got := roots[ri].r, rs[ri].r
				inOut := func(i int) bool {
					if msg != "" {
						return false
					}
					for k := range call.outputs {
						if outRoot[k] != ri {
							continue
						}
						j := i - (outPos[k][1]*orig.Stride + outPos[k][0])
						if j >= 0 && j/orig.Stride < h && j%orig.Stride < w {
							return true
						}
					}
					return false
				}
				for i := range orig.Data {
					if !inOut(i) && math.Float32bits(orig.Data[i]) != math.Float32bits(got.Data[i]) {
						t.Fatalf("%s: root %d cell %d outside the outputs changed (panic %q)", rid, ri, i, msg)
					}
				}
				for bit := range len(orig.Valid) * 64 {
					i := bit - orig.ValidOffset
					if (i < 0 || i >= len(orig.Data) || !inOut(i)) && raster.MaskGet(orig.Valid, bit) != raster.MaskGet(got.Valid, bit) {
						t.Fatalf("%s: root %d mask bit %d outside the outputs changed (panic %q)", rid, ri, bit, msg)
					}
				}
			}
			if msg != "" {
				continue
			}

			for k, out := range outs {
				for y := range h {
					for x := range w {
						v := out.Data[out.Index(x, y)]
						if x == 0 || y == 0 || x == w-1 || y == h-1 {
							if v == v || out.Valid != nil && out.IsValid(x, y) {
								t.Fatalf("%s: output %d border cell (%d, %d) = %v valid %v, want invalid NaN", rid, k, x, y, v, out.IsValid(x, y))
							}
							continue
						}
						valid, moderate := true, true
						var z [9]float64
						for j := -1; j <= 1; j++ {
							for i := -1; i <= 1; i++ {
								valid = valid && dem0.IsValid(x+i, y+j)
								e := dem0.Data[dem0.Index(x+i, y+j)]
								moderate = moderate && math.Abs(float64(e)) <= 1e37
								z[(j+1)*3+i+1] = float64(e)
							}
						}
						if got := out.IsValid(x, y); got != valid {
							t.Fatalf("%s: output %d cell (%d, %d) valid = %v, want %v", rid, k, x, y, got, valid)
						}
						if !valid {
							continue
						}
						if r := ref[k].Data[ref[k].Index(x, y)]; !sameBits(v, r) {
							t.Fatalf("%s: output %d cell (%d, %d) = %v (%#08x), scalar compact run %v (%#08x)",
								rid, k, x, y, v, math.Float32bits(v), r, math.Float32bits(r))
						}
						if v == v && !call.inRange(v) {
							t.Fatalf("%s: output %d cell (%d, %d) = %v, outside the documented range", rid, k, x, y, v)
						}
						if v != v && moderate && call.finite(z) {
							t.Fatalf("%s: output %d cell (%d, %d) is NaN for elevations %v", rid, k, x, y, z)
						}
					}
				}
			}

			if first == nil {
				first = rs
				continue
			}
			for ri := range rs {
				for i := range rs[ri].r.Data {
					if !sameBits(rs[ri].r.Data[i], first[ri].r.Data[i]) {
						t.Fatalf("%s: root %d cell %d = %v, plain wrote %v", rid, ri, i, rs[ri].r.Data[i], first[ri].r.Data[i])
					}
				}
				for k := range rs[ri].r.Valid {
					if rs[ri].r.Valid[k] != first[ri].r.Valid[k] {
						t.Fatalf("%s: root %d mask word %d = %#x, plain wrote %#x", rid, ri, k, rs[ri].r.Valid[k], first[ri].r.Valid[k])
					}
				}
			}
		}
	})
}
