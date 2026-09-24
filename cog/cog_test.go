package cog

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
)

// sampleType is one of the sample types the reader supports.
type sampleType struct {
	name         string
	format, size int
}

var sampleTypes = []sampleType{
	{"uint8", sampleUint, 1}, {"int8", sampleInt, 1},
	{"uint16", sampleUint, 2}, {"int16", sampleInt, 2},
	{"uint32", sampleUint, 4}, {"int32", sampleInt, 4},
	{"float32", sampleFloat, 4}, {"float64", sampleFloat, 8},
}

// randomValue draws a value the type holds exactly, favouring its
// extremes, which catch sign and width mistakes.
func randomValue(rng *rand.Rand, t sampleType) float64 {
	if t.format == sampleFloat {
		switch rng.IntN(12) {
		case 0:
			return math.Inf(1 - 2*rng.IntN(2))
		case 1:
			return math.Copysign(0, -1)
		case 2:
			if t.size == 8 {
				return 1e300 * float64(1-2*rng.IntN(2)) // beyond float32
			}
		}
		v := rng.NormFloat64() * 1000
		if t.size == 4 {
			v = float64(float32(v))
		}
		return v
	}
	bits := 8 * t.size
	lo, hi := 0.0, math.Exp2(float64(bits))-1
	if t.format == sampleInt {
		lo, hi = -math.Exp2(float64(bits-1)), math.Exp2(float64(bits-1))-1
	}
	switch rng.IntN(8) {
	case 0:
		return lo
	case 1:
		return hi
	}
	return math.Floor(lo + rng.Float64()*(hi-lo+1))
}

// expected is what reading v should give: v rounded to float32, which
// overflows to ±Inf, as GDAL reads it (acceptance/cogcheck.sh measures
// that against GDAL on a file of values near 1e300).
func expected(v float64) float32 {
	if math.Abs(v) > math.MaxFloat32 && !math.IsInf(v, 0) {
		// Every value randomValue draws past MaxFloat32 is far past it.
		return float32(math.Copysign(math.Inf(1), v))
	}
	return float32(v)
}

func TestToFloat32(t *testing.T) {
	maxUlp := 0x1p104 // float32's ulp at MaxFloat32
	for _, c := range []struct {
		v    float64
		want float32
	}{
		{math.MaxFloat32, math.MaxFloat32},
		{math.MaxFloat32 + maxUlp/2 - 0x1p80, math.MaxFloat32}, // below the tie (float64 spacing here is 2^76)
		{math.MaxFloat32 + maxUlp/2, float32(math.Inf(1))},     // the tie: to even, Inf
		{-math.MaxFloat32 - maxUlp/2, float32(math.Inf(-1))},
		{1e300, float32(math.Inf(1))},
		{math.Inf(-1), float32(math.Inf(-1))},
		{0.1, 0.1},
	} {
		if got := toFloat32(c.v); math.Float32bits(got) != math.Float32bits(c.want) {
			t.Errorf("toFloat32(%g) = %g, want %g", c.v, got, c.want)
		}
	}
}

// layout is a block layout to test.
type layout struct {
	name           string
	tiled          bool
	blockW, blockH int
	planar         int
}

var layouts = []layout{
	{"tiles16x8", true, 16, 8, planarChunky},
	{"tiles16x8-planar", true, 16, 8, planarSeparate},
	{"strips5", false, 0, 5, planarChunky},
	{"strips5-planar", false, 0, 5, planarSeparate},
	{"onestrip", false, 0, 1 << 20, planarChunky},
}

var compressions = []uint64{compressionNone, compressionDeflate, compressionPackBits, compressionZSTD}

// TestMatrix writes files across byte order, BigTIFF, layout, compression,
// predictor and sample type, some with NoData, and reads every band back
// whole and in random windows.
func TestMatrix(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	const w, h, bands = 37, 23, 3
	files := 0
	for _, order := range []order{binary.LittleEndian, binary.BigEndian} {
		for _, big := range []bool{false, true} {
			for _, lay := range layouts {
				for _, comp := range compressions {
					for _, st := range sampleTypes {
						preds := []int{predictorNone, predictorHorizontal}
						if st.format == sampleFloat {
							preds = append(preds, predictorFloat)
						}
						for _, pred := range preds {
							name := fmt.Sprintf("%v/big=%v/%s/%s/%s/pred%d", order, big, lay.name,
								compressionName(comp), st.name, pred)
							withND := rng.IntN(2) == 0
							checkRoundTrip(t, name, rng, order, big, lay, comp, st, pred, w, h, bands, withND)
							files++
						}
					}
				}
			}
		}
	}
	t.Logf("%d files", files)
}

func checkRoundTrip(t *testing.T, name string, rng *rand.Rand, order order, big bool, lay layout,
	comp uint64, st sampleType, pred int, w, h, bands int, withND bool) {
	t.Helper()
	vals := make([][]float64, bands)
	for b := range vals {
		vals[b] = make([]float64, w*h)
		for i := range vals[b] {
			vals[b][i] = randomValue(rng, st)
		}
	}
	sp := imageSpec{w: w, h: h, bands: bands, format: st.format, size: st.size,
		tiled: lay.tiled, blockW: lay.blockW, blockH: lay.blockH, planar: lay.planar,
		compression: comp, predictor: pred, vals: vals}
	var nd float64
	if withND {
		nd = randomValue(rng, st)
		if math.IsInf(nd, 0) || math.Abs(nd) > math.MaxFloat32 {
			nd = 0
		}
		for b := range vals {
			for range 20 {
				vals[b][rng.IntN(w*h)] = nd
			}
		}
		sp.extra = append(sp.extra, tagValue{tag: tagGDALNoData, typ: typeASCII, str: fmt.Sprint(nd)})
	}
	file := fileSpec{order: order, big: big, images: []imageSpec{sp}}.write()
	f, err := Open(bytes.NewReader(file))
	if err != nil {
		t.Fatalf("%s: Open: %v", name, err)
	}
	if gw, gh := f.Size(0); gw != w || gh != h || f.Bands() != bands || f.Levels() != 1 {
		t.Fatalf("%s: %d×%d, %d bands, %d levels", name, gw, gh, f.Bands(), f.Levels())
	}
	for b := range bands {
		src, err := f.Source(SourceOptions{Band: b, CacheBytes: int64(rng.IntN(3) - 1)})
		if err != nil {
			t.Fatalf("%s: Source: %v", name, err)
		}
		if src.Masked() != withND {
			t.Fatalf("%s: Masked() = %v", name, src.Masked())
		}
		checkWindow(t, name, src, vals[b], w, 0, 0, w, h, withND, nd)
		for range 5 {
			x, y := rng.IntN(w), rng.IntN(h)
			checkWindow(t, name, src, vals[b], w, x, y, 1+rng.IntN(w-x), 1+rng.IntN(h-y), withND, nd)
		}
	}
}

// checkWindow reads the ww×wh window at (x, y) into a padded destination
// whose mask starts at an odd bit, and compares it with vals.
func checkWindow(t *testing.T, name string, src *Source, vals []float64, w, x, y, ww, wh int, withND bool, nd float64) {
	t.Helper()
	stride := ww + 3
	dst := raster.NewFloat32Stride(ww, wh, stride, make([]float32, (wh-1)*stride+ww))
	dst.Valid = raster.NewMask(dst.Stride*wh + 70)
	dst.ValidOffset = 5
	if err := src.ReadWindow(context.Background(), dst, x, y); err != nil {
		t.Fatalf("%s: ReadWindow(%d, %d, %d×%d): %v", name, x, y, ww, wh, err)
	}
	for j := range wh {
		for i := range ww {
			v := vals[(y+j)*w+x+i]
			valid := !withND || v != nd
			if dst.IsValid(i, j) != valid {
				t.Fatalf("%s: cell (%d, %d) = %v: valid %v, want %v", name, x+i, y+j, v, dst.IsValid(i, j), valid)
			}
			got, want := dst.Data[dst.Index(i, j)], expected(v)
			if valid && math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("%s: cell (%d, %d) = %v, want %v", name, x+i, y+j, got, want)
			}
		}
	}
}

// grid16 is a 16×12 float32 image in 8×4 tiles, value x + 100 y.
func grid16(extra ...tagValue) imageSpec {
	w, h := 16, 12
	v := make([]float64, w*h)
	for i := range v {
		v[i] = float64(i%w + 100*(i/w))
	}
	return imageSpec{w: w, h: h, bands: 1, format: sampleFloat, size: 4, tiled: true,
		blockW: 8, blockH: 4, planar: planarChunky, compression: compressionNone,
		vals: [][]float64{v}, extra: extra}
}

func geoTags(modelType, rasterType, code uint64) tagValue {
	key := uint64(keyProjectedType)
	if modelType == modelGeographic {
		key = keyGeographicType
	}
	return tagValue{tag: tagGeoKeyDirectory, typ: typeShort, uints: []uint64{
		1, 1, 0, 3,
		keyModelType, 0, 1, modelType,
		keyRasterType, 0, 1, rasterType,
		key, 0, 1, code,
	}}
}

func scaleTiepoint(sx, sy, x, y float64) []tagValue {
	return []tagValue{
		{tag: tagModelPixelScale, typ: typeDouble, floats: []float64{sx, sy, 0}},
		{tag: tagModelTiepoint, typ: typeDouble, floats: []float64{0, 0, 0, x, y, 0}},
	}
}

// TestGeoreferencing checks the grid against what GDAL reports for the
// same tags: the tiepoint is the outer corner for PixelIsArea and a cell
// centre for PixelIsPoint.
func TestGeoreferencing(t *testing.T) {
	cases := []struct {
		name  string
		tags  []tagValue
		want  raster.Grid
		geo   bool
		error string
	}{
		{
			name: "area",
			tags: append(scaleTiepoint(10, 10, 500000, 7000000), geoTags(modelProjected, 1, 32633)),
			want: raster.Grid{Width: 16, Height: 12, ResolutionX: 10, ResolutionY: -10,
				OriginX: 500000, OriginY: 7000000, CRS: raster.CRS{Code: "EPSG:32633"}},
			geo: true,
		},
		{
			name: "point",
			tags: append(scaleTiepoint(10, 10, 500000, 7000000), geoTags(modelProjected, rasterPixelIsPt, 25833)),
			want: raster.Grid{Width: 16, Height: 12, ResolutionX: 10, ResolutionY: -10,
				OriginX: 499995, OriginY: 7000005, CRS: raster.CRS{Code: "EPSG:25833"}},
			geo: true,
		},
		{
			name: "transform",
			tags: []tagValue{
				{tag: tagModelTransform, typ: typeDouble, floats: []float64{
					0.5, 0, 0, 10, 0, -0.25, 0, 60, 0, 0, 0, 0, 0, 0, 0, 1}},
				geoTags(modelGeographic, 1, 4326),
			},
			want: raster.Grid{Width: 16, Height: 12, ResolutionX: 0.5, ResolutionY: -0.25,
				OriginX: 10, OriginY: 60, CRS: raster.CRS{Code: "EPSG:4326"}},
			geo: true,
		},
		{
			name: "user-defined CRS",
			tags: append(scaleTiepoint(1, 1, 0, 0), geoTags(modelProjected, 1, userDefined)),
			want: raster.Grid{Width: 16, Height: 12, ResolutionX: 1, ResolutionY: -1},
			geo:  true,
		},
		{
			name: "none",
			want: raster.Grid{Width: 16, Height: 12, ResolutionX: 1, ResolutionY: 1},
		},
		{
			name: "rotated",
			tags: []tagValue{{tag: tagModelTransform, typ: typeDouble, floats: []float64{
				1, 0.1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}}},
			error: "rotated",
		},
	}
	for _, c := range cases {
		file := fileSpec{order: binary.LittleEndian, images: []imageSpec{grid16(c.tags...)}}.write()
		f, err := Open(bytes.NewReader(file))
		if c.error != "" {
			if err == nil || !strings.Contains(err.Error(), c.error) {
				t.Errorf("%s: Open error %v, want one mentioning %q", c.name, err, c.error)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if g := f.Grid(0); g != c.want || f.Georeferenced() != c.geo {
			t.Errorf("%s: grid %+v georeferenced %v, want %+v %v", c.name, g, f.Georeferenced(), c.want, c.geo)
		}
	}
}

// TestOverviews builds a COG-shaped file: the full image, a mask, and two
// overviews, and checks that the mask is not taken for a level and each
// level reads its own cells on a grid scaled as GDAL scales it. The mask
// has no holes here; TestMasks reads masks.
func TestOverviews(t *testing.T) {
	full := grid16(append(scaleTiepoint(10, 10, 1000, 2000), geoTags(modelProjected, 1, 3006))...)
	mask := maskOf(full, 8, func(int, int) bool { return false })
	ov1 := halve(full)
	ov1.subfile = subfileReduced
	ov2 := halve(ov1)
	ov2.subfile = subfileReduced
	file := fileSpec{order: binary.BigEndian, big: true, images: []imageSpec{full, mask, ov1, ov2}}.write()
	f, err := Open(bytes.NewReader(file))
	if err != nil {
		t.Fatal(err)
	}
	if f.Levels() != 3 {
		t.Fatalf("%d levels, want 3", f.Levels())
	}
	for lvl, sp := range []imageSpec{full, ov1, ov2} {
		g := f.Grid(lvl)
		scale := float64(full.w) / float64(sp.w)
		want := raster.Grid{Width: sp.w, Height: sp.h, ResolutionX: 10 * scale, ResolutionY: -10 * float64(full.h) / float64(sp.h),
			OriginX: 1000, OriginY: 2000, CRS: raster.CRS{Code: "EPSG:3006"}}
		if g != want {
			t.Errorf("level %d: grid %+v, want %+v", lvl, g, want)
		}
		src, err := f.Source(SourceOptions{Level: lvl})
		if err != nil {
			t.Fatal(err)
		}
		if src.Grid() != g {
			t.Errorf("level %d: Source.Grid %+v", lvl, src.Grid())
		}
		checkWindow(t, fmt.Sprintf("level %d", lvl), src, sp.vals[0], sp.w, 0, 0, sp.w, sp.h, false, 0)
	}
}

// halve returns an image of every other cell of sp, as an overview's
// stand-in: its values only need to differ from the full image's.
func halve(sp imageSpec) imageSpec {
	w, h := (sp.w+1)/2, (sp.h+1)/2
	v := make([]float64, w*h)
	for y := range h {
		for x := range w {
			v[y*w+x] = sp.vals[0][2*y*sp.w+2*x] + 0.5
		}
	}
	o := sp
	o.w, o.h, o.vals, o.extra = w, h, [][]float64{v}, nil
	return o
}

// TestSparse checks blocks with offset and count 0: NoData with a NoData
// value, 0 without, as GDAL reads them.
func TestSparse(t *testing.T) {
	for _, withND := range []bool{false, true} {
		sp := grid16()
		sp.sparse = func(b int) bool { return b == 1 || b == 4 }
		if withND {
			sp.extra = []tagValue{{tag: tagGDALNoData, typ: typeASCII, str: "-9999"}}
		}
		f, err := Open(bytes.NewReader(fileSpec{order: binary.LittleEndian, images: []imageSpec{sp}}.write()))
		if err != nil {
			t.Fatal(err)
		}
		src, _ := f.Source(SourceOptions{})
		dst := raster.NewFloat32(16, 12, make([]float32, 16*12))
		dst.Valid = raster.NewMask(16 * 12)
		if err := src.ReadWindow(context.Background(), dst, 0, 0); err != nil {
			t.Fatal(err)
		}
		for y := range 12 {
			for x := range 16 {
				sparse := (y/4)*2+x/8 == 1 || (y/4)*2+x/8 == 4
				switch {
				case sparse && withND:
					if dst.IsValid(x, y) {
						t.Fatalf("nodata=%v: sparse cell (%d, %d) is valid", withND, x, y)
					}
				case sparse:
					if !dst.IsValid(x, y) || dst.Data[dst.Index(x, y)] != 0 {
						t.Fatalf("nodata=%v: sparse cell (%d, %d) = %v", withND, x, y, dst.Data[dst.Index(x, y)])
					}
				default:
					if !dst.IsValid(x, y) || dst.Data[dst.Index(x, y)] != float32(x+100*y) {
						t.Fatalf("nodata=%v: cell (%d, %d) = %v", withND, x, y, dst.Data[dst.Index(x, y)])
					}
				}
			}
		}
	}
}

// TestNoDataGDAL checks that cells compare with NoData as GDAL's NoData
// mask compares them (gcore/gdalnodatamaskband.cpp), which
// acceptance/corpus's generated-gdal/*nodata* files confirm:
//
//   - integers: a NoData value outside the type's range is none at all
//     (not Masked), and one inside it is truncated toward zero;
//   - floats: equal within ARE_REAL_EQUAL's 2·FLT_EPSILON·|a+b|, which
//     is four float32 ulps either side of -9999 and 2^30 float64 ulps,
//     computed in float32 for float32 samples, so that a sum that
//     overflows makes every such value equal;
//   - a float32 NoData is first rounded to float32, a value just past
//     MaxFloat32 onto it and one beyond float32's range to Inf.
//
// The first version of this test expected exact comparison in the
// sample type, fractional integer NoData matching nothing, and 1e39 on
// float32 matching nothing. GDAL does none of those.
func TestNoDataGDAL(t *testing.T) {
	f32ulps := func(v float32, n int32) float64 {
		return float64(math.Float32frombits(uint32(int32(math.Float32bits(v)) + n)))
	}
	f64ulps := func(v float64, n int64) float64 {
		return math.Float64frombits(uint64(int64(math.Float64bits(v)) + n))
	}
	f32, f64 := sampleType{"f32", sampleFloat, 4}, sampleType{"f64", sampleFloat, 8}
	cases := []struct {
		nd     string
		st     sampleType
		vals   []float64
		valid  []bool
		masked bool
	}{
		{"1e-9", f32, []float64{float64(float32(1e-9)), 1e-9 * 2}, []bool{false, true}, true},
		{"1e-9", f64, []float64{float64(float32(1e-9)), 1e-9, 1.000001e-9}, []bool{false, false, true}, true},
		{"nan", f32, []float64{math.NaN(), 1}, []bool{false, true}, true},
		{"nan", sampleType{"i16", sampleInt, 2}, []float64{0, 1}, []bool{true, true}, false},
		{"2.5", sampleType{"u8", sampleUint, 1}, []float64{2, 3}, []bool{false, true}, true},
		{"-3.7", sampleType{"i16", sampleInt, 2}, []float64{-3, -4}, []bool{false, true}, true},
		{"300", sampleType{"u8", sampleUint, 1}, []float64{255, 44}, []bool{true, true}, false},
		{"-1", sampleType{"i8", sampleInt, 1}, []float64{-1, 255 - 256 + 1}, []bool{false, true}, true},
		{"1e39", f32, []float64{math.Inf(1), 1, math.MaxFloat32}, []bool{false, true, true}, true},
		// Near MaxFloat32, a+b overflows float32 to Inf, and ARE_REAL_EQUAL
		// then holds for any value whose sum with NoData overflows.
		{"3.4028234663852886e+38", f32, []float64{math.MaxFloat32, 3e38, 1e30}, []bool{false, false, true}, true},
		{"-9999", f32, []float64{f32ulps(-9999, 4), f32ulps(-9999, -4), f32ulps(-9999, 5), f32ulps(-9999, -5)},
			[]bool{false, false, true, true}, true},
		{"-9999", f64, []float64{f64ulps(-9999, 1<<30), f64ulps(-9999, -1<<30), f64ulps(-9999, 1<<32)},
			[]bool{false, false, true}, true},
	}
	for _, c := range cases {
		sp := imageSpec{w: len(c.vals), h: 1, bands: 1, format: c.st.format, size: c.st.size, blockH: 1,
			planar: planarChunky, compression: compressionNone, vals: [][]float64{c.vals},
			extra: []tagValue{{tag: tagGDALNoData, typ: typeASCII, str: c.nd}}}
		f, err := Open(bytes.NewReader(fileSpec{order: binary.LittleEndian, images: []imageSpec{sp}}.write()))
		if err != nil {
			t.Fatal(err)
		}
		src, _ := f.Source(SourceOptions{})
		if src.Masked() != c.masked {
			t.Errorf("NoData %s on %s: Masked %v", c.nd, c.st.name, src.Masked())
		}
		dst := raster.NewFloat32(len(c.vals), 1, make([]float32, len(c.vals)))
		dst.Valid = raster.NewMask(len(c.vals))
		if err := src.ReadWindow(context.Background(), dst, 0, 0); err != nil {
			t.Fatal(err)
		}
		for i, want := range c.valid {
			if dst.IsValid(i, 0) != want {
				t.Errorf("NoData %s on %s: cell %v valid %v, want %v", c.nd, c.st.name, c.vals[i], !want, want)
			}
		}
	}
}

// TestChunkedSlope runs terrain's bounded-memory slope over a COG source
// and requires the bits of the same slope computed in memory, for tile
// heights that do and do not line up with the file's tiles.
func TestChunkedSlope(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	const w, h = 150, 97
	vals := make([]float64, w*h)
	for i := range vals {
		x, y := float64(i%w), float64(i/w)
		vals[i] = float64(float32(100*math.Sin(x/17)*math.Cos(y/11) + rng.Float64()))
	}
	for range 200 {
		vals[rng.IntN(w*h)] = -9999
	}
	sp := imageSpec{w: w, h: h, bands: 1, format: sampleFloat, size: 4, tiled: true, blockW: 32, blockH: 32,
		planar: planarChunky, compression: compressionDeflate, predictor: predictorFloat, vals: [][]float64{vals},
		extra: []tagValue{{tag: tagGDALNoData, typ: typeASCII, str: "-9999"}}}
	f, err := Open(bytes.NewReader(fileSpec{order: binary.LittleEndian, images: []imageSpec{sp}}.write()))
	if err != nil {
		t.Fatal(err)
	}

	dem := raster.NewFloat32(w, h, make([]float32, w*h))
	dem.Valid = raster.NewMask(w * h)
	for i, v := range vals {
		dem.Data[i] = float32(v)
		raster.MaskSet(dem.Valid, i, v != -9999)
	}
	opts := terrain.SlopeOptions{CellSize: 10}
	want := raster.NewFloat32Like(dem)
	terrain.Slope(want, dem, opts)

	for _, eo := range []engine.Options{
		{TileHeight: 32, Workers: 1},
		{TileHeight: 7, Workers: 4},
		{TileHeight: 50, TileWidth: 33, Workers: 3},
	} {
		src, err := f.Source(SourceOptions{CacheBytes: 16 << 10})
		if err != nil {
			t.Fatal(err)
		}
		got := raster.NewFloat32Like(dem)
		if err := terrain.SlopeChunked(context.Background(), engine.NewMemorySink(got), src, opts, eo); err != nil {
			t.Fatal(err)
		}
		for i := range w * h {
			gv, wv := raster.MaskGet(got.Valid, i), raster.MaskGet(want.Valid, i)
			if gv != wv || (wv && math.Float32bits(got.Data[i]) != math.Float32bits(want.Data[i])) {
				t.Fatalf("%+v: cell %d: %v (valid %v), want %v (valid %v)", eo, i, got.Data[i], gv, want.Data[i], wv)
			}
		}
	}
}

// TestConcurrent reads random windows from many goroutines through a
// cache small enough to evict constantly; run it with -race.
func TestConcurrent(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	const w, h = 64, 48
	st := sampleType{"int16", sampleInt, 2}
	vals := make([]float64, w*h)
	for i := range vals {
		vals[i] = randomValue(rng, st)
	}
	sp := imageSpec{w: w, h: h, bands: 1, format: st.format, size: st.size, tiled: true, blockW: 16, blockH: 16,
		planar: planarChunky, compression: compressionZSTD, predictor: predictorHorizontal, vals: [][]float64{vals}}
	f, err := Open(bytes.NewReader(fileSpec{order: binary.LittleEndian, images: []imageSpec{sp}}.write()))
	if err != nil {
		t.Fatal(err)
	}
	// Block buffers are reused once released (see block), so readers
	// that race an eviction are the case to catch: a buffer reused while
	// one of them still copies from it would show up as wrong cells. A
	// cache of about two blocks evicts constantly; none releases every
	// block as soon as its reader is done.
	for _, cacheBytes := range []int64{3000, -1} {
		src, _ := f.Source(SourceOptions{CacheBytes: cacheBytes})
		var wg sync.WaitGroup
		for g := range 8 {
			wg.Go(func() {
				rng := rand.New(rand.NewPCG(uint64(g), 7))
				for range 50 {
					x, y := rng.IntN(w), rng.IntN(h)
					checkWindow(t, fmt.Sprintf("concurrent, cache %d", cacheBytes), src, vals, w, x, y,
						1+rng.IntN(w-x), 1+rng.IntN(h-y), false, 0)
				}
			})
		}
		wg.Wait()
	}
}

// failingReader fails every ReadAt past a byte offset.
type failingReader struct {
	r     io.ReaderAt
	after int64
}

var errInjected = errors.New("injected IO fault")

func (f failingReader) ReadAt(p []byte, off int64) (int, error) {
	if off+int64(len(p)) > f.after {
		return 0, errInjected
	}
	return f.r.ReadAt(p, off)
}

// TestFaults checks that IO failures come back as errors that still
// match the reader's error, whether they hit the IFDs or a block, and
// that a failed block is retried rather than cached.
func TestFaults(t *testing.T) {
	sp := grid16()
	sp.compression = compressionDeflate
	file := fileSpec{order: binary.LittleEndian, images: []imageSpec{sp}}.write()
	if _, err := Open(failingReader{bytes.NewReader(file), int64(len(file) - 10)}); !errors.Is(err, errInjected) {
		t.Errorf("Open with a fault in the IFD: %v", err)
	}
	// The blocks come first in these files, the IFD last, so a fault
	// just past the first block fails later blocks but not Open.
	fr := &switchReader{r: bytes.NewReader(file)}
	f, err := Open(fr)
	if err != nil {
		t.Fatal(err)
	}
	src, _ := f.Source(SourceOptions{})
	dst := raster.NewFloat32(16, 12, make([]float32, 16*12))
	fr.fail = true
	if err := src.ReadWindow(context.Background(), dst, 0, 0); !errors.Is(err, errInjected) {
		t.Errorf("ReadWindow with a fault: %v", err)
	}
	fr.fail = false
	if err := src.ReadWindow(context.Background(), dst, 0, 0); err != nil {
		t.Errorf("ReadWindow after the fault cleared: %v", err)
	}

	// A truncated file: the IFD is intact but blocks point past the end.
	trunc := fileSpec{order: binary.LittleEndian, images: []imageSpec{sp}}.write()
	f, err = Open(bytes.NewReader(trunc))
	if err != nil {
		t.Fatal(err)
	}
	f.levels[0].offsets[2] = uint64(len(trunc)) + 100
	src, _ = f.Source(SourceOptions{})
	if err := src.ReadWindow(context.Background(), dst, 0, 0); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Errorf("ReadWindow past the end: %v", err)
	}
}

type switchReader struct {
	r    io.ReaderAt
	mu   sync.Mutex
	fail bool
}

func (s *switchReader) ReadAt(p []byte, off int64) (int, error) {
	s.mu.Lock()
	fail := s.fail
	s.mu.Unlock()
	if fail {
		return 0, errInjected
	}
	return s.r.ReadAt(p, off)
}

func TestCancel(t *testing.T) {
	f, err := Open(bytes.NewReader(fileSpec{order: binary.LittleEndian, images: []imageSpec{grid16()}}.write()))
	if err != nil {
		t.Fatal(err)
	}
	src, _ := f.Source(SourceOptions{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := src.ReadWindow(ctx, raster.NewFloat32(4, 4, make([]float32, 16)), 0, 0); !errors.Is(err, context.Canceled) {
		t.Errorf("ReadWindow on a cancelled context: %v", err)
	}
}

// TestUnsupported checks that what the package does not read is refused
// with an error that names it.
func TestUnsupported(t *testing.T) {
	cases := []struct {
		name  string
		edit  func(*imageSpec)
		error string
	}{
		{"JPEG", func(sp *imageSpec) { sp.compression = compressionJPEG }, "JPEG compression is not supported"},
		{"LERC", func(sp *imageSpec) { sp.compression = compressionLERC }, "LERC compression"},
		{"float16", func(sp *imageSpec) { sp.size = 2 }, "16-bit floating-point"},
		{"float predictor on ints", func(sp *imageSpec) { sp.format = sampleInt; sp.predictor = predictorFloat }, "floating-point predictor"},
	}
	for _, c := range cases {
		sp := grid16()
		c.edit(&sp)
		_, err := Open(bytes.NewReader(fileSpec{order: binary.LittleEndian, images: []imageSpec{sp}}.write()))
		if err == nil || !strings.Contains(err.Error(), c.error) {
			t.Errorf("%s: %v, want an error mentioning %q", c.name, err, c.error)
		}
	}
	if _, err := Open(strings.NewReader("GIF89a")); err == nil {
		t.Error("Open accepted a GIF")
	}
}

func TestPanics(t *testing.T) {
	sp := grid16(tagValue{tag: tagGDALNoData, typ: typeASCII, str: "0"})
	f, _ := Open(bytes.NewReader(fileSpec{order: binary.LittleEndian, images: []imageSpec{sp}}.write()))
	src, _ := f.Source(SourceOptions{})
	for name, fn := range map[string]func(){
		"outside": func() {
			d := raster.NewFloat32(4, 4, make([]float32, 16))
			d.Valid = raster.NewMask(16)
			_ = src.ReadWindow(context.Background(), d, 13, 0)
		},
		"no mask": func() { _ = src.ReadWindow(context.Background(), raster.NewFloat32(4, 4, make([]float32, 16)), 0, 0) },
		"level":   func() { f.Grid(1) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: no panic", name)
				}
			}()
			fn()
		}()
	}
	if _, err := f.Source(SourceOptions{Band: 1}); err == nil {
		t.Error("Source accepted band 1 of 1")
	}
	if _, err := f.Source(SourceOptions{Level: 1}); err == nil {
		t.Error("Source accepted level 1 of 1")
	}
}

// countingReader counts its ReadAt calls.
type countingReader struct {
	r io.ReaderAt
	n atomic.Int64
}

func (c *countingReader) ReadAt(p []byte, off int64) (int, error) {
	c.n.Add(1)
	return c.r.ReadAt(p, off)
}

// TestSharedBlocks checks that the sources of a pixel-interleaved file
// read each block once between them, whether they read one after another
// or all at once, and that a band-interleaved file's sources read their
// own blocks once each.
func TestSharedBlocks(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	const w, h, bands = 64, 48, 3
	st := sampleType{"uint16", sampleUint, 2}
	vals := make([][]float64, bands)
	for b := range vals {
		vals[b] = make([]float64, w*h)
		for i := range vals[b] {
			vals[b][i] = randomValue(rng, st)
		}
	}
	for _, planar := range []int{planarChunky, planarSeparate} {
		for _, together := range []bool{false, true} {
			// Uncompressed and big-endian, so decoding a shared block
			// in place would corrupt it for the next band.
			sp := imageSpec{w: w, h: h, bands: bands, format: st.format, size: st.size, tiled: true,
				blockW: 16, blockH: 16, planar: planar, compression: compressionNone, vals: vals}
			file := fileSpec{order: binary.BigEndian, images: []imageSpec{sp}}.write()
			cr := &countingReader{r: bytes.NewReader(file)}
			f, err := Open(cr)
			if err != nil {
				t.Fatal(err)
			}
			cr.n.Store(0)
			var wg sync.WaitGroup
			for b := range bands {
				src, err := f.Source(SourceOptions{Band: b})
				if err != nil {
					t.Fatal(err)
				}
				read := func() {
					for range 4 {
						checkWindow(t, "shared", src, vals[b], w, 0, 0, w, h, false, 0)
					}
				}
				if together {
					wg.Go(read)
				} else {
					read()
				}
			}
			wg.Wait()
			blocks := int64(len(f.levels[0].offsets))
			if got := cr.n.Load(); got != blocks {
				t.Errorf("planar %d, together %v: %d reads for %d blocks", planar, together, got, blocks)
			}
		}
	}
}

// TestSharedMaskBlocks reads a pixel-interleaved file whose per-band
// transparency mask is pixel-interleaved too, so each band's source and
// its mask source read through the File's shared cache: the image's and
// the mask's blocks have the same indexes at the same level, and must not
// be taken for each other. Each band has its own holes, and each block of
// either image is read once for all bands.
func TestSharedMaskBlocks(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	const w, h, bands, hole = 64, 48, 3, 7
	st := sampleType{"uint16", sampleUint, 2}
	isHole := func(x, y, b int) bool { return (3*x+y+5*b)%7 == 0 }
	vals := make([][]float64, bands)
	masks := make([][]float64, bands)
	for b := range vals {
		vals[b] = make([]float64, w*h)
		masks[b] = make([]float64, w*h)
		for i := range vals[b] {
			v := randomValue(rng, st)
			if v == hole {
				v++
			}
			vals[b][i], masks[b][i] = v, 255
			if isHole(i%w, i/w, b) {
				vals[b][i], masks[b][i] = hole, 0 // checkWindow's NoData stands for the mask
			}
		}
	}
	for _, together := range []bool{false, true} {
		sp := imageSpec{w: w, h: h, bands: bands, format: st.format, size: st.size, tiled: true,
			blockW: 16, blockH: 16, planar: planarChunky, compression: compressionNone, vals: vals}
		mask := imageSpec{w: w, h: h, bands: bands, format: sampleUint, size: 1, tiled: true,
			blockW: 16, blockH: 16, planar: planarChunky, compression: compressionDeflate,
			vals: masks, subfile: subfileMask}
		file := fileSpec{order: binary.BigEndian, images: []imageSpec{sp, mask}}.write()
		cr := &countingReader{r: bytes.NewReader(file)}
		f, err := Open(cr)
		if err != nil {
			t.Fatal(err)
		}
		if f.masks[0] == nil || f.masks[0].bands != bands {
			t.Fatalf("together %v: no per-band mask", together)
		}
		cr.n.Store(0)
		var wg sync.WaitGroup
		for b := range bands {
			src, err := f.Source(SourceOptions{Band: b})
			if err != nil {
				t.Fatal(err)
			}
			if !src.Masked() {
				t.Fatalf("band %d not Masked", b)
			}
			read := func() {
				for range 4 {
					checkWindow(t, fmt.Sprintf("band %d", b), src, vals[b], w, 0, 0, w, h, true, hole)
				}
			}
			if together {
				wg.Go(read)
			} else {
				read()
			}
		}
		wg.Wait()
		blocks := int64(len(f.levels[0].offsets) + len(f.masks[0].offsets))
		if got := cr.n.Load(); got != blocks {
			t.Errorf("together %v: %d reads for %d blocks", together, got, blocks)
		}
	}
}
