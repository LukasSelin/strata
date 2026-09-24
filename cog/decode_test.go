package cog

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/LukasSelin/strata/cog/internal/kern"
	"github.com/LukasSelin/strata/raster"
)

// The decoder's fast paths, each against the plain loop it replaced.
// acceptance/cogcheck.sh judges the whole reader against GDAL; these
// pin the pieces, at the lengths and strides the files do not reach.

// floatPredictorRowRef is the floating-point predictor as libtiff's
// fpAcc writes it: bytewise differencing stride bytes apart, then the
// byte planes put back together, most significant first.
func floatPredictorRowRef(row []byte, size, stride int) {
	for i := stride; i < len(row); i++ {
		row[i] += row[i-stride]
	}
	tmp := slices.Clone(row)
	n := len(row) / size
	for i := range n {
		for k := range size {
			row[i*size+k] = tmp[(size-1-k)*n+i]
		}
	}
}

func TestFloatPredictorRow(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, size := range []int{2, 4, 8} {
		for _, stride := range []int{1, 2, 3, 4} {
			for _, samples := range []int{1, 2, 7, 8, 9, 64, 513} {
				n := samples * stride * size
				row := make([]byte, n)
				for i := range row {
					row[i] = byte(rng.Uint32())
				}
				want := slices.Clone(row)
				floatPredictorRowRef(want, size, stride)
				got := slices.Clone(row)
				floatPredictorRow(got, make([]byte, n), size, stride)
				if !bytes.Equal(got, want) {
					t.Fatalf("size %d, stride %d, %d samples: differs from libtiff's loop", size, stride, samples)
				}
			}
		}
	}
}

// TestFloat32FastPath checks copyFloat32 and float32Validity against
// convert, the general path, for every stride, band and kind of NoData,
// on samples that include NaN, ±0, ±Inf, subnormals, the NoData values
// themselves and their neighbours to either side of ARE_REAL_EQUAL's
// tolerance.
func TestFloat32FastPath(t *testing.T) {
	special := []float32{0, float32(math.Copysign(0, -1)), -9999, 1, float32(math.NaN()),
		float32(math.Inf(1)), float32(math.Inf(-1)), math.SmallestNonzeroFloat32, math.MaxFloat32}
	rng := rand.New(rand.NewPCG(5, 6))
	nds := []noData{
		{},
		prepareNoData(-9999, true, sampleFloat, 32),
		prepareNoData(0, true, sampleFloat, 32),
		prepareNoData(math.NaN(), true, sampleFloat, 32),
		prepareNoData(0.1, true, sampleFloat, 32), // rounds to float32
		prepareNoData(math.MaxFloat32, true, sampleFloat, 32),
	}
	for _, stride := range []int{1, 3} {
		for _, cells := range []int{1, 63, 64, 65, 200} {
			data := make([]byte, 4*stride*cells)
			for i := 0; i < len(data); i += 4 {
				v := special[rng.IntN(len(special))]
				if rng.IntN(2) == 0 {
					v = float32(rng.NormFloat64())
				}
				if rng.IntN(7) == 0 {
					v = float32(0.1)
				}
				if rng.IntN(3) == 0 { // up to 12 ulps from a NoData
					v = []float32{-9999, 0.1, math.MaxFloat32}[rng.IntN(3)]
					v = math.Float32frombits(math.Float32bits(v) + uint32(rng.IntN(25)) - 12) // #nosec G115 -- ulp steps, wrapping as uint32
				}
				binary.LittleEndian.PutUint32(data[i:], math.Float32bits(v))
			}
			for first := range stride {
				for _, nd := range nds {
					want := make([]float32, cells)
					var wantValid []uint64
					if convert(want, data, sampleFloat, 32, stride, first, nd, nil) {
						wantValid = make([]uint64, raster.MaskWords(cells))
						convert(want, data, sampleFloat, 32, stride, first, nd, wantValid)
					}
					got := make([]float32, cells)
					copyFloat32(got, data, stride, first)
					gotValid := float32Validity(got, nd)
					for i := range got {
						g, w := got[i], want[i]
						if math.Float32bits(g) != math.Float32bits(w) && !(g != g && w != w) {
							t.Fatalf("stride %d, band %d, %d cells: cell %d is %v, want %v", stride, first, cells, i, g, w)
						}
					}
					if !slices.Equal(gotValid, wantValid) {
						t.Fatalf("stride %d, band %d, %d cells, NoData %+v: validity %x, want %x",
							stride, first, cells, nd, gotValid, wantValid)
					}
				}
			}
		}
	}
}

// TestZlibHeader checks the header test that replaced zlib.NewReader's:
// every header a zlib writer makes passes, and each rule a reader
// enforces rejects a header that breaks it.
func TestZlibHeader(t *testing.T) {
	for level := -1; level <= 9; level++ {
		var buf bytes.Buffer
		w, err := zlib.NewWriterLevel(&buf, level)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte("strata"))
		_ = w.Close()
		if err := zlibHeader(buf.Bytes()); err != nil {
			t.Errorf("level %d: a zlib writer's header refused: %v", level, err)
		}
	}
	for _, tc := range []struct {
		name string
		h    []byte
	}{
		{"empty", nil},
		{"one byte", []byte{0x78}},
		{"not Deflate", []byte{0x79, 0x9c}},
		{"window too large", []byte{0x88, 0x98}},
		{"check fails", []byte{0x78, 0x9d}},
		{"preset dictionary", []byte{0x78, 0xbb}},
	} {
		if zlibHeader(tc.h) == nil {
			t.Errorf("%s: accepted %x", tc.name, tc.h)
		}
	}
}

// TestFloatPredictorRow32 checks the fused path against the two steps
// it fuses: the predictor undone in place, then the samples copied.
func TestFloatPredictorRow32(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	for _, n := range []int{1, 3, 64, 512, 701} {
		row := make([]byte, 4*n)
		for i := range row {
			row[i] = byte(rng.Uint32())
		}
		ref := slices.Clone(row)
		floatPredictorRow(ref, make([]byte, len(ref)), 4, 1)
		want := make([]float32, n)
		copyFloat32(want, ref, 1, 0)
		got := make([]float32, n)
		kern.PlanesRow(got, slices.Clone(row))
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%d samples: sample %d has bits %08x, want %08x",
					n, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
}

// BenchmarkFloat32Validity times the NoData test on one 512 × 512 block
// with a few NoData cells, the shape of a COG block.
func BenchmarkFloat32Validity(b *testing.B) {
	vals := make([]float32, 512*512)
	for i := range vals {
		vals[i] = float32(i % 1000)
	}
	for i := 0; i < len(vals); i += 997 {
		vals[i] = 65535
	}
	nd := prepareNoData(65535, true, sampleFloat, 32)
	b.SetBytes(int64(4 * len(vals)))
	for b.Loop() {
		float32Validity(vals, nd)
	}
}

// TestDefaultCacheBytes checks that the default cache is DefaultCacheRows
// rows of decoded blocks, within its floor and cap, and that a source
// reports the bound it chose.
func TestDefaultCacheBytes(t *testing.T) {
	row := func(width, bw, bh int) int64 { // one row of blocks, as block.size counts it
		cells := bw * bh
		return int64((width+bw-1)/bw) * int64(4*cells+8*raster.MaskWords(cells)+64)
	}
	for _, tc := range []struct {
		name          string
		width, bw, bh int
		want          int64
	}{
		{"the benchmark's 11264-wide COG", 11264, 512, 512, DefaultCacheRows * row(11264, 512, 512)},
		{"narrow: the floor", 701, 256, 256, DefaultCacheBytes},
		{"strips: the floor", 20000, 20000, 8, DefaultCacheBytes},
		{"very wide: the cap", 200000, 512, 512, MaxDefaultCacheBytes},
		{"absurdly wide: the cap, no overflow", 1 << 40, 16384, 16384, MaxDefaultCacheBytes},
	} {
		got := defaultCacheBytes(&image{width: tc.width, blockW: tc.bw, blockH: tc.bh})
		if got != tc.want {
			t.Errorf("%s: %d bytes, want %d", tc.name, got, tc.want)
		}
	}
	if got := DefaultCacheRows * row(11264, 512, 512); got < 180<<20 || got > 190<<20 {
		t.Errorf("11264-wide: %d MiB, the benchmark's reasoning expects about 182", got>>20)
	}

	f, err := Open(bytes.NewReader(fileSpec{order: binary.LittleEndian, images: []imageSpec{{
		w: 40, h: 30, bands: 1, format: sampleFloat, size: 4, tiled: true, blockW: 16, blockH: 16,
		planar: planarChunky, compression: compressionNone, vals: [][]float64{make([]float64, 40*30)},
	}}}.write()))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ opt, want int64 }{{0, DefaultCacheBytes}, {-1, 0}, {5000, 5000}} {
		src, err := f.Source(SourceOptions{CacheBytes: tc.opt})
		if err != nil {
			t.Fatal(err)
		}
		if got := src.CacheBytes(); got != tc.want {
			t.Errorf("CacheBytes %d: the source reports %d, want %d", tc.opt, got, tc.want)
		}
	}
}

// TestIntFastPath checks intRow, convertInts and exactValidity against
// convert, the general path they replace for 8- and 16-bit integers,
// with and without the horizontal predictor, on every NoData value that
// prepareNoData can make of them, fractions and -0 included.
func TestIntFastPath(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	le := binary.LittleEndian
	for _, st := range []sampleType{{"uint8", sampleUint, 1}, {"int8", sampleInt, 1},
		{"uint16", sampleUint, 2}, {"int16", sampleInt, 2}} {
		bits, signed := 8*st.size, st.format == sampleInt
		for _, n := range []int{1, 63, 64, 65, 300} {
			raw := make([]byte, n*st.size)
			for i := range raw {
				raw[i] = byte(rng.Uint32())
			}
			for i := 0; i < n; i += 7 { // make some cells hold 0 and 12
				v := uint16(0)
				if i%2 == 1 {
					v = 12
				}
				if st.size == 1 {
					raw[i] = byte(v)
				} else {
					le.PutUint16(raw[2*i:], v)
				}
			}
			for _, pred := range []bool{false, true} {
				// The reference: undo the predictor in place, then convert.
				ref := slices.Clone(raw)
				if pred {
					horizontalRow(ref, st.size, 1)
				}
				for _, v := range []float64{-0.5, 0, 12, 12.7, -1, 65535, 1e9} {
					nd := prepareNoData(v, true, st.format, bits)
					want := make([]float32, n)
					var wantValid []uint64
					if convert(want, ref, st.format, bits, 1, 0, nd, nil) {
						wantValid = make([]uint64, raster.MaskWords(n))
						convert(want, ref, st.format, bits, 1, 0, nd, wantValid)
					}
					got := make([]float32, n)
					intRow(got, slices.Clone(raw), bits, signed, pred)
					got2 := make([]float32, n)
					convertInts(got2, ref, bits, signed, 1, 0)
					for i := range got {
						if got[i] != want[i] || got2[i] != want[i] {
							t.Fatalf("%s, %d cells, pred %v: cell %d is %v (intRow) and %v (convertInts), want %v",
								st.name, n, pred, i, got[i], got2[i], want[i])
						}
					}
					if gotValid := exactValidity(got, nd); !slices.Equal(gotValid, wantValid) {
						t.Fatalf("%s, %d cells, pred %v, NoData %v: validity %x, want %x",
							st.name, n, pred, v, gotValid, wantValid)
					}
				}
			}
		}
	}
}

// TestValidatorRows checks that validity built a row at a time is the
// validity float32Validity and exactValidity build from the whole block,
// for widths that do and do not divide into words.
func TestValidatorRows(t *testing.T) {
	rng := rand.New(rand.NewPCG(15, 16))
	for _, w := range []int{1, 37, 64, 100, 128, 512} {
		for _, rows := range []int{1, 3, 8} {
			n := w * rows
			vals := make([]float32, n)
			for i := range vals {
				vals[i] = []float32{0, float32(math.Copysign(0, -1)), -9999, 12, 0.1, float32(math.NaN()),
					float32(math.Inf(1)), float32(i%7) + 0.5}[rng.IntN(8)]
			}
			for _, v := range []float64{-9999, 0, 12, 0.1, math.NaN()} {
				fnd := prepareNoData(v, true, sampleFloat, 32)
				ind := prepareNoData(v, true, sampleInt, 16)
				for _, c := range []struct {
					name string
					test nodataTest
					want []uint64
				}{
					{"float32", float32Test(fnd), float32Validity(vals, fnd)},
					{"int16", intTest(ind), exactValidity(vals, ind)},
				} {
					val := newValidator(c.test, n)
					for r := range rows {
						val.upto(vals, (r+1)*w)
					}
					if got := val.finish(vals); !slices.Equal(got, c.want) {
						t.Fatalf("%s, %d×%d, NoData %v: %x, want %x", c.name, w, rows, v, got, c.want)
					}
				}
			}
		}
	}
}
