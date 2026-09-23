package cog

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

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
		floatPredictorRow32(got, slices.Clone(row))
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
