package nodata

import (
	"fmt"
	"math"
	"testing"
)

// testShapes straddle the 8-lane AVX2 width and the 64-bit mask word,
// including odd widths whose rows do not start on a word boundary.
var testShapes = [][2]int{{3, 3}, {9, 4}, {10, 7}, {64, 5}, {65, 9}, {67, 45}, {130, 33}, {1027, 13}}

const testCellSize = 10

func forEachFixture(t *testing.T, fn func(t *testing.T, f *Fixture)) {
	for _, s := range testShapes {
		for _, p := range Patterns {
			f := NewFixture(s[0], s[1], p)
			t.Run(fmt.Sprintf("%dx%d/%s", s[0], s[1], p.Name), func(t *testing.T) { fn(t, f) })
		}
	}
}

// TestBackend records whether the AVX2 variants were exercised; without
// GOEXPERIMENT=simd on amd64 they fall back to scalar.
func TestBackend(t *testing.T) {
	t.Logf("HaveAVX2 = %v", HaveAVX2)
}

func sameBits(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) }

func checkSentinel(t *testing.T, name string, got, ref []float32, invalid []bool) {
	t.Helper()
	for i := range got {
		if invalid[i] != (got[i] == Sentinel) {
			t.Fatalf("%s: cell %d invalid=%v got %v", name, i, invalid[i], got[i])
		}
		if !invalid[i] && !sameBits(got[i], ref[i]) {
			t.Fatalf("%s: cell %d got %v want %v", name, i, got[i], ref[i])
		}
	}
}

func checkNaN(t *testing.T, name string, got, ref []float32, invalid []bool) {
	t.Helper()
	for i := range got {
		if invalid[i] != isNaN32(got[i]) {
			t.Fatalf("%s: cell %d invalid=%v got %v", name, i, invalid[i], got[i])
		}
		if !invalid[i] && !sameBits(got[i], ref[i]) {
			t.Fatalf("%s: cell %d got %v want %v", name, i, got[i], ref[i])
		}
	}
}

func checkMask(t *testing.T, name string, got, ref []float32, valid []uint64, invalid []bool) {
	t.Helper()
	for i := range got {
		if invalid[i] == MaskBit(valid, i) {
			t.Fatalf("%s: cell %d invalid=%v but mask bit=%v", name, i, invalid[i], MaskBit(valid, i))
		}
		if !invalid[i] && !sameBits(got[i], ref[i]) {
			t.Fatalf("%s: cell %d got %v want %v", name, i, got[i], ref[i])
		}
	}
	if tail := len(got) & 63; tail != 0 && valid[len(valid)-1]>>uint(tail) != 0 {
		t.Fatalf("%s: mask bits set past the last cell", name)
	}
}

func checkFilled(t *testing.T, name string, got []float32, invalid []bool, fill float32) {
	t.Helper()
	for i := range got {
		if invalid[i] && !sameBits(got[i], fill) {
			t.Fatalf("%s: invalid cell %d holds %v, want fill %v", name, i, got[i], fill)
		}
	}
}

func TestAddVariantsAgree(t *testing.T) {
	forEachFixture(t, func(t *testing.T, f *Fixture) {
		n := f.W * f.H
		invalid := make([]bool, n)
		ref := make([]float32, n)
		for i := range ref {
			invalid[i] = f.InvalidA[i] || f.InvalidB[i]
			ref[i] = f.SentinelA[i] + f.SentinelB[i]
		}
		dst := make([]float32, n)
		mask := make([]uint64, MaskWords(n))

		AddSentinelBranchy(dst, f.SentinelA, f.SentinelB, Sentinel)
		checkSentinel(t, "sentinel/branchy", dst, ref, invalid)
		AddSentinelSelect(dst, f.SentinelA, f.SentinelB, Sentinel)
		checkSentinel(t, "sentinel/select", dst, ref, invalid)
		AddSentinelVecFixup(dst, f.SentinelA, f.SentinelB, Sentinel)
		checkSentinel(t, "sentinel/vec-fixup", dst, ref, invalid)
		if HaveAVX2 {
			AddSentinelAVX2(dst, f.SentinelA, f.SentinelB, Sentinel)
			checkSentinel(t, "sentinel/avx2-blend", dst, ref, invalid)
		}

		AddNaNScalar(dst, f.NaNA, f.NaNB)
		checkNaN(t, "nan/scalar", dst, ref, invalid)
		AddNaNVec(dst, f.NaNA, f.NaNB)
		checkNaN(t, "nan/vec", dst, ref, invalid)

		AddMaskScalar(dst, f.SentinelA, f.SentinelB, mask, f.ValidA, f.ValidB)
		checkMask(t, "mask/scalar", dst, ref, mask, invalid)
		AddMaskVec(dst, f.SentinelA, f.SentinelB, mask, f.ValidA, f.ValidB)
		checkMask(t, "mask/vec", dst, ref, mask, invalid)
		AddMaskVecFill(dst, f.SentinelA, f.SentinelB, mask, f.ValidA, f.ValidB, NaN32)
		checkMask(t, "mask/vec-fill", dst, ref, mask, invalid)
		checkFilled(t, "mask/vec-fill", dst, invalid, NaN32)
	})
}

// slopeReference computes validity and values cell by cell from the
// ground-truth NoData flags, independent of any representation.
func slopeReference(f *Fixture) (ref []float32, invalid []bool) {
	w, h := f.W, f.H
	ref = make([]float32, w*h)
	invalid = make([]bool, w*h)
	invx, invy := hornScales(testCellSize)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			if x == 0 || y == 0 || x == w-1 || y == h-1 {
				invalid[i] = true
				continue
			}
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if f.InvalidA[i+dy*w+dx] {
						invalid[i] = true
					}
				}
			}
			z := func(dx, dy int) float32 { return f.SentinelA[i+dy*w+dx] }
			ref[i] = horn(z(-1, -1), z(0, -1), z(1, -1), z(-1, 0), z(1, 0), z(-1, 1), z(0, 1), z(1, 1), invx, invy)
		}
	}
	return ref, invalid
}

func TestSlopeVariantsAgree(t *testing.T) {
	forEachFixture(t, func(t *testing.T, f *Fixture) {
		w, h := f.W, f.H
		n := w * h
		ref, invalid := slopeReference(f)
		dst := make([]float32, n)
		mask := make([]uint64, MaskWords(n))
		scratch := make([]uint64, SlopeMaskScratchWords(w))
		poison := func() {
			for k := range mask {
				mask[k] = ^uint64(0)
			}
		}

		SlopeSentinelBranchy(dst, f.SentinelA, w, h, testCellSize, Sentinel)
		checkSentinel(t, "sentinel/branchy", dst, ref, invalid)
		SlopeSentinelSelect(dst, f.SentinelA, w, h, testCellSize, Sentinel)
		checkSentinel(t, "sentinel/select", dst, ref, invalid)
		SlopeNaNScalar(dst, f.NaNA, w, h, testCellSize)
		checkNaN(t, "nan/scalar", dst, ref, invalid)
		poison()
		SlopeMaskBranchy(dst, f.SentinelA, mask, f.ValidA, w, h, testCellSize)
		checkMask(t, "mask/branchy", dst, ref, mask, invalid)
		poison()
		SlopeMaskScalar(dst, f.SentinelA, mask, f.ValidA, w, h, testCellSize, scratch)
		checkMask(t, "mask/scalar", dst, ref, mask, invalid)

		if !HaveAVX2 {
			return
		}
		SlopeSentinelAVX2(dst, f.SentinelA, w, h, testCellSize, Sentinel)
		checkSentinel(t, "sentinel/avx2-blend", dst, ref, invalid)
		SlopeNaNAVX2(dst, f.NaNA, w, h, testCellSize)
		checkNaN(t, "nan/avx2", dst, ref, invalid)
		poison()
		SlopeMaskAVX2(dst, f.SentinelA, mask, f.ValidA, w, h, testCellSize, scratch)
		checkMask(t, "mask/avx2", dst, ref, mask, invalid)
		poison()
		SlopeMaskAVX2Fill(dst, f.SentinelA, mask, f.ValidA, w, h, testCellSize, scratch, Sentinel)
		checkMask(t, "mask/avx2-fill", dst, ref, mask, invalid)
		checkFilled(t, "mask/avx2-fill", dst, invalid, Sentinel)
	})
}

func TestBitHelpers(t *testing.T) {
	src := []uint64{0x0123456789abcdef, 0xfedcba9876543210, 0x0f0f0f0f0f0f0f0f}
	for off := 0; off < 130; off += 7 {
		for n := 1; off+n <= 192; n += 11 {
			got := make([]uint64, MaskWords(n))
			extractBits(got, src, off, n)
			for i := 0; i < n; i++ {
				if MaskBit(got, i) != MaskBit(src, off+i) {
					t.Fatalf("extract off=%d n=%d bit %d", off, n, i)
				}
			}
			if r := n & 63; r != 0 && got[len(got)-1]>>uint(r) != 0 {
				t.Fatalf("extract off=%d n=%d left high bits set", off, n)
			}

			orig := []uint64{^uint64(0), 0, ^uint64(0)}
			dst := append([]uint64(nil), orig...)
			depositBits(dst, off, got, n)
			for i := 0; i < 192; i++ {
				want := MaskBit(orig, i)
				if i >= off && i < off+n {
					want = MaskBit(src, i)
				}
				if MaskBit(dst, i) != want {
					t.Fatalf("deposit off=%d n=%d bit %d", off, n, i)
				}
			}

			copy(dst, orig)
			clearBits(dst, off, n)
			for i := 0; i < 192; i++ {
				want := MaskBit(orig, i) && (i < off || i >= off+n)
				if MaskBit(dst, i) != want {
					t.Fatalf("clear off=%d n=%d bit %d", off, n, i)
				}
			}
		}
	}
}

func TestIngestConversions(t *testing.T) {
	f := NewFixture(131, 17, Patterns[2])
	n := f.W * f.H
	m := make([]uint64, MaskWords(n))
	MaskFromSentinel(m, f.SentinelA, Sentinel)
	for k := range m {
		if m[k] != f.ValidA[k] {
			t.Fatalf("MaskFromSentinel word %d = %#x want %#x", k, m[k], f.ValidA[k])
		}
	}
	d := make([]float32, n)
	SentinelToNaN(d, f.SentinelA, Sentinel)
	for i := range d {
		if isNaN32(d[i]) != f.InvalidA[i] {
			t.Fatalf("SentinelToNaN cell %d", i)
		}
	}
}
