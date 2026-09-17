package raster

import (
	"math/rand/v2"
	"testing"
)

func TestMaskWords(t *testing.T) {
	for n, want := range map[int]int{0: 0, 1: 1, 63: 1, 64: 1, 65: 2, 128: 2, 129: 3} {
		if got := MaskWords(n); got != want {
			t.Errorf("MaskWords(%d) = %d, want %d", n, got, want)
		}
	}
}

func TestNewMask(t *testing.T) {
	for _, n := range []int{0, 1, 5, 63, 64, 65, 127, 128, 200} {
		m := NewMask(n)
		if len(m) != MaskWords(n) {
			t.Fatalf("NewMask(%d) has %d words", n, len(m))
		}
		for i := range len(m) * 64 {
			if got, want := MaskGet(m, i), i < n; got != want {
				t.Fatalf("NewMask(%d) bit %d = %v, want %v", n, i, got, want)
			}
		}
	}
	mustPanic(t, "negative mask size", func() { NewMask(-1) })
}

func TestMaskGetSet(t *testing.T) {
	m := make([]uint64, 3)
	for _, i := range []int{0, 1, 63, 64, 100, 191} {
		MaskSet(m, i, true)
		if !MaskGet(m, i) {
			t.Fatalf("bit %d not set", i)
		}
	}
	if m[0] != 1<<63|3 || m[1] != 1|1<<36 || m[2] != 1<<63 {
		t.Fatalf("words = %#x, LSB-first layout broken", m)
	}
	MaskSet(m, 63, false)
	MaskSet(m, 63, false) // idempotent
	MaskSet(m, 1, true)   // idempotent
	if MaskGet(m, 63) || m[0] != 3 {
		t.Fatalf("clearing bit 63: word 0 = %#x", m[0])
	}
}

func TestMaskAnd(t *testing.T) {
	a := NewMask(130)
	b := NewMask(130)
	MaskSet(a, 3, false)
	MaskSet(b, 70, false)
	MaskSet(a, 129, false)
	MaskSet(b, 129, false)

	dst := make([]uint64, len(a))
	MaskAnd(dst, a, b)
	for i := range 130 {
		want := i != 3 && i != 70 && i != 129
		if MaskGet(dst, i) != want {
			t.Fatalf("AND bit %d = %v, want %v", i, !want, want)
		}
	}
	if dst[2]>>2 != 0 {
		t.Fatal("AND set bits past the cell count")
	}

	MaskAnd(a, a, b) // aliasing dst with an input
	for k := range dst {
		if a[k] != dst[k] {
			t.Fatalf("aliased AND word %d = %#x, want %#x", k, a[k], dst[k])
		}
	}

	mustPanic(t, "equal length", func() { MaskAnd(make([]uint64, 2), a, b) })
	mustPanic(t, "equal length", func() { MaskAnd(dst, a[:2], b) })
}

// TestMaskRanges checks the Range helpers bit by bit against MaskGet and
// MaskSet, over aligned and misaligned offsets, lengths around word
// boundaries, and the bits outside the destination range, which must be
// left untouched.
func TestMaskRanges(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 0))
	const words = 8
	random := func() []uint64 {
		m := make([]uint64, words)
		for k := range m {
			m[k] = rng.Uint64()
		}
		return m
	}
	offsets := []int{0, 1, 37, 63, 64, 65, 100, 128, 190}
	lengths := []int{0, 1, 2, 31, 63, 64, 65, 127, 128, 129, 200}
	for _, n := range lengths {
		for _, dOff := range offsets {
			for _, aOff := range offsets {
				bOff := offsets[rng.IntN(len(offsets))]
				a, b, dst := random(), random(), random()
				orig := append([]uint64(nil), dst...)
				want := func(i int, got func(int) bool) {
					t.Helper()
					for bit := range words * 64 {
						exp := MaskGet(orig, bit)
						if bit >= dOff && bit < dOff+n {
							exp = got(bit - dOff)
						}
						if MaskGet(dst, bit) != exp {
							t.Fatalf("case %d n=%d dst@%d a@%d b@%d: bit %d = %v, want %v",
								i, n, dOff, aOff, bOff, bit, !exp, exp)
						}
					}
				}

				MaskAndRange(dst, dOff, a, aOff, b, bOff, n)
				want(0, func(i int) bool { return MaskGet(a, aOff+i) && MaskGet(b, bOff+i) })

				copy(dst, orig)
				MaskCopyRange(dst, dOff, a, aOff, n)
				want(1, func(i int) bool { return MaskGet(a, aOff+i) })

				for _, v := range []bool{false, true} {
					copy(dst, orig)
					MaskFillRange(dst, dOff, n, v)
					want(2, func(int) bool { return v })
				}
			}
		}
	}
}

func TestMaskRangesInPlace(t *testing.T) {
	a := []uint64{0xf0f0_1234_5678_9abc, 0x0123_4567_89ab_cdef, 0xffff_0000_ffff_0000}
	b := []uint64{0xffff_ffff_0000_0000, 0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555}
	want := append([]uint64(nil), a...)
	MaskAndRange(want, 3, a, 3, b, 3, 180)
	got := append([]uint64(nil), a...)
	MaskAndRange(got, 3, got, 3, b, 3, 180)
	for k := range got {
		if got[k] != want[k] {
			t.Fatalf("in-place AND word %d = %#x, want %#x", k, got[k], want[k])
		}
	}
	MaskCopyRange(got, 7, got, 7, 150) // self-copy is a no-op
	for k := range got {
		if got[k] != want[k] {
			t.Fatalf("self copy changed word %d", k)
		}
	}
}

func TestMaskRangePanics(t *testing.T) {
	m := make([]uint64, 2)
	mustPanic(t, "outside 128-bit mask dst", func() { MaskAndRange(m, 100, m, 0, m, 0, 29) })
	mustPanic(t, "outside 128-bit mask a", func() { MaskAndRange(m, 0, m, -1, m, 0, 1) })
	mustPanic(t, "outside 128-bit mask b", func() { MaskAndRange(m, 0, m, 0, m, 127, 2) })
	mustPanic(t, "-1 bits at offset 0", func() { MaskFillRange(m, 0, -1, true) })
	mustPanic(t, "outside 64-bit mask src", func() { MaskCopyRange(m, 0, m[:1], 1, 64) })
	mustPanic(t, "outside 128-bit mask m", func() { MaskFillRange(m, 128, 1, true) })
	MaskFillRange(m, 128, 0, true) // empty range at the end is fine
}
