package raster

import "testing"

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
