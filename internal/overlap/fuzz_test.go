package overlap

import (
	"testing"
	"unsafe"

	"strata/internal/fuzzdata"
	"strata/raster"
)

// FuzzRelations builds two rasters of equal size over one buffer and one
// mask array, at any offsets and strides, including float32 views that
// are misaligned by a few bytes and masks that are different slices of
// one array, and checks Data, Bits, DataSpans, BitSpans and Words against
// brute force over every cell:
//
//   - Same exactly when the two are the same cells in the same order;
//   - never Disjoint when a cell (or bit) is shared, and exactly Disjoint
//     otherwise for equal strides and aligned views;
//   - Partial without a shared cell only where the spans meet;
//   - the span functions exact, Words sound, and everything symmetric.
func FuzzRelations(f *testing.F) {
	f.Add([]byte{3, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0})
	f.Add([]byte{4, 3, 2, 10, 3, 1, 2, 17, 0, 0, 1, 1, 64, 0})
	f.Add([]byte{1, 1, 0, 5, 7, 1, 0, 5, 3, 1, 1, 2, 0, 5})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		w, h := d.Range(1, 12), d.Range(1, 12)
		const bufCells = 700
		buf := make([]float32, bufCells)
		masks := [2][]uint64{make([]uint64, 24), make([]uint64, 24)}

		type view struct {
			r     raster.Float32Raster
			first int // byte address of the first cell, relative to buf
			mask  int // index into masks, or -1
			bit0  int // bit address of the first validity bit in its mask array
		}
		newView := func() view {
			stride := w + d.IntN(24)
			span := (h-1)*stride + w
			start := d.IntN(bufCells - span)
			shift := 0
			if d.IntN(4) == 0 {
				shift = d.Range(1, 3)
			}
			p := (*float32)(unsafe.Add(unsafe.Pointer(&buf[start]), shift))
			v := view{
				r:     raster.Float32Raster{Data: unsafe.Slice(p, span), Width: w, Height: h, Stride: stride},
				first: 4*start + shift,
				mask:  -1,
			}
			if d.IntN(4) != 0 {
				v.mask = d.IntN(4) / 3 // mostly the first array, so masks meet
				k := d.IntN(4)
				v.r.Valid = masks[v.mask][k:]
				v.r.ValidOffset = d.IntN(len(v.r.Valid)*64 - span + 1)
				v.bit0 = 64*k + v.r.ValidOffset
			}
			return v
		}
		a, b := newView(), newView()

		// Brute force over cells: shared bytes of Data, shared validity
		// bits and shared validity words.
		sharedData, sharedBit, sharedWord := false, false, false
		sameMask := a.mask >= 0 && a.mask == b.mask
		for ay := range h {
			for ax := range w {
				ai := ay*a.r.Stride + ax
				for by := range h {
					for bx := range w {
						bi := by*b.r.Stride + bx
						if pa, pb := a.first+4*ai, b.first+4*bi; pa < pb+4 && pb < pa+4 {
							sharedData = true
						}
						if sameMask {
							ba, bb := a.bit0+ai, b.bit0+bi
							sharedBit = sharedBit || ba == bb
							sharedWord = sharedWord || ba>>6 == bb>>6
						}
					}
				}
			}
		}
		// The same cells in the same order: the same first cell and, over
		// more than one row, the same stride.
		identical := func(p0, q0 int) bool {
			return p0 == q0 && (a.r.Stride == b.r.Stride || h == 1)
		}
		spanA, spanB := span(a.r), span(b.r)

		check := func(kind string, rel, relBA Relation, same, shared, exact, spansMeet bool) {
			t.Helper()
			if rel != relBA {
				t.Fatalf("%s is not symmetric: %v and %v", kind, rel, relBA)
			}
			switch {
			case same && rel != Same:
				t.Fatalf("%s = %v for identical cells", kind, rel)
			case !same && rel == Same:
				t.Fatalf("%s = Same for different cells (w %d h %d strides %d %d)", kind, w, h, a.r.Stride, b.r.Stride)
			case shared && rel == Disjoint:
				t.Fatalf("%s = Disjoint but a cell is shared (w %d h %d strides %d %d)", kind, w, h, a.r.Stride, b.r.Stride)
			case !shared && rel == Partial && (exact || !spansMeet):
				t.Fatalf("%s = Partial but no cell is shared (w %d h %d strides %d %d, spans meet %v)",
					kind, w, h, a.r.Stride, b.r.Stride, spansMeet)
			}
		}

		aligned := (a.first-b.first)%4 == 0
		dataSpans := a.first < b.first+4*spanB && b.first < a.first+4*spanA
		check("Data", Data(a.r, b.r), Data(b.r, a.r),
			identical(a.first, b.first), sharedData,
			aligned && a.r.Stride == b.r.Stride, dataSpans)
		if got := DataSpans(a.r, b.r); got != dataSpans {
			t.Fatalf("DataSpans = %v, want %v", got, dataSpans)
		}

		bitSpans := sameMask && a.bit0 < b.bit0+spanB && b.bit0 < a.bit0+spanA
		check("Bits", Bits(a.r, b.r), Bits(b.r, a.r),
			sameMask && identical(a.bit0, b.bit0), sharedBit,
			a.r.Stride == b.r.Stride, bitSpans)
		if got := BitSpans(a.r, b.r); got != bitSpans {
			t.Fatalf("BitSpans = %v, want %v", got, bitSpans)
		}

		words := Words(a.r, b.r)
		if words != Words(b.r, a.r) {
			t.Fatalf("Words is not symmetric")
		}
		if sharedWord && !words {
			t.Fatalf("Words = false but a validity word is shared (bits at %d and %d)", a.bit0, b.bit0)
		}
		wordSpans := sameMask && a.bit0>>6 <= (b.bit0+spanB-1)>>6 && b.bit0>>6 <= (a.bit0+spanA-1)>>6
		if words && !wordSpans {
			t.Fatalf("Words = true but the word spans do not meet (bits at %d and %d)", a.bit0, b.bit0)
		}
	})
}
