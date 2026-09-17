package engine_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"testing"

	"strata/engine"
	"strata/internal/fuzzdata"
	"strata/raster"
)

// fuzzWindow returns a w×h raster decoded from d: compact, strided, or a
// window at an offset of a larger root with a mask offset, and the root.
// Values and mask bits are arbitrary; masked selects a mask.
func fuzzWindow(d *fuzzdata.Reader, w, h int, masked bool) (win, root raster.Float32Raster) {
	rootW, rootH, x, y := w, h, 0, 0
	layout := d.IntN(3)
	if layout == 2 {
		x, y = d.IntN(4), d.IntN(3)
		rootW, rootH = w+x+d.IntN(3), h+y+d.IntN(3)
	}
	stride := rootW
	if layout > 0 {
		stride += d.IntN(70)
	}
	n := (rootH-1)*stride + rootW
	root = raster.NewFloat32Stride(rootW, rootH, stride, make([]float32, n))
	for i := range root.Data {
		root.Data[i] = d.Float32()
	}
	if masked {
		off := d.IntN(130)
		root.Valid = make([]uint64, raster.MaskWords(off+n)+d.IntN(2))
		root.ValidOffset = off
		for k := range root.Valid {
			root.Valid[k] = d.Uint64()
		}
	}
	return root.Window(x, y, w, h), root
}

// panicked runs f and reports whether it panicked, failing the test on a
// panic without an "engine: " message.
func panicked(t *testing.T, id string, f func()) (p bool) {
	t.Helper()
	defer func() {
		if v := recover(); v != nil {
			p = true
			if s, ok := v.(string); !ok || !strings.HasPrefix(s, "engine: ") {
				t.Fatalf("%s: panic %v (%T), want an \"engine: \" message", id, v, v)
			}
		}
	}()
	f()
	return false
}

// FuzzRawRoundTrip writes a raster to a raw file through RawSink in
// arbitrary tiles, then reads arbitrary windows back through RawSource
// into rasters of arbitrary layout, with and without a fill value (NaN,
// -0 or any other), with IO calls grouped at arbitrary sizes. Every read
// cell must hold the bits that were written (any NaN payload included),
// or the fill value's validity: invalid cells come back invalid, and so
// do valid cells holding the fill value. Nothing outside a read window
// may change. Reads past the end of a truncated file must fail with
// io.ErrUnexpectedEOF and only then; a done context fails without
// reading; regions outside the raster, masks the file cannot store and
// fill values without a mask to read them into panic.
func FuzzRawRoundTrip(f *testing.F) {
	f.Add([]byte{10, 3, 1, 0, 0, 5, 2, 3})
	f.Add([]byte{65, 4, 0, 1, 8, 64, 1, 2, 0})
	f.Add([]byte{1, 1, 1, 1, 0, 0, 0})
	f.Add([]byte{17, 6, 1, 7, 3, 4, 5, 1, 2, 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		ctx := context.Background()
		W, H := d.Range(1, 90), d.Range(1, 8)
		opts := engine.RawOptions{HasFill: d.Bool(), Fill: d.Float32()}
		defer engine.SetRawCallBytes(d.Range(0, 5*4*W))()
		id := fmt.Sprintf("%d×%d fill %v %v", W, H, opts.HasFill, opts.Fill)

		src, _ := fuzzWindow(d, W, H, opts.HasFill && d.Bool())
		type cell struct {
			v     float32
			valid bool
		}
		orig := make([]cell, W*H)
		for y := range H {
			for x := range W {
				orig[y*W+x] = cell{src.Data[src.Index(x, y)], src.IsValid(x, y)}
			}
		}

		// Write in tiles, out of order.
		file := &memFile{}
		sink := engine.NewRawSink(file, W, H, opts)
		tw, th := d.Range(1, W), d.Range(1, H)
		var tiles [][2]int
		for y := 0; y < H; y += th {
			for x := 0; x < W; x += tw {
				tiles = append(tiles, [2]int{x, y})
			}
		}
		for i := range tiles {
			j := i + d.IntN(len(tiles)-i)
			tiles[i], tiles[j] = tiles[j], tiles[i]
		}
		for _, tl := range tiles {
			x, y := tl[0], tl[1]
			win := src.Window(x, y, min(tw, W-x), min(th, H-y))
			if err := sink.WriteWindow(ctx, win, x, y); err != nil {
				t.Fatalf("%s: write tile at (%d, %d): %v", id, x, y, err)
			}
		}
		if len(file.b) != 4*W*H {
			t.Fatalf("%s: file has %d bytes, want %d", id, len(file.b), 4*W*H)
		}
		// A mask on src without a fill value to store it panics.
		if !opts.HasFill {
			masked, _ := fuzzWindow(d, 1, 1, true)
			if !panicked(t, id, func() { _ = sink.WriteWindow(ctx, masked, 0, 0) }) {
				t.Fatalf("%s: writing a masked window without a fill value did not panic", id)
			}
		}

		// Truncate the file sometimes.
		fileLen := 4 * W * H
		if d.IntN(4) == 0 {
			fileLen = d.IntN(fileLen)
			file.b = file.b[:fileLen]
		}
		source := engine.NewRawSource(file, W, H, opts)

		for range d.Range(1, 4) {
			w, h := d.Range(1, W), d.Range(1, H)
			x, y := d.Range(0, W-w), d.Range(0, H-h)
			outside := d.IntN(10) == 0
			if outside {
				x, y = d.Range(-2, W+1), d.Range(-2, H+1)
			}
			dstMasked := opts.HasFill || d.Bool()
			missingMask := opts.HasFill && d.IntN(10) == 0
			if missingMask {
				dstMasked = false
			}
			dst, root := fuzzWindow(d, w, h, dstMasked)
			before := raster.Float32Raster{Data: append([]float32(nil), root.Data...), Valid: append([]uint64(nil), root.Valid...)}
			rid := fmt.Sprintf("%s read %d×%d at (%d, %d) masked %v", id, w, h, x, y, dstMasked)

			inside := x >= 0 && y >= 0 && x <= W-w && y <= H-h
			var err error
			readCtx := ctx
			cancelled := d.IntN(8) == 0
			if cancelled {
				c, cancel := context.WithCancel(ctx)
				cancel()
				readCtx = c
			}
			p := panicked(t, rid, func() { err = source.ReadWindow(readCtx, dst, x, y) })
			if p != (!inside || missingMask) {
				t.Fatalf("%s: panicked = %v, want %v (inside %v, missing mask %v)", rid, p, !inside || missingMask, inside, missingMask)
			}
			if p || cancelled {
				if !p && !errors.Is(err, context.Canceled) {
					t.Fatalf("%s: cancelled read: err = %v", rid, err)
				}
				requireRootUnchanged(t, rid, root, before, nil)
				continue
			}
			short := 4*((y+h-1)*W+x+w) > fileLen
			if short != (err != nil) {
				t.Fatalf("%s: err = %v, file has %d bytes, want an error %v", rid, err, fileLen, short)
			}
			if err != nil {
				if !errors.Is(err, io.ErrUnexpectedEOF) {
					t.Fatalf("%s: err = %v, want io.ErrUnexpectedEOF", rid, err)
				}
				continue // the window's cells are unspecified
			}
			for yy := range h {
				for xx := range w {
					c := orig[(y+yy)*W+x+xx]
					stored := c.v
					if !c.valid {
						stored = opts.Fill
					}
					wantValid := !opts.HasFill || !(stored == opts.Fill || opts.Fill != opts.Fill && stored != stored)
					if got := dst.IsValid(xx, yy); got != wantValid {
						t.Fatalf("%s: cell (%d, %d) valid = %v, want %v (stored %v)", rid, xx, yy, got, wantValid, stored)
					}
					if got := dst.Data[dst.Index(xx, yy)]; wantValid && math.Float32bits(got) != math.Float32bits(stored) {
						t.Fatalf("%s: cell (%d, %d) = %#08x, want %#08x", rid, xx, yy, math.Float32bits(got), math.Float32bits(stored))
					}
				}
			}
			requireRootUnchanged(t, rid, root, before, &dst)
		}
	})
}

// requireRootUnchanged fails if a cell or mask bit of root outside win (or
// anywhere, for a nil win) differs from before.
func requireRootUnchanged(t *testing.T, id string, root, before raster.Float32Raster, win *raster.Float32Raster) {
	t.Helper()
	first := 0
	if win != nil {
		first = indexIn(root, *win)
	}
	in := func(i int) bool {
		if win == nil {
			return false
		}
		j := i - first
		return j >= 0 && j/win.Stride < win.Height && j%win.Stride < win.Width
	}
	for i := range root.Data {
		if !in(i) && math.Float32bits(root.Data[i]) != math.Float32bits(before.Data[i]) {
			t.Fatalf("%s: root cell %d outside the window changed", id, i)
		}
	}
	for bit := range len(before.Valid) * 64 {
		i := bit - root.ValidOffset
		if (i < 0 || i >= len(root.Data) || !in(i)) && raster.MaskGet(root.Valid, bit) != raster.MaskGet(before.Valid, bit) {
			t.Fatalf("%s: mask bit %d outside the window changed", id, bit)
		}
	}
}

// indexIn returns the index in root.Data of win's first cell.
func indexIn(root, win raster.Float32Raster) int {
	for i := range root.Data {
		if &root.Data[i] == &win.Data[0] {
			return i
		}
	}
	panic("window is not in root")
}
