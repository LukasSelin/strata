package exec_test

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"

	"strata/engine"
	"strata/internal/exec"
	"strata/internal/fuzzdata"
	"strata/raster"
)

// FuzzPlan checks the band plan for any raster size, tile size and band
// size target: the bands cover every cell exactly once, each lies within
// one tile and spans its full width, and they come tile by tile in
// row-major order, top to bottom within a tile.
func FuzzPlan(f *testing.F) {
	f.Add(uint8(10), uint8(7), uint8(3), uint8(2), uint16(1))
	f.Add(uint8(200), uint8(1), uint8(0), uint8(0), uint16(0))
	f.Add(uint8(5), uint8(255), uint8(6), uint8(0), uint16(65535))
	f.Fuzz(func(t *testing.T, w8, h8, tw8, th8 uint8, cells uint16) {
		w, h, tw, th := int(w8)%200+1, int(h8)%200+1, int(tw8), int(th8)
		if cells > 0 {
			defer exec.SetBandCells(int(cells))()
		}
		tileW, tileH := w, h
		if tw > 0 {
			tileW = min(tw, w)
		}
		if th > 0 {
			tileH = min(th, h)
		}
		id := fmt.Sprintf("%d×%d tiles %d×%d cells %d", w, h, tw, th, cells)
		covered := make([]bool, w*h)
		prevTile, prevY1 := -1, 0
		for _, b := range exec.Bands(w, h, tw, th) {
			x0, y0, x1, y1 := b[0], b[1], b[2], b[3]
			if x0 < 0 || y0 < 0 || x1 > w || y1 > h || x0 >= x1 || y0 >= y1 {
				t.Fatalf("%s: band %v is empty or outside the raster", id, b)
			}
			tx, ty := x0/tileW, y0/tileH
			if x0 != tx*tileW || x1 != min((tx+1)*tileW, w) || y1 > min((ty+1)*tileH, h) {
				t.Fatalf("%s: band %v does not span the width of one tile", id, b)
			}
			tile := ty*((w+tileW-1)/tileW) + tx
			switch {
			case tile == prevTile && y0 != prevY1:
				t.Fatalf("%s: band %v does not follow the previous band's rows (ended at %d)", id, b, prevY1)
			case tile != prevTile && (tile != prevTile+1 || y0 != ty*tileH):
				t.Fatalf("%s: band %v starts tile %d after tile %d", id, b, tile, prevTile)
			}
			prevTile, prevY1 = tile, y1
			for y := y0; y < y1; y++ {
				for x := x0; x < x1; x++ {
					if covered[y*w+x] {
						t.Fatalf("%s: cell (%d, %d) is in two bands", id, x, y)
					}
					covered[y*w+x] = true
				}
			}
		}
		for i, c := range covered {
			if !c {
				t.Fatalf("%s: cell (%d, %d) is in no band", id, i%w, i/w)
			}
		}
	})
}

// fuzzOperand decodes a w×h operand: compact, or a window with Stride >
// Width and a mask offset into a larger root, with arbitrary values and
// mask bits.
func fuzzOperand(d *fuzzdata.Reader, w, h int, masked bool) operand {
	rootW, rootH, stride, x, y, off := w, h, w, 0, 0, 0
	if d.Bool() {
		x, y = d.IntN(4), d.IntN(3)
		rootW, rootH = w+x+d.IntN(3), h+y+d.IntN(3)
		stride = rootW + d.IntN(70)
		off = d.IntN(100)
	}
	n := (rootH-1)*stride + rootW
	root := raster.NewFloat32Stride(rootW, rootH, stride, make([]float32, n))
	for i := range root.Data {
		root.Data[i] = d.Float32()
	}
	if masked {
		root.Valid = make([]uint64, raster.MaskWords(off+n)+1)
		root.ValidOffset = off
		for k := range root.Valid {
			root.Valid[k] = d.Uint64() | d.Uint64()
		}
	}
	return operand{r: root.Window(x, y, w, h), root: root, x: x, y: y}
}

// FuzzProcess runs box kernels of radius 0 to 3 with zero to three inputs
// and one to three outputs through ProcessN and ProcessChunked, with any
// tiles, workers and band size, over operands of arbitrary layout, values
// and masks; outputs that are disjoint windows sharing one root and mask
// (so workers share mask words); and radius 0 in place. Both must write
// exactly what the naive reference writes into every output's root, and
// an input with a mask and an output without one must panic before
// anything is written.
func FuzzProcess(f *testing.F) {
	f.Add([]byte{1, 1, 1, 9, 5, 0})
	f.Add([]byte{0, 2, 1, 66, 3, 1, 1, 1, 0})
	f.Add([]byte{3, 0, 3, 7, 9, 0, 1, 2, 3})
	f.Add([]byte{2, 3, 2, 64, 8, 1, 0, 1, 1, 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		d := fuzzdata.New(data)
		r, nin, nout := d.Range(0, 3), d.Range(0, 3), d.Range(1, 3)
		w, h := d.Range(1, 70), d.Range(1, 9)
		customEdge := d.Bool()
		edge := d.Float32()

		ins := make([]operand, nin)
		inMasked := false
		for i := range ins {
			m := d.Bool()
			inMasked = inMasked || m
			ins[i] = fuzzOperand(d, w, h, m)
		}
		outMasked := inMasked || d.Bool()
		missingMask := inMasked && d.IntN(10) == 0
		if missingMask {
			outMasked = false
		}
		outs := make([]operand, nout)
		sharedRoot := nout > 1 && d.Bool()
		// sinksShareWords: shared-root outputs whose mask word spans meet,
		// which ProcessChunked rejects (two memory sinks would race on a
		// word) and ProcessN handles under its mask lock.
		sinksShareWords := false
		if sharedRoot {
			// Disjoint windows of one root, stacked, sharing one mask.
			stride := w + d.IntN(70)
			rootH := nout*(h+1) - 1
			n := (rootH-1)*stride + w
			root := raster.NewFloat32Stride(w, rootH, stride, make([]float32, n))
			if outMasked {
				root.Valid = make([]uint64, raster.MaskWords(n))
				for k := range root.Valid {
					root.Valid[k] = d.Uint64()
				}
			}
			for i := range outs {
				outs[i] = operand{r: root.Window(0, i*(h+1), w, h), root: root, y: i * (h + 1)}
				if i > 0 && outMasked {
					prevLast := (i-1)*(h+1)*stride + (h-1)*stride + w - 1
					sinksShareWords = sinksShareWords || prevLast>>6 >= i*(h+1)*stride>>6
				}
			}
		} else {
			for i := range outs {
				outs[i] = fuzzOperand(d, w, h, outMasked)
			}
		}
		inPlace := r == 0 && nin > 0 && nout == 1 && !missingMask && d.IntN(4) == 0
		if inPlace && (ins[0].r.Valid != nil) != outMasked {
			inPlace = false
		}

		opts := engine.Options{TileWidth: d.Range(0, w+2), TileHeight: d.Range(0, h+2), Workers: d.Range(0, 4)}
		bandCells := d.Range(0, 3*w)
		if bandCells > 0 {
			defer exec.SetBandCells(bandCells)()
		}
		box := boxKernel{r: r, inputs: nin, outputs: nout}
		var k exec.Kernel = box
		if !customEdge {
			edge = float32(math.NaN())
		} else {
			k = edgeBox{box, edge}
		}
		id := fmt.Sprintf("r %d inputs %d outputs %d %d×%d masks in %v out %v shared %v in place %v opts %+v band cells %d edge %v",
			r, nin, nout, w, h, inMasked, outMasked, sharedRoot, inPlace, opts, bandCells, edge)

		// clones returns fresh copies of the operands, keeping shared
		// roots shared and an in-place output the first input.
		clones := func() (cin, cout []operand) {
			for _, in := range ins {
				cin = append(cin, in.clone())
			}
			if sharedRoot {
				c := outs[0].clone()
				for _, o := range outs {
					cout = append(cout, operand{r: c.root.Window(o.x, o.y, w, h), root: c.root, x: o.x, y: o.y})
				}
			} else if inPlace {
				cout = []operand{cin[0]}
			} else {
				for _, o := range outs {
					cout = append(cout, o.clone())
				}
			}
			return cin, cout
		}
		views := func(ops []operand) []raster.Float32Raster {
			v := make([]raster.Float32Raster, len(ops))
			for i, o := range ops {
				v[i] = o.r
			}
			return v
		}

		// The reference: naive over clones, reading a snapshot of the
		// inputs so in-place runs compare correctly.
		wantIn, wantOut := clones()
		snapshot := make([]raster.Float32Raster, nin)
		for i, in := range wantIn {
			snapshot[i] = in.clone().r
		}
		for _, o := range wantOut {
			naiveBox(o.r, snapshot, r, edge)
		}

		ctx := context.Background()
		for _, chunked := range []bool{false, true} {
			if chunked && inPlace {
				continue
			}
			gotIn, gotOut := clones()
			orig := make([]raster.Float32Raster, len(gotOut))
			for i, o := range gotOut {
				orig[i] = o.clone().root
			}
			rid := fmt.Sprintf("%s chunked %v", id, chunked)
			var err error
			msg := func() (msg string) {
				defer func() {
					if v := recover(); v != nil {
						s, ok := v.(string)
						if !ok || !strings.HasPrefix(s, "engine: ") {
							t.Fatalf("%s: panic %v (%T), want an \"engine: \" message", rid, v, v)
						}
						msg = s
					}
				}()
				if chunked {
					err = exec.ProcessChunked(ctx, memorySinks(gotOut), memorySources(gotIn), k, opts)
				} else {
					err = exec.ProcessN(ctx, views(gotOut), views(gotIn), k, opts)
				}
				return ""
			}()
			if wantPanic := missingMask || chunked && sinksShareWords; (msg != "") != wantPanic {
				t.Fatalf("%s: panic %q, want a panic %v", rid, msg, wantPanic)
			}
			if err != nil {
				t.Fatalf("%s: %v", rid, err)
			}
			for i, o := range gotOut {
				if msg != "" {
					requireSameRoots(t, fmt.Sprintf("%s: panicked, dst[%d]", rid, i), o.root, orig[i])
					continue
				}
				requireSameRoots(t, fmt.Sprintf("%s: dst[%d]", rid, i), o.root, wantOut[i].root)
			}
		}
	})
}
