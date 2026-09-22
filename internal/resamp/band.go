package resamp

import (
	"math"
	"math/bits"
	"sync"

	"github.com/LukasSelin/strata/raster"
)

// Source is the source data a band reads: a raster view whose cell (0, 0)
// is source cell (X0, Y0). It must hold the band's footprint.
type Source struct {
	R      raster.Float32Raster
	X0, Y0 int
}

// Workspace is one worker's scratch for Band, grown on demand and reused
// across bands. The zero value is ready to use; it must not be shared
// between goroutines.
type Workspace struct {
	t, tm, tf []float32
	xm, xf    []float32
	hs        []float32
	nm, d     []float32
	cnt, csum []float32
	ones      []float32
	wcnt      []float32
	wsum      []float32
	prefix    []int32
	vbits     []uint64
}

func grow[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	return s[:n]
}

// nan is the value every invalid output cell gets in Data, so that the
// unspecified contents under a cleared bit (DESIGN.md §31) are at least
// the same for every tiling and backend.
var nan = float32(math.NaN())

// maskedChunk is the widest run of output columns Band gives one
// footprint when the source has a mask. A footprint with an invalid cell
// takes the masked path, three horizontal passes and a per-cell finish,
// so a band is cut into chunks that each decide for themselves, and one
// invalid cell slows only the chunk around it. Unmasked sources take the
// whole band at once.
const maskedChunk = 256

// Band computes the output cells [x0, x0+dst.Width) × [y0,
// y0+dst.Height) into dst, a view whose cell (0, 0) is output cell (x0,
// y0), reading src. If dst has a mask, Band writes its bits for the band
// under mu, which may be nil when no other goroutine writes dst's mask
// words. It returns the number of source cells the band read, for
// engine.Stats.
//
// A cell is valid when it is covered on both axes and, if src has a
// mask, when its centre cell is valid (except for Average), the valid
// weight is positive, and, under HalfValid, at least half the cells its
// kernel reaches are valid (not required of a cell that copies one). A valid cell whose taps are all valid has the
// unmasked sums' value; one with invalid taps is renormalised over the
// valid ones. Invalid cells get NaN in Data. The value and validity of a
// cell depend only on the tables and the source, not on the band, so
// every tiling gives the same bits.
func (p *Plan) Band(ws *Workspace, dst raster.Float32Raster, x0, y0 int, src Source, mu *sync.Mutex) (read int) {
	w, h := dst.Width, dst.Height
	x1, y1 := x0+w, y0+h
	cx0, cx1 := max(x0, p.X.Lo), min(x1, p.X.Hi)
	cy0, cy1 := max(y0, p.Y.Lo), min(y1, p.Y.Hi)
	words := raster.MaskWords(w)
	ws.vbits = grow(ws.vbits, h*words)
	if dst.Valid != nil {
		clear(ws.vbits)
	}
	for r := range h {
		row := dst.Row(r)
		if y := y0 + r; y < cy0 || y >= cy1 || cx0 >= cx1 {
			fillNaN(row)
			continue
		}
		fillNaN(row[:cx0-x0])
		fillNaN(row[cx1-x0:])
	}
	if cx0 < cx1 && cy0 < cy1 {
		if p.Method == Nearest {
			read = p.nearest(ws, dst, x0, y0, src, cx0, cx1, cy0, cy1, words)
		} else {
			step := cx1 - cx0
			if src.R.Valid != nil {
				step = maskedChunk
			}
			for a := cx0; a < cx1; a += step {
				read += p.chunk(ws, dst, x0, y0, src, a, min(a+step, cx1), cy0, cy1, words)
			}
		}
	}
	if dst.Valid != nil {
		if mu != nil {
			mu.Lock()
			defer mu.Unlock()
		}
		for r := range h {
			raster.MaskCopyRange(dst.Valid, dst.ValidOffset+r*dst.Stride, ws.vbits[r*words:(r+1)*words], 0, w)
		}
	}
	return read
}

// setBit marks output column c of band row r valid.
func setBit(v []uint64, words, r, i int) {
	v[r*words+i>>6] |= 1 << uint(i&63)
}

// srcValid reports whether source cell (x, y) is valid.
func srcValid(src Source, x, y int) bool {
	return raster.MaskGet(src.R.Valid, src.R.ValidOffset+(y-src.Y0)*src.R.Stride+x-src.X0)
}

// nearest is Band for Nearest: a copy through the index tables, bits and
// NaN payloads included.
func (p *Plan) nearest(ws *Workspace, dst raster.Float32Raster, x0, y0 int, src Source, cx0, cx1, cy0, cy1, words int) int {
	masked := src.R.Valid != nil
	for y := cy0; y < cy1; y++ {
		r := y - y0
		row := dst.Row(r)
		sy := int(p.Y.Centre[y])
		srow := src.R.Row(sy - src.Y0)
		for c := cx0; c < cx1; c++ {
			sx := int(p.X.Centre[c])
			if masked && !srcValid(src, sx, sy) {
				row[c-x0] = nan
				continue
			}
			row[c-x0] = srow[sx-src.X0]
			setBit(ws.vbits, words, r, c-x0)
		}
	}
	return (cx1 - cx0) * (cy1 - cy0)
}

// chunk computes the covered output columns [cx0, cx1) of rows [cy0,
// cy1) of the band, over their own footprint, and returns its size.
func (p *Plan) chunk(ws *Workspace, dst raster.Float32Raster, x0, y0 int, src Source, cx0, cx1, cy0, cy1, words int) int {
	fx0, fy0, fx1, fy1 := p.Footprint(cx0, cy0, cx1, cy1)
	fpW, fpH, cw := fx1-fx0, fy1-fy0, cx1-cx0
	stride := src.R.Stride
	sdata := src.R.Data[(fy0-src.Y0)*stride+fx0-src.X0:]
	masked := src.R.Valid != nil && !allValid(src, fx0, fy0, fpW, fpH)

	ws.hs = grow(ws.hs, HScratch(fpW))
	ws.t = grow(ws.t, fpH*cw)
	HRows(ws.t, cw, sdata, stride, fpH, &p.X, cx0, cx1, fx0, ws.hs)
	if masked {
		p.planes(ws, src, fx0, fy0, fpW, fpH, cx0, cx1)
		ws.tm = grow(ws.tm, fpH*cw)
		ws.tf = grow(ws.tf, fpH*cw)
		HRows(ws.tm, cw, ws.xm, fpW, fpH, &p.X, cx0, cx1, fx0, ws.hs)
		HRows(ws.tf, cw, ws.xf, fpW, fpH, &p.X, cx0, cx1, fx0, ws.hs)
		ws.nm = grow(ws.nm, cw)
		ws.d = grow(ws.d, cw)
		ws.csum = grow(ws.csum, cw)
		ws.wsum = grow(ws.wsum, cw)
		if n := max(p.Y.MaxTaps, p.Y.MaxWin); len(ws.ones) < n {
			ws.ones = make([]float32, n)
			for i := range ws.ones {
				ws.ones[i] = 1
			}
		}
	}

	for y := cy0; y < cy1; y++ {
		r := y - y0
		out := dst.Row(r)[cx0-x0 : cx1-x0]
		ty := int(p.Y.First[y]) - fy0
		ny := int(p.Y.Taps[y])
		wy := p.Y.W[p.Y.Off[y] : int(p.Y.Off[y])+ny]
		VRow(out, ws.t[ty*cw:], cw, wy)
		if !masked {
			switch {
			case p.Cubic4 && p.Y.Clipped[y]:
				for c := cx0; c < cx1; c++ {
					out[c-cx0] = p.fallback(src, c, y, false)
				}
			case p.Cubic4:
				for _, c := range p.ClippedX {
					if c := int(c); c >= cx0 && c < cx1 {
						out[c-cx0] = p.fallback(src, c, y, false)
					}
				}
			}
			if dst.Valid != nil {
				raster.MaskFillRange(ws.vbits[r*words:(r+1)*words], cx0-x0, cw, true)
			}
			continue
		}

		VRow(ws.nm, ws.tm[ty*cw:], cw, wy)
		VRow(ws.d, ws.tf[ty*cw:], cw, wy)
		// The valid taps of each cell: counts are small integers, exact in
		// float32, so the vertical pass with unit weights sums them.
		csum := ws.csum[:cw]
		VRow(csum, ws.cnt[ty*cw:], cw, ws.ones[:ny])
		if p.HalfValid {
			// The valid cells of each window, summed like the tap counts.
			VRow(ws.wsum[:cw], ws.wcnt[(int(p.Y.WinFirst[y])-fy0)*cw:], cw, ws.ones[:p.Y.WinN[y]])
		}
		cy := int(p.Y.Centre[y])
		for c := cx0; c < cx1; c++ {
			i := c - cx0
			ok := ws.d[i] > 0
			if p.Method != Average {
				ok = ok && srcValid(src, int(p.X.Centre[c]), cy)
			}
			if ok && p.HalfValid && (!p.X.Exact[c] || !p.Y.Exact[y]) {
				ok = 2*int64(ws.wsum[i]) >= int64(p.X.WinN[c])*int64(p.Y.WinN[y])
			}
			if !ok {
				out[i] = nan
				continue
			}
			full := int(csum[i]) == int(p.X.Taps[c])*ny
			switch {
			case p.Cubic4 && (!full || p.X.Clipped[c] || p.Y.Clipped[y]):
				out[i] = p.fallback(src, c, y, true)
			case !full:
				out[i] = ws.nm[i] / ws.d[i]
			}
			setBit(ws.vbits, words, r, c-x0)
		}
	}
	return fpW * fpH
}

// planes builds the masked planes of the footprint: xm holds the valid
// cells' values and +0 under cleared bits, xf holds 1 and +0, both by
// selection, never by multiplying (Data under a cleared bit may be Inf or
// NaN). cnt[j*cw + c-cx0] is the number of valid cells among output
// column c's taps in footprint row j.
func (p *Plan) planes(ws *Workspace, src Source, fx0, fy0, fpW, fpH, cx0, cx1 int) {
	cw := cx1 - cx0
	ws.xm = grow(ws.xm, fpW*fpH)
	ws.xf = grow(ws.xf, fpW*fpH)
	ws.cnt = grow(ws.cnt, fpH*cw)
	if p.HalfValid {
		ws.wcnt = grow(ws.wcnt, fpH*cw)
	}
	ws.prefix = grow(ws.prefix, fpW+1)
	prefix := ws.prefix
	for j := range fpH {
		sy := fy0 + j - src.Y0
		srow := src.R.Row(sy)[fx0-src.X0 : fx0-src.X0+fpW]
		off := src.R.ValidOffset + sy*src.R.Stride + fx0 - src.X0
		xm := ws.xm[j*fpW : (j+1)*fpW]
		xf := ws.xf[j*fpW : (j+1)*fpW]
		n := int32(0)
		for i := 0; i < fpW; i += 64 {
			k := min(64, fpW-i)
			b := raster.MaskBits(src.R.Valid, off+i, k)
			for q := range k {
				prefix[i+q] = n
				if b&(1<<uint(q)) != 0 {
					xm[i+q], xf[i+q] = srow[i+q], 1
					n++
				} else {
					xm[i+q], xf[i+q] = 0, 0
				}
			}
		}
		prefix[fpW] = n
		cnt := ws.cnt[j*cw : (j+1)*cw]
		for c := cx0; c < cx1; c++ {
			f := int(p.X.First[c]) - fx0
			cnt[c-cx0] = float32(prefix[f+int(p.X.Taps[c])] - prefix[f])
		}
		if p.HalfValid {
			wcnt := ws.wcnt[j*cw : (j+1)*cw]
			for c := cx0; c < cx1; c++ {
				f := int(p.X.WinFirst[c]) - fx0
				wcnt[c-cx0] = float32(prefix[f+int(p.X.WinN[c])] - prefix[f])
			}
		}
	}
}

// fallback is gdalwarp's bilinear for a four-sample cubic cell that lost
// a tap to the edge or to NoData: unwidened bilinear over the valid
// cells. Its sums run in the canonical order; with all its taps valid it
// is the unmasked sum, otherwise the masked sum over the valid weight.
func (p *Plan) fallback(src Source, c, r int, masked bool) float32 {
	bx, by := &p.BX, &p.BY
	fx, nx := int(bx.First[c]), int(bx.Taps[c])
	fy, ny := int(by.First[r]), int(by.Taps[r])
	wx := bx.W[bx.Off[c] : int(bx.Off[c])+nx]
	wy := by.W[by.Off[r] : int(by.Off[r])+ny]
	var n, nm, d float32
	full := true
	for k := range ny {
		sy := fy + k - src.Y0
		row := src.R.Row(sy)
		var t, tm, tf float32
		for i := range nx {
			sx := fx + i - src.X0
			x := row[sx]
			v := !masked || src.R.IsValid(sx, sy)
			var xm, xf float32
			if v {
				xm, xf = x, 1
			} else {
				full = false
			}
			if i == 0 {
				t, tm, tf = float32(wx[0]*x), float32(wx[0]*xm), float32(wx[0]*xf)
				continue
			}
			t += float32(wx[i] * x)
			tm += float32(wx[i] * xm)
			tf += float32(wx[i] * xf)
		}
		if k == 0 {
			n, nm, d = float32(wy[0]*t), float32(wy[0]*tm), float32(wy[0]*tf)
			continue
		}
		n += float32(wy[k] * t)
		nm += float32(wy[k] * tm)
		d += float32(wy[k] * tf)
	}
	if full {
		return n
	}
	return nm / d
}

// allValid reports whether every source cell of the fpW×fpH footprint at
// (fx0, fy0) is valid.
func allValid(src Source, fx0, fy0, fpW, fpH int) bool {
	for j := range fpH {
		off := src.R.ValidOffset + (fy0+j-src.Y0)*src.R.Stride + fx0 - src.X0
		for i := 0; i < fpW; i += 64 {
			k := min(64, fpW-i)
			if b := raster.MaskBits(src.R.Valid, off+i, k); bits.OnesCount64(b) != k {
				return false
			}
		}
	}
	return true
}

func fillNaN(s []float32) {
	for i := range s {
		s[i] = nan
	}
}
