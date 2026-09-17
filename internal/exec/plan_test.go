package exec_test

import (
	"testing"

	"strata/internal/exec"
)

// TestBands checks the band plan against nested loops: tiles in row-major
// order, bands of max(1, bandCells/tileWidth) rows within each tile.
func TestBands(t *testing.T) {
	for _, bc := range []int{1, 5, 64, 1 << 16} {
		restore := exec.SetBandCells(bc)
		for _, sz := range [][2]int{{0, 0}, {0, 3}, {4, 0}, {1, 1}, {1, 9}, {9, 1}, {7, 5}, {64, 33}, {100, 257}} {
			w, h := sz[0], sz[1]
			for _, tile := range [][2]int{{0, 0}, {1, 1}, {3, 2}, {7, 64}, {64, 7}, {256, 256}, {1000, 3}} {
				tw, th := tile[0], tile[1]
				var want [][4]int
				if w > 0 && h > 0 {
					etw, eth := w, h
					if tw > 0 {
						etw = min(tw, w)
					}
					if th > 0 {
						eth = min(th, h)
					}
					rows := max(1, bc/etw)
					for ty := 0; ty < h; ty += eth {
						for tx := 0; tx < w; tx += etw {
							for y := ty; y < min(ty+eth, h); y += rows {
								want = append(want, [4]int{tx, y, min(tx+etw, w), min(y+rows, ty+eth, h)})
							}
						}
					}
				}
				got := exec.Bands(w, h, tw, th)
				if len(got) != len(want) {
					t.Fatalf("bandCells=%d %dx%d tiles %dx%d: %d bands, want %d", bc, w, h, tw, th, len(got), len(want))
				}
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("bandCells=%d %dx%d tiles %dx%d: band %d = %v, want %v", bc, w, h, tw, th, i, got[i], want[i])
					}
				}
			}
		}
		restore()
	}
}
