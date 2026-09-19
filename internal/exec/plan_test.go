package exec_test

import (
	"fmt"
	"testing"

	"github.com/LukasSelin/strata/internal/exec"
)

// TestBandShape pins the rule that decides a band's shape from its tile
// and the kernel's radius (DESIGN.md §53). It is the one place the rule
// is written down twice; TestBands below takes the shape from here rather
// than restating it, so that only the numbering is checked there.
func TestBandShape(t *testing.T) {
	const cells = 1 << 16
	cases := []struct {
		name            string
		cells, minWidth int
		tileW, tileH, r int
		wantW, wantH    int
	}{
		// Radius 0 is whole tile rows, whatever the tile: there is no halo
		// to save and a full-width band keeps a pointwise kernel compact.
		{"pointwise wide", cells, 1024, 4096, 4096, 0, 4096, 16},
		{"pointwise narrow", cells, 1024, 64, 4096, 0, 64, 1024},
		{"pointwise one row", cells, 1024, 1 << 17, 4096, 0, 1 << 17, 1},

		// The shape this change exists for: a full-width strip of a wide
		// raster, banded 1024×64 instead of 4096×16.
		{"strip of a 4096 raster", cells, 1024, 4096, 256, 1, 1024, 64},
		{"whole 4096 raster", cells, 1024, 4096, 4096, 1, 1024, 64},
		{"8192 wide", cells, 1024, 8192, 1024, 1, 1024, 64},

		// A tile no taller than one whole-row band is already one band;
		// splitting it across its width would only add perimeter.
		{"tile is one band", cells, 1024, 4096, 16, 1, 4096, 16},
		{"tile is under one band", cells, 1024, 4096, 15, 1, 4096, 16},

		// Rounding to nearest, not up: 1536 stays whole rather than
		// becoming two columns of 768, and 1537 splits into two of 769.
		// That is the 3/4·minBandWidth bound the rule guarantees.
		{"just under the split", cells, 1024, 1535, 4096, 1, 1535, 42},
		{"just over the split", cells, 1024, 1536, 4096, 1, 768, 85},
		{"narrower than the floor", cells, 1024, 1000, 4096, 1, 1000, 65},

		// Equal columns, not a floor plus a narrow remainder.
		{"4000 divides evenly", cells, 1024, 4000, 4096, 1, 1000, 65},
		{"3000 into three", cells, 1024, 3000, 4096, 1, 1000, 65},

		// A lowered floor is what lets a small test raster reach a
		// two-dimensional band at all.
		{"low floor", 97, 8, 64, 64, 1, 10, 9},
		{"low floor, radius 0", 97, 8, 64, 64, 0, 64, 1},

		// Above the floor the square wins, so a large band area is
		// squarer than minBandWidth alone would make it.
		{"square beats the floor", 1 << 24, 1024, 8192, 8192, 1, 4096, 4096},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer exec.SetBandCells(c.cells)()
			defer exec.SetBandMinWidth(c.minWidth)()
			w, h := exec.BandShape(c.tileW, c.tileH, c.r)
			if w != c.wantW || h != c.wantH {
				t.Errorf("BandShape(%d, %d, %d) = %d×%d, want %d×%d",
					c.tileW, c.tileH, c.r, w, h, c.wantW, c.wantH)
			}
			if w > c.tileW {
				t.Errorf("band %d is wider than its %d-wide tile", w, c.tileW)
			}
		})
	}
}

// TestBands checks the band plan against nested loops: tiles in row-major
// order, and within a tile the bands of the shape BandShape chose, row of
// bands by row of bands and left to right within a row.
func TestBands(t *testing.T) {
	for _, bc := range []int{1, 5, 64, 1 << 16} {
		for _, mw := range []int{2, 8, 1024} {
			for _, sz := range [][2]int{{0, 0}, {0, 3}, {4, 0}, {1, 1}, {1, 9}, {9, 1}, {7, 5}, {64, 33}, {100, 257}} {
				w, h := sz[0], sz[1]
				for _, tile := range [][2]int{{0, 0}, {1, 1}, {3, 2}, {7, 64}, {64, 7}, {256, 256}, {1000, 3}} {
					for _, r := range []int{0, 1, 3} {
						tw, th := tile[0], tile[1]
						restore := exec.SetBandCells(bc)
						restoreW := exec.SetBandMinWidth(mw)
						want := wantBands(w, h, tw, th, r)
						got := exec.Bands(w, h, tw, th, r)
						restoreW()
						restore()
						id := func() string {
							return fmt.Sprintf("bandCells=%d minWidth=%d %dx%d tiles %dx%d r=%d",
								bc, mw, w, h, tw, th, r)
						}
						if len(got) != len(want) {
							t.Fatalf("%s: %d bands, want %d", id(), len(got), len(want))
						}
						for i := range want {
							if got[i] != want[i] {
								t.Fatalf("%s: band %d = %v, want %v", id(), i, got[i], want[i])
							}
						}
					}
				}
			}
		}
	}
}

// wantBands is the plan written as the nested loops it stands for.
func wantBands(w, h, tw, th, r int) [][4]int {
	if w <= 0 || h <= 0 {
		return nil
	}
	tileW, tileH := w, h
	if tw > 0 {
		tileW = min(tw, w)
	}
	if th > 0 {
		tileH = min(th, h)
	}
	bandW, bandH := exec.BandShape(tileW, tileH, r)
	var out [][4]int
	for ty := 0; ty < h; ty += tileH {
		for tx := 0; tx < w; tx += tileW {
			for y := ty; y < min(ty+tileH, h); y += bandH {
				for x := tx; x < min(tx+tileW, w); x += bandW {
					out = append(out, [4]int{
						x, y,
						min(x+bandW, tx+tileW, w),
						min(y+bandH, ty+tileH, h),
					})
				}
			}
		}
	}
	return out
}
