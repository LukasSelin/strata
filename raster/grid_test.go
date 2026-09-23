package raster

import "testing"

func TestGridWindow(t *testing.T) {
	g := Grid{
		Width: 100, Height: 50,
		ResolutionX: 10, ResolutionY: -10,
		OriginX: 500000, OriginY: 6400000,
		CRS: CRS{Code: "EPSG:25833"},
	}
	w := g.Window(20, 5, 30, 10)
	want := Grid{
		Width: 30, Height: 10,
		ResolutionX: 10, ResolutionY: -10,
		OriginX: 500200, OriginY: 6399950,
		CRS: CRS{Code: "EPSG:25833"},
	}
	if w != want {
		t.Fatalf("Grid.Window = %+v, want %+v", w, want)
	}
	if g.Width != 100 || g.OriginX != 500000 {
		t.Fatal("Grid.Window modified its receiver")
	}
	if nested := w.Window(3, 4, 1, 1); nested != g.Window(23, 9, 1, 1) {
		t.Fatalf("nested grid window %+v != direct %+v", nested, g.Window(23, 9, 1, 1))
	}
	mustPanic(t, "outside", func() { g.Window(71, 0, 30, 1) })
	mustPanic(t, "must be positive", func() { g.Window(0, 0, 0, 1) })
}

func TestCRSMatches(t *testing.T) {
	a, b, unknown := CRS{Code: "EPSG:25833"}, CRS{Code: "EPSG:4326"}, CRS{}
	for _, c := range []struct {
		x, y CRS
		want bool
	}{
		{a, a, true},
		{a, b, false},
		{a, unknown, true},
		{unknown, b, true},
		{unknown, unknown, true},
		// Codes are opaque: the same system under another spelling differs.
		{a, CRS{Code: "epsg:25833"}, false},
	} {
		if got := c.x.Matches(c.y); got != c.want {
			t.Errorf("%q.Matches(%q) = %v, want %v", c.x.Code, c.y.Code, got, c.want)
		}
		if got := c.y.Matches(c.x); got != c.want {
			t.Errorf("%q.Matches(%q) = %v, want %v", c.y.Code, c.x.Code, got, c.want)
		}
	}
}

func TestDataset(t *testing.T) {
	g := Grid{Width: 6, Height: 4, ResolutionX: 2, ResolutionY: -2}
	r := NewFloat32Stride(6, 4, 8, seq(dataLen(6, 4, 8)))
	r.Valid = NewMask(len(r.Data))
	d := NewDataset(g, r)

	w := d.Window(2, 1, 3, 2)
	if w.Grid != g.Window(2, 1, 3, 2) {
		t.Fatalf("dataset grid window = %+v", w.Grid)
	}
	checkView(t, r, w.Raster, 2, 1)

	mustPanic(t, "grid is 5×4 but raster is 6×4", func() {
		NewDataset(Grid{Width: 5, Height: 4}, r)
	})
}
