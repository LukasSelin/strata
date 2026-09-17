package raster

import (
	"strings"
	"testing"
)

// mustPanic fails the test unless f panics with a message containing want.
func mustPanic(t *testing.T, want string, f func()) {
	t.Helper()
	defer func() {
		t.Helper()
		r := recover()
		if r == nil {
			t.Fatalf("expected panic containing %q, got none", want)
		}
		msg, _ := r.(string)
		if !strings.Contains(msg, want) {
			t.Fatalf("panic %q does not contain %q", msg, want)
		}
	}()
	f()
}

// seq returns 0, 1, ..., n-1 as float32.
func seq(n int) []float32 {
	d := make([]float32, n)
	for i := range d {
		d[i] = float32(i)
	}
	return d
}

// cell is the reference accessor the tests check views against.
func cell(r Float32Raster, x, y int) float32 {
	return r.Data[y*r.Stride+x]
}

func TestNewFloat32RejectsBadShapes(t *testing.T) {
	cases := []struct {
		name         string
		w, h, stride int
		dataLen      int
		want         string
	}{
		{"zero width", 0, 3, 0, 10, "dimensions must be positive"},
		{"zero height", 3, 0, 3, 10, "dimensions must be positive"},
		{"negative width", -1, 3, 3, 10, "dimensions must be positive"},
		{"negative height", 3, -2, 3, 10, "dimensions must be positive"},
		{"stride below width", 4, 2, 3, 20, "stride 3 is less than width 4"},
		{"short data", 3, 3, 3, 8, "data has 8 cells, need 9"},
		{"nil data", 1, 1, 1, 0, "data has 0 cells, need 1"},
		{"short strided data", 3, 3, 5, 12, "data has 12 cells, need 13"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			data := make([]float32, c.dataLen)
			mustPanic(t, c.want, func() { NewFloat32Stride(c.w, c.h, c.stride, data) })
			if c.stride == c.w {
				mustPanic(t, c.want, func() { NewFloat32(c.w, c.h, data) })
			}
		})
	}
	mustPanic(t, "dimensions must be positive", func() { NewFloat32Like(Float32Raster{}) })
}

func TestNewFloat32Wraps(t *testing.T) {
	data := seq(12) // two cells more than needed
	r := NewFloat32(5, 2, data)
	if r.Width != 5 || r.Height != 2 || r.Stride != 5 || r.Valid != nil || r.ValidOffset != 0 {
		t.Fatalf("unexpected header %+v", r)
	}
	if len(r.Data) != 10 || cap(r.Data) != 10 {
		t.Fatalf("len/cap = %d/%d, want 10/10", len(r.Data), cap(r.Data))
	}
	data[7] = 99
	if cell(r, 2, 1) != 99 {
		t.Fatal("NewFloat32 copied data")
	}
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestOneByOne(t *testing.T) {
	r := NewFloat32(1, 1, []float32{42})
	if row := r.Row(0); len(row) != 1 || row[0] != 42 {
		t.Fatalf("Row(0) = %v", row)
	}
	w := r.Window(0, 0, 1, 1)
	w.Row(0)[0] = 7
	if r.Data[0] != 7 {
		t.Fatal("1×1 window does not share memory")
	}
	if !w.IsValid(0, 0) {
		t.Fatal("cell of maskless raster reported invalid")
	}
	mustPanic(t, "outside", func() { r.Window(0, 0, 2, 1) })
	mustPanic(t, "outside", func() { r.Window(1, 0, 1, 1) })
	mustPanic(t, "row 1 out of range", func() { r.Row(1) })

	r.Valid = NewMask(1)
	if len(r.Valid) != 1 || r.Valid[0] != 1 {
		t.Fatalf("NewMask(1) = %b", r.Valid)
	}
	w = r.Window(0, 0, 1, 1)
	w.SetValid(0, 0, false)
	if r.IsValid(0, 0) || r.Valid[0] != 0 {
		t.Fatal("invalidating through 1×1 window not visible in parent")
	}
}

func TestRowStride(t *testing.T) {
	for _, shape := range []struct{ w, h, stride int }{
		{7, 5, 7}, {7, 5, 9}, {3, 1, 64}, {1, 6, 2}, {13, 3, 64},
	} {
		n := dataLen(shape.w, shape.h, shape.stride)
		r := NewFloat32Stride(shape.w, shape.h, shape.stride, seq(n))
		if len(r.Data) != n {
			t.Fatalf("%+v: len(Data) = %d, want %d", shape, len(r.Data), n)
		}
		for y := range r.Height {
			row := r.Row(y)
			if len(row) != shape.w || cap(row) != shape.w {
				t.Fatalf("%+v: row %d len/cap = %d/%d", shape, y, len(row), cap(row))
			}
			for x, v := range row {
				if want := float32(y*shape.stride + x); v != want {
					t.Fatalf("%+v: row %d[%d] = %v, want %v", shape, y, x, v, want)
				}
			}
		}
		mustPanic(t, "out of range", func() { r.Row(-1) })
		mustPanic(t, "out of range", func() { r.Row(r.Height) })
	}
}

func TestRowWritesSkipPadding(t *testing.T) {
	const w, h, stride = 3, 3, 5
	data := make([]float32, w*h+(h-1)*(stride-w)) // 13
	for i := range data {
		data[i] = -1
	}
	r := NewFloat32Stride(w, h, stride, data)
	for y := range h {
		row := r.Row(y)
		for x := range row {
			row[x] = 1
		}
		_ = append(row, 2) // capacity is Width: must reallocate, not clobber padding
	}
	for i, v := range data {
		inRow := i%stride < w
		if inRow && v != 1 || !inRow && v != -1 {
			t.Fatalf("data[%d] = %v after row writes (in row: %v)", i, v, inRow)
		}
	}
}

func TestIndexAndCellAccessPanics(t *testing.T) {
	r := NewFloat32Stride(4, 3, 6, make([]float32, 16))
	if got := r.Index(3, 2); got != 15 {
		t.Fatalf("Index(3, 2) = %d, want 15", got)
	}
	for _, xy := range [][2]int{{-1, 0}, {0, -1}, {4, 0}, {0, 3}} {
		mustPanic(t, "outside", func() { r.Index(xy[0], xy[1]) })
		mustPanic(t, "outside", func() { r.IsValid(xy[0], xy[1]) })
	}
	mustPanic(t, "without a validity mask", func() { r.SetValid(0, 0, false) })
}

func TestNewFloat32Like(t *testing.T) {
	parent := NewFloat32Stride(10, 6, 16, seq(dataLen(10, 6, 16)))
	win := parent.Window(2, 1, 5, 3)

	like := NewFloat32Like(win)
	if like.Width != 5 || like.Height != 3 || like.Stride != 5 || len(like.Data) != 15 {
		t.Fatalf("NewFloat32Like(window) = %dx%d stride %d len %d",
			like.Width, like.Height, like.Stride, len(like.Data))
	}
	if like.Valid != nil {
		t.Fatal("NewFloat32Like gave a mask to a raster whose source has none")
	}
	for _, v := range like.Data {
		if v != 0 {
			t.Fatal("NewFloat32Like data not zeroed")
		}
	}
	like.Data[0] = 1
	if parent.Data[parent.Index(2, 1)] == 1 {
		t.Fatal("NewFloat32Like shares memory with its source")
	}

	parent.Valid = NewMask(len(parent.Data))
	parent.SetValid(3, 2, false)
	like = NewFloat32Like(parent.Window(2, 1, 5, 3))
	if like.Valid == nil || len(like.Valid) != MaskWords(15) {
		t.Fatalf("NewFloat32Like mask = %v, want %d words", like.Valid, MaskWords(15))
	}
	for y := range like.Height {
		for x := range like.Width {
			if !like.IsValid(x, y) {
				t.Fatalf("NewFloat32Like cell (%d, %d) not valid", x, y)
			}
		}
	}
	if err := like.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestValidate(t *testing.T) {
	good := NewFloat32Stride(5, 4, 8, make([]float32, dataLen(5, 4, 8)))
	good.Valid = NewMask(len(good.Data))
	if err := good.Validate(); err != nil {
		t.Fatalf("valid raster: %v", err)
	}
	if err := good.Window(1, 1, 4, 3).Validate(); err != nil {
		t.Fatalf("valid window: %v", err)
	}

	cases := []struct {
		name   string
		mutate func(r *Float32Raster)
		want   string
	}{
		{"zero value", func(r *Float32Raster) { *r = Float32Raster{} }, "dimensions must be positive"},
		{"stride", func(r *Float32Raster) { r.Stride = 4 }, "stride 4 is less than width 5"},
		{"short data", func(r *Float32Raster) { r.Data = r.Data[:len(r.Data)-1] }, "data has"},
		{"negative offset", func(r *Float32Raster) { r.ValidOffset = -1 }, "negative ValidOffset"},
		{"short mask", func(r *Float32Raster) { r.ValidOffset = 36 }, "mask has 64 bits, need 65"}, // 36 + 29 cells
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := good
			c.mutate(&r)
			err := r.Validate()
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("Validate() = %v, want error containing %q", err, c.want)
			}
		})
	}
}

// TestRowAndWindowDoNotAllocate checks that views cost only their headers.
func TestRowAndWindowDoNotAllocate(t *testing.T) {
	r := NewFloat32Stride(100, 80, 128, make([]float32, dataLen(100, 80, 128)))
	masked := r
	masked.Valid = NewMask(len(r.Data))

	var sink float32
	var sinkRaster Float32Raster
	checks := map[string]func(){
		"Row": func() {
			sink += r.Row(17)[3]
		},
		"Window": func() {
			sinkRaster = r.Window(3, 5, 40, 30)
		},
		"masked Window": func() {
			sinkRaster = masked.Window(3, 5, 40, 30)
		},
		"nested Window + Row": func() {
			w := masked.Window(3, 5, 40, 30).Window(1, 2, 10, 10)
			sink += w.Row(9)[9]
			_ = w.IsValid(9, 9)
		},
	}
	for name, f := range checks {
		if n := testing.AllocsPerRun(1000, f); n != 0 {
			t.Errorf("%s: %v allocs per run, want 0", name, n)
		}
	}
	_, _ = sink, sinkRaster
}
