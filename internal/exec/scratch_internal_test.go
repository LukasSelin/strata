package exec

import (
	"testing"

	"github.com/LukasSelin/strata/raster"
)

// TestSizeClass checks what slab relies on: a class holds n, wastes at
// most an eighth above 16, maps its own size back to itself (which is
// how put finds a block's pool from its capacity), and fits the table.
func TestSizeClass(t *testing.T) {
	check := func(n int) {
		c, size := sizeClass(n)
		if size < n || (n > 16 && size-n > n/8) {
			t.Fatalf("sizeClass(%d) = size %d", n, size)
		}
		if c2, size2 := sizeClass(size); c2 != c || size2 != size {
			t.Fatalf("sizeClass(%d) = (%d, %d), but sizeClass(%d) = (%d, %d)", n, c, size, size, c2, size2)
		}
		if c < 0 || c >= classes {
			t.Fatalf("sizeClass(%d) = class %d, want [0, %d)", n, c, classes)
		}
	}
	prev := 0
	for n := 1; n <= 1<<17; n++ {
		check(n)
		c, _ := sizeClass(n)
		if c < prev {
			t.Fatalf("sizeClass(%d) = class %d, below %d for n-1", n, c, prev)
		}
		prev = c
	}
	for _, n := range []int{1 << 20, 1<<20 + 1, 3 << 30, 1 << 40, 1<<61 + 1, 1 << 62} {
		check(n)
	}
	for k := range 62 {
		if _, size := sizeClass(1 << k); size != 1<<k {
			t.Errorf("sizeClass(1<<%d) = size %d, want no rounding", k, size)
		}
	}
}

// TestReleaseClearsViews checks that a pooled view block keeps no raster
// alive: releaseScratch clears it, including beyond the length the last
// call asked for.
func TestReleaseClearsViews(t *testing.T) {
	e := &job{workers: make([]worker, 1)}
	wk := &e.workers[0]
	wk.views = viewPool.get(20)
	block := wk.views
	full := (*block)[:cap(*block)]
	for i := range full {
		full[i] = raster.NewFloat32(1, 1, []float32{1})
	}
	*block = (*block)[:3]
	e.releaseScratch()
	for i, v := range full {
		if v.Data != nil {
			t.Fatalf("view %d still holds %v after release", i, v.Data)
		}
	}
	if wk.views != nil || wk.kscratch.Views != nil {
		t.Fatal("worker still holds its scratch after release")
	}
}
