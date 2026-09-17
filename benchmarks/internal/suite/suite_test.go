package suite

import (
	"math"
	"runtime"
	"testing"

	"strata/raster"
)

func TestCaseName(t *testing.T) {
	c := Case{Size: 4096, Masked: true, Backend: SIMD, Workers: 1}
	if got, want := c.Name(), "size=4096/mask=on/backend=simd/workers=1"; got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
}

func TestCPUs(t *testing.T) {
	physical, logical := CPUs()
	t.Logf("physical cores: %d, logical CPUs: %d, usable: %d", physical, logical, runtime.NumCPU())
	if physical < 0 || logical < 0 || (logical > 0 && physical > logical) {
		t.Errorf("CPUs() = %d physical, %d logical", physical, logical)
	}
	if logical > 0 && runtime.NumCPU() > logical {
		t.Errorf("runtime.NumCPU() = %d exceeds %d logical CPUs", runtime.NumCPU(), logical)
	}
}

func TestFillUniform(t *testing.T) {
	data := make([]float32, 1<<16)
	FillUniform(data, 1, -100, 100)
	var sum float64
	for i, v := range data {
		if v < -100 || v >= 100 {
			t.Fatalf("data[%d] = %v, outside [-100, 100)", i, v)
		}
		sum += float64(v)
	}
	if mean := sum / float64(len(data)); math.Abs(mean) > 2 {
		t.Errorf("mean = %v, want about 0", mean)
	}
}

func TestRandomMask(t *testing.T) {
	const n = 1<<16 + 7
	m := RandomMask(n, 1, 0.1)
	invalid := 0
	for i := range n {
		if !raster.MaskGet(m, i) {
			invalid++
		}
	}
	if frac := float64(invalid) / n; frac < 0.09 || frac > 0.11 {
		t.Errorf("invalid fraction = %v, want about 0.1", frac)
	}
	if last := m[len(m)-1]; last>>(n&63) != 0 {
		t.Errorf("bits past n are set: %#x", last)
	}
}
