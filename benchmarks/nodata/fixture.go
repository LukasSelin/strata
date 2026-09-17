package nodata

import (
	"math"
	"math/rand/v2"
)

// Pattern describes how NoData cells are distributed in a fixture.
type Pattern struct {
	Name      string
	Density   float64 // target fraction of NoData cells
	Clustered bool    // discs of NoData rather than independent cells
}

// Patterns are the NoData distributions the spike benchmarks.
var Patterns = []Pattern{
	{Name: "0pct", Density: 0},
	{Name: "1pct-scattered", Density: 0.01},
	{Name: "30pct-scattered", Density: 0.30},
	{Name: "30pct-clustered", Density: 0.30, Clustered: true},
}

// Fixture holds one pair of inputs (A, B) in all three representations.
// The mask representation reuses the sentinel slices as its Data: the
// value under a cleared bit is unspecified, and -9999 is what a reader
// that did not normalise the fill value would leave there.
type Fixture struct {
	W, H int

	// Invalid[i] is the ground truth NoData flag for cell i of A and B.
	InvalidA, InvalidB []bool

	SentinelA, SentinelB []float32
	NaNA, NaNB           []float32
	ValidA, ValidB       []uint64
}

// NewFixture builds a deterministic fixture. A is a synthetic DEM, B an
// independent smooth field; each gets its own NoData layout of the same
// pattern (different seeds), so Add's output NoData density is roughly
// 1-(1-d)² of the input density d.
func NewFixture(w, h int, p Pattern) *Fixture {
	n := w * h
	f := &Fixture{W: w, H: h}
	f.InvalidA = invalidLayout(w, h, p, 1)
	f.InvalidB = invalidLayout(w, h, p, 2)

	demA := make([]float32, n)
	demB := make([]float32, n)
	rng := rand.New(rand.NewPCG(42, 7))
	for y := 0; y < h; y++ {
		fy := float64(y)
		for x := 0; x < w; x++ {
			fx := float64(x)
			i := y*w + x
			demA[i] = float32(800 +
				300*math.Sin(fx/97)*math.Cos(fy/131) +
				40*math.Sin((fx+fy)/13) +
				rng.Float64()*2)
			demB[i] = float32(15 + 10*math.Cos(fx/211+fy/173) + rng.Float64())
		}
	}

	f.SentinelA = applyFill(demA, f.InvalidA, Sentinel)
	f.SentinelB = applyFill(demB, f.InvalidB, Sentinel)
	f.NaNA = applyFill(demA, f.InvalidA, NaN32)
	f.NaNB = applyFill(demB, f.InvalidB, NaN32)
	f.ValidA = MaskFromInvalid(f.InvalidA)
	f.ValidB = MaskFromInvalid(f.InvalidB)
	return f
}

func applyFill(src []float32, invalid []bool, fill float32) []float32 {
	out := make([]float32, len(src))
	for i, v := range src {
		if invalid[i] {
			v = fill
		}
		out[i] = v
	}
	return out
}

func invalidLayout(w, h int, p Pattern, seed uint64) []bool {
	n := w * h
	inv := make([]bool, n)
	if p.Density == 0 {
		return inv
	}
	rng := rand.New(rand.NewPCG(seed, uint64(w)<<32|uint64(h))) // #nosec G115 -- any bits make a seed
	if !p.Clustered {
		for i := range inv {
			inv[i] = rng.Float64() < p.Density
		}
		return inv
	}

	// Clustered: random discs (voids, lakes, radar shadow) with radii
	// independent of raster size, stamped until the target is reached.
	target := int(p.Density * float64(n))
	count := 0
	for count < target {
		r := 4 + rng.IntN(60)
		cx, cy := rng.IntN(w), rng.IntN(h)
		for y := max(cy-r, 0); y <= min(cy+r, h-1); y++ {
			dy := y - cy
			for x := max(cx-r, 0); x <= min(cx+r, w-1); x++ {
				dx := x - cx
				if dx*dx+dy*dy > r*r {
					continue
				}
				i := y*w + x
				if !inv[i] {
					inv[i] = true
					count++
				}
			}
		}
	}
	return inv
}

// InvalidFraction returns the fraction of true entries.
func InvalidFraction(inv []bool) float64 {
	c := 0
	for _, v := range inv {
		if v {
			c++
		}
	}
	return float64(c) / float64(len(inv))
}
