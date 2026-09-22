package resamp

import "testing"

// TestCubic4 checks the plan's gdalwarp cubic rules: four-sample mode
// when neither axis widens, with the edge columns flagged Clipped.
func TestCubic4(t *testing.T) {
	x := Spec{N: 16, Res: 0.5, SrcN: 8, SrcRes: 1}
	p := NewPlan(Cubic, x, x)
	if !p.Cubic4 || p.HalfValid {
		t.Fatalf("up2: Cubic4 %v HalfValid %v", p.Cubic4, p.HalfValid)
	}
	var clipped []int
	for c, v := range p.X.Clipped {
		if v {
			clipped = append(clipped, c)
		}
	}
	// u = 0.25, 0.75, 1.25 reach cell -1 with non-zero weight; mirrored on
	// the right.
	want := []int{0, 1, 2, 13, 14, 15}
	if len(clipped) != len(want) {
		t.Fatalf("clipped columns %v, want %v", clipped, want)
	}
	for i := range want {
		if clipped[i] != want[i] {
			t.Fatalf("clipped columns %v, want %v", clipped, want)
		}
	}
	down := Spec{N: 4, Res: 2, SrcN: 8, SrcRes: 1}
	if p := NewPlan(Cubic, down, x); p.Cubic4 {
		t.Error("a widened axis must leave four-sample mode")
	}
	if p := NewPlan(Lanczos, down, x); !p.HalfValid {
		t.Error("a widened Lanczos must apply the half-valid rule")
	}
}
