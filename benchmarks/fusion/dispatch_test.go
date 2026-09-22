package fusion

import (
	"reflect"
	"testing"

	"github.com/LukasSelin/strata/internal/vec"
)

// TestFusedFollowsBackend checks that NewFused picks the vector loop
// exactly when internal/vec runs this architecture's SIMD backend, so the
// fused form is measured on the same backend as the other two.
func TestFusedFollowsBackend(t *testing.T) {
	defer vec.UseScalar(false)
	same := func(f, g func(dst, a, b, c, d, e, f []float32)) bool {
		return reflect.ValueOf(f).Pointer() == reflect.ValueOf(g).Pointer()
	}
	vec.UseScalar(true)
	if !same(NewFused().mul6, mul6Scalar) {
		t.Error("with vec on scalar, NewFused did not pick mul6Scalar")
	}
	vec.UseScalar(false)
	wantSIMD := haveSIMD && vec.Backend() == simdBackend
	if got := same(NewFused().mul6, mul6SIMD); got != wantSIMD {
		t.Errorf("vec backend %q: NewFused picked mul6SIMD = %v, want %v", vec.Backend(), got, wantSIMD)
	}
	if haveSIMD && vec.Backend() != simdBackend {
		t.Errorf("vec backend %q, want %q on a machine with this build's SIMD", vec.Backend(), simdBackend)
	}
}
