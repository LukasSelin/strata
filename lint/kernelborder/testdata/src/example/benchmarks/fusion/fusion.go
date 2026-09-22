// Package fusion is exempt: benchmark packages carry their own SIMD
// experiments.
package fusion

import "fakesimd/archsimd"

var _ archsimd.Float32x8
