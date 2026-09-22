// Package public is an unmarked package, such as terrain.
package public

import "fakesimd/archsimd" // want `fakesimd/archsimd imported outside a kernel package`

var _ archsimd.Float32x8
