// Package archsimd stands in for simd/archsimd, which only exists in
// GOEXPERIMENT=simd builds.
package archsimd

type Float32x8 struct{ v [8]float32 }
