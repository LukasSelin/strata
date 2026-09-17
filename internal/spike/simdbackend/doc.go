// Package simdbackend is a throwaway spike for STRATA-2 (SIMD backend
// decision). It is NOT part of strata's compute path and nothing outside
// this directory may import it; delete it once docs/adr/0001-simd-backend.md
// has been acted on.
//
// It implements Add, Clamp and a 3×3 Horn slope-magnitude row kernel (the
// shape of DESIGN.md §14) four ways so they can be compared side by side:
//
//   - scalar.go            canonical scalar reference
//   - asm_amd64.{go,s}     hand-written AVX2 Plan 9 assembly (slope row only;
//     the Add and Clamp asm numbers in the ADR came from internal/vec's
//     former asm backend, which has since been removed)
//   - archsimd_amd64.go    Go 1.27 simd/archsimd, fixed-width Float32x8
//   - portable.go          Go 1.27 simd, width-agnostic Float32s
//
// The simd variants require GOEXPERIMENT=simd:
//
//	GOEXPERIMENT=simd go test -bench . ./internal/spike/simdbackend
package simdbackend
