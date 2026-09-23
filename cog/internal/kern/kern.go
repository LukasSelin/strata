// Package kern holds the cog reader's per-row decoding kernels: turning
// a block row's bytes into float32 samples, and testing samples against
// NoData into validity bits. They are span-level, as strata's own kernel
// packages are (DESIGN.md §12): slices and scalars in, nothing about
// rasters, files or blocks.
//
// Every kernel has a scalar form, which every build runs, and an AVX2
// form in GOEXPERIMENT=simd builds on CPUs that have it (simd_amd64.go),
// written with simd/archsimd as ADR 0001 requires, which is why they live
// under internal/ (DESIGN.md §14). The two give the same bits.
//
// Exported functions panic on mismatched lengths, with a "kern: " prefix.
//
//strata:kernel
package kern

import (
	"fmt"
	"math"
)

// The NoData tests Word makes, on the bits of float32 values.
const (
	// ModeExact: NoData where bits&care == want.
	ModeExact = 1 + iota
	// ModeRange: NoData where bits-lo, unsigned, is at most span.
	ModeRange
	// ModeNaN: NoData where the value is NaN.
	ModeNaN
)

// Backend function variables, swapped in init by SIMD builds.
var (
	planesRow = scalarPlanesRow
	uint16Row = scalarUint16Row
	word64    = scalarWord
	copyRow   = scalarCopyRow
)

// kernelSet is one backend's kernels.
type kernelSet struct {
	planesRow func(vals []float32, row []byte)
	uint16Row func(vals []float32, row []byte, signed, pred bool)
	word64    func(chunk []float32, mode int, want, care, lo uint32, span uint64) uint64
	copyRow   func(vals []float32, row []byte)
}

var scalarKernels = kernelSet{planesRow: scalarPlanesRow, uint16Row: scalarUint16Row, word64: scalarWord, copyRow: scalarCopyRow}

// simdKernels is the SIMD set, or nil when this build or CPU has none.
var (
	simdKernels *kernelSet
	simdName    string
	usingScalar bool
)

func (k *kernelSet) install() {
	planesRow, uint16Row, word64, copyRow = k.planesRow, k.uint16Row, k.word64, k.copyRow
}

// Backend returns the kernel set in use: "scalar", or the SIMD set's name.
func Backend() string {
	if simdKernels != nil && !usingScalar {
		return simdName
	}
	return "scalar"
}

// UseScalar selects the scalar kernels, or with false the SIMD ones if
// this build and CPU have them. It is for tests and benchmarks, and is
// not safe to call while kernels run.
func UseScalar(scalar bool) {
	usingScalar = scalar
	if scalar || simdKernels == nil {
		scalarKernels.install()
		return
	}
	simdKernels.install()
}

// PlanesRow writes one row of single-band float32 samples, stored with
// libtiff's floating-point predictor, to vals: row is the row's bytes,
// byte-differenced and split into four planes of len(vals) bytes, most
// significant first. It uses row as scratch.
func PlanesRow(vals []float32, row []byte) {
	if len(row) != 4*len(vals) {
		panic(fmt.Sprintf("kern: PlanesRow: %d bytes for %d samples", len(row), len(vals)))
	}
	planesRow(vals, row)
}

// Uint16Row writes one row of single-band little-endian 16-bit samples,
// signed or not, to vals, undoing horizontal differencing first if pred.
// It may use row as scratch.
func Uint16Row(vals []float32, row []byte, signed, pred bool) {
	if len(row) < 2*len(vals) {
		panic(fmt.Sprintf("kern: Uint16Row: %d bytes for %d samples", len(row), len(vals)))
	}
	uint16Row(vals, row[:2*len(vals)], signed, pred)
}

// CopyRow writes one row of single-band little-endian float32 samples,
// stored without a predictor, to vals, bit for bit.
func CopyRow(vals []float32, row []byte) {
	if len(row) < 4*len(vals) {
		panic(fmt.Sprintf("kern: CopyRow: %d bytes for %d samples", len(row), len(vals)))
	}
	copyRow(vals, row[:4*len(vals)])
}

// Word returns the validity bits of chunk, at most 64 cells, under the
// NoData test mode with its parameters: bit j is set where chunk[j] is
// not NoData.
func Word(chunk []float32, mode int, want, care, lo uint32, span uint64) uint64 {
	switch {
	case len(chunk) > 64:
		panic(fmt.Sprintf("kern: Word: %d cells", len(chunk)))
	case len(chunk) == 64:
		return word64(chunk, mode, want, care, lo, span)
	}
	return scalarWord(chunk, mode, want, care, lo, span)
}

func scalarCopyRow(vals []float32, row []byte) {
	row = row[:4*len(vals)]
	for i := range vals {
		vals[i] = math.Float32frombits(uint32(row[4*i]) | uint32(row[4*i+1])<<8 | uint32(row[4*i+2])<<16 | uint32(row[4*i+3])<<24)
	}
}

func scalarPlanesRow(vals []float32, row []byte) {
	var acc byte
	for i, b := range row {
		acc += b
		row[i] = acc
	}
	n := len(vals)
	p0, p1, p2, p3 := row[:n], row[n:2*n], row[2*n:3*n], row[3*n:4*n]
	for i := range vals {
		vals[i] = math.Float32frombits(uint32(p3[i]) | uint32(p2[i])<<8 | uint32(p1[i])<<16 | uint32(p0[i])<<24)
	}
}

func scalarUint16Row(vals []float32, row []byte, signed, pred bool) {
	row = row[:2*len(vals)]
	var acc uint16
	switch {
	case pred && signed:
		for i := range vals {
			acc += uint16(row[2*i]) | uint16(row[2*i+1])<<8
			vals[i] = float32(int16(acc)) // #nosec G115 -- reinterpreting the bits is the point
		}
	case pred:
		for i := range vals {
			acc += uint16(row[2*i]) | uint16(row[2*i+1])<<8
			vals[i] = float32(acc)
		}
	case signed:
		for i := range vals {
			vals[i] = float32(int16(uint16(row[2*i]) | uint16(row[2*i+1])<<8)) // #nosec G115 -- as above
		}
	default:
		for i := range vals {
			vals[i] = float32(uint16(row[2*i]) | uint16(row[2*i+1])<<8)
		}
	}
}

func scalarWord(chunk []float32, mode int, want, care, lo uint32, span uint64) uint64 {
	var word uint64
	switch mode {
	case ModeNaN:
		for j, v := range chunk {
			m := math.Float32bits(v) & 0x7fffffff
			word |= uint64((0x7f800000-m)>>31^1) << (uint(j) & 63)
		}
	case ModeRange:
		for j, v := range chunk {
			d := uint64(math.Float32bits(v) - lo)
			word |= (span - d) >> 63 << (uint(j) & 63)
		}
	default:
		for j, v := range chunk {
			d := (math.Float32bits(v) ^ want) & care
			word |= uint64((d|-d)>>31) << (uint(j) & 63)
		}
	}
	return word
}
