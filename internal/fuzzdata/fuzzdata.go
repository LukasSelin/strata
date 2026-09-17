// Package fuzzdata decodes the bytes of a fuzz input into the values the
// module's fuzz tests build operands from: bounded integers, choices and
// float32 bit patterns biased towards the values kernels get wrong (NaN
// payloads, signed zeros, infinities, subnormals, extremes).
//
// A Reader never fails. Once its bytes run out it continues from a PCG
// seeded with a hash of the input, so a short input still describes a
// whole raster and every input decodes deterministically.
package fuzzdata

import (
	"encoding/binary"
	"hash/fnv"
	"math"
	"math/rand/v2"
)

// Reader hands out values decoded from a fuzz input.
type Reader struct {
	b   []byte
	rng *rand.Rand
}

// New returns a Reader over b.
func New(b []byte) *Reader {
	h := fnv.New64a()
	h.Write(b)
	s := h.Sum64()
	return &Reader{b: b, rng: rand.New(rand.NewPCG(s, s^0x9e3779b97f4a7c15))}
}

// Uint64 returns the next 8 bytes as a uint64.
func (r *Reader) Uint64() uint64 {
	if len(r.b) >= 8 {
		v := binary.LittleEndian.Uint64(r.b)
		r.b = r.b[8:]
		return v
	}
	return r.rng.Uint64()
}

// Byte returns the next byte.
func (r *Reader) Byte() byte {
	if len(r.b) > 0 {
		v := r.b[0]
		r.b = r.b[1:]
		return v
	}
	return byte(r.rng.Uint32())
}

// Bool returns the low bit of the next byte.
func (r *Reader) Bool() bool { return r.Byte()&1 != 0 }

// IntN returns a value in [0, n), from one byte when n <= 256 and eight
// otherwise. It panics if n <= 0.
func (r *Reader) IntN(n int) int {
	if n <= 0 {
		panic("fuzzdata: IntN needs n > 0")
	}
	if n <= 256 {
		return int(r.Byte()) % n
	}
	return int(r.Uint64() % uint64(n))
}

// Range returns a value in [lo, hi].
func (r *Reader) Range(lo, hi int) int { return lo + r.IntN(hi-lo+1) }

// Specials are the float32 values Float32 favours.
var Specials = []float32{
	float32(math.NaN()),
	math.Float32frombits(0x7fc0_0001), // quiet NaN with a payload
	math.Float32frombits(0xffc0_0000), // negative quiet NaN
	math.Float32frombits(0x7f80_0001), // signalling NaN
	float32(math.Inf(1)), float32(math.Inf(-1)),
	0, float32(math.Copysign(0, -1)),
	1, -1, 0.5, 255, 360, -9999,
	math.MaxFloat32, -math.MaxFloat32,
	math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
	math.Float32frombits(0x000f_ffff), // largest subnormal
	1 << 24, 1<<24 + 2,
}

// Float32 returns a special value, raw bits, or a small DEM-like value
// (multiples of 1/4 around 1000), chosen by the next byte.
func (r *Reader) Float32() float32 {
	switch sel := r.Byte(); {
	case sel < 48:
		return Specials[int(sel)%len(Specials)]
	case sel < 128:
		return math.Float32frombits(uint32(r.Uint64()))
	default:
		return 1000 + float32(int8(r.Byte()))/4
	}
}

// specials64 are float64 values past float32's range.
var specials64 = []float64{
	math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat64,
	1e300, 1e-300, 1e39, 1e-39, 1e-46, 1e20,
}

// Float64 is Float32 for float64: a special value (of either width), raw
// bits, or a small value.
func (r *Reader) Float64() float64 {
	switch sel := r.Byte(); {
	case sel < 24:
		return specials64[int(sel)%len(specials64)]
	case sel < 48:
		return float64(Specials[int(sel)%len(Specials)])
	case sel < 128:
		return math.Float64frombits(r.Uint64())
	default:
		return float64(int8(r.Byte())) / 4
	}
}
