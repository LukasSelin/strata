package kern

import (
	"math"
	"math/rand/v2"
	"testing"
)

// The kernels on one 512-sample row, the width of a COG block, on each
// backend the build and CPU have.

func backends(b *testing.B, run func(b *testing.B)) {
	for _, scalar := range []bool{true, false} {
		UseScalar(scalar)
		b.Run("backend="+Backend(), run)
		if Backend() == "scalar" && !scalar {
			break // no SIMD set: the scalar run was the only one
		}
	}
	UseScalar(false)
}

func BenchmarkPlanesRow(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	src, row, vals := make([]byte, 4*512), make([]byte, 4*512), make([]float32, 512)
	for i := range src {
		src[i] = byte(rng.Uint32())
	}
	backends(b, func(b *testing.B) {
		b.SetBytes(int64(len(row)))
		for b.Loop() {
			copy(row, src)
			PlanesRow(vals, row)
		}
	})
}

func BenchmarkSumBytes(b *testing.B) {
	rng := rand.New(rand.NewPCG(5, 6))
	src, row := make([]byte, 8*512), make([]byte, 8*512)
	for i := range src {
		src[i] = byte(rng.Uint32())
	}
	backends(b, func(b *testing.B) {
		b.SetBytes(int64(len(row)))
		for b.Loop() {
			copy(row, src)
			SumBytes(row)
		}
	})
}

func BenchmarkUint8RowPredicted(b *testing.B) {
	rng := rand.New(rand.NewPCG(7, 8))
	src, row, vals := make([]byte, 512), make([]byte, 512), make([]float32, 512)
	for i := range src {
		src[i] = byte(rng.Uint32())
	}
	backends(b, func(b *testing.B) {
		b.SetBytes(int64(len(row)))
		for b.Loop() {
			copy(row, src)
			Uint8Row(vals, row, false, true)
		}
	})
}

func BenchmarkUint16RowPredicted(b *testing.B) {
	rng := rand.New(rand.NewPCG(3, 4))
	src, row, vals := make([]byte, 2*512), make([]byte, 2*512), make([]float32, 512)
	for i := range src {
		src[i] = byte(rng.Uint32())
	}
	backends(b, func(b *testing.B) {
		b.SetBytes(int64(len(row)))
		for b.Loop() {
			copy(row, src)
			Uint16Row(vals, row, false, true)
		}
	})
}

func BenchmarkCopyRow(b *testing.B) {
	row, vals := make([]byte, 4*512), make([]float32, 512)
	backends(b, func(b *testing.B) {
		b.SetBytes(int64(len(row)))
		for b.Loop() {
			CopyRow(vals, row)
		}
	})
}

func BenchmarkWordRange(b *testing.B) {
	chunk := make([]float32, 64)
	for i := range chunk {
		chunk[i] = float32(i)
	}
	nd := math.Float32bits(65535)
	backends(b, func(b *testing.B) {
		b.SetBytes(4 * 64)
		for b.Loop() {
			_ = Word(chunk, ModeRange, 0, 0, nd-2, 4)
		}
	})
}
