package array_test

import (
	"testing"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/array"
)

// The benchmarks set a [16, 512, 512] float32 stack, 16 MiB, against
// package algebra over the same cells and against itself through other
// layouts, and time the reductions along a leading and a trailing axis.

const bt, by, bx = 16, 512, 512

func stack() array.Array[float32] {
	a := array.New[float32](bt, by, bx)
	for i := range a.Data {
		a.Data[i] = float32(i%1000) * 0.25
	}
	return a
}

func BenchmarkAdd(b *testing.B) {
	a := stack()
	dst := array.New[float32](bt, by, bx)
	b.Run("compact", func(b *testing.B) {
		b.SetBytes(int64(3 * 4 * a.Len()))
		for b.Loop() {
			array.Add(dst, a, a)
		}
	})
	b.Run("algebra", func(b *testing.B) {
		r := array.ToRaster(a.Reshape(bt*by, bx))
		d := array.ToRaster(dst.Reshape(bt*by, bx))
		b.SetBytes(int64(3 * 4 * a.Len()))
		for b.Loop() {
			algebra.Add(d, r, r)
		}
	})
	b.Run("broadcast", func(b *testing.B) {
		layer := a.Select(0, 0)
		b.SetBytes(int64(3 * 4 * a.Len()))
		for b.Loop() {
			array.Add(dst, a, layer)
		}
	})
	b.Run("strided", func(b *testing.B) {
		// Every input read across its rows: the scalar loop.
		t := a.Transpose(0, 2, 1)
		b.SetBytes(int64(3 * 4 * a.Len()))
		for b.Loop() {
			array.Add(dst, t, t)
		}
	})
}

func BenchmarkReduce(b *testing.B) {
	a := stack()
	for _, c := range []struct {
		name string
		axis int
		out  []int
	}{{"time", 0, []int{by, bx}}, {"x", 2, []int{bt, by}}} {
		b.Run("Sum/"+c.name, func(b *testing.B) {
			dst := array.New[float64](c.out...)
			b.SetBytes(int64(4 * a.Len()))
			for b.Loop() {
				array.SumOver(dst, a, c.axis)
			}
		})
		b.Run("Mean/"+c.name, func(b *testing.B) {
			dst := array.New[float64](c.out...)
			b.SetBytes(int64(4 * a.Len()))
			for b.Loop() {
				array.MeanOver(dst, a, c.axis)
			}
		})
		b.Run("Min/"+c.name, func(b *testing.B) {
			dst := array.New[float32](c.out...)
			b.SetBytes(int64(4 * a.Len()))
			for b.Loop() {
				array.MinOver(dst, a, c.axis)
			}
		})
	}
}
