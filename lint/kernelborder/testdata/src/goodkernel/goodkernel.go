// want package:"kernel"

// Package goodkernel keeps to every rule.
//
//strata:kernel
package goodkernel

import (
	"fmt"
	"math"

	"fakesimd/archsimd"
	"otherkernel"
)

type Region struct {
	Words  []uint64
	Stride int
}

type Number interface{ ~float32 | ~float64 }

var _ archsimd.Float32x8

func Sqrt(dst, src []float32) {
	if len(dst) != len(src) {
		panic(fmt.Sprintf("lengths %d, %d", len(dst), len(src)))
	}
	for i := range dst {
		dst[i] = float32(math.Sqrt(float64(src[i])))
	}
}

func Erode(dst Region, srcs []Region, w, h int, scratch *[4]uint64) {}

func Fold(acc *otherkernel.Acc, xs []float32) { acc.Add(xs) }

func Scale[T Number](dst []T, k T) {}

func Pick(f func(dst, src []float32)) func([]float32) float32 { return nil }

// unexported helpers may use anything.
func helper(m map[string]int, x any) {}

type hidden struct{}

// A method on an unexported type is not part of the API.
func (hidden) Exported(x any) {}
