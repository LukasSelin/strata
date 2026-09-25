package zarr

import (
	"context"
	"fmt"
	"math"

	"github.com/LukasSelin/strata/raster"
	zarrv3 "github.com/LukasSelin/zarr"
)

// number is the element types a Source reads.
type number interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float32 | float64
}

// loader returns the function that reads and decodes a chunk of a: the
// cells cells of its y-x plane from plane, as float32, and their validity
// if fill is set.
func loader(a *zarrv3.Array, plane, cells int, fill bool) (func(context.Context, []int) (*chunk, error), error) {
	switch d := a.DataType(); d {
	case zarrv3.Int8:
		return loaderOf[int8](a, plane, cells, fill), nil
	case zarrv3.Int16:
		return loaderOf[int16](a, plane, cells, fill), nil
	case zarrv3.Int32:
		return loaderOf[int32](a, plane, cells, fill), nil
	case zarrv3.Int64:
		return loaderOf[int64](a, plane, cells, fill), nil
	case zarrv3.Uint8:
		return loaderOf[uint8](a, plane, cells, fill), nil
	case zarrv3.Uint16:
		return loaderOf[uint16](a, plane, cells, fill), nil
	case zarrv3.Uint32:
		return loaderOf[uint32](a, plane, cells, fill), nil
	case zarrv3.Uint64:
		return loaderOf[uint64](a, plane, cells, fill), nil
	case zarrv3.Float32:
		return loaderOf[float32](a, plane, cells, fill), nil
	case zarrv3.Float64:
		return loaderOf[float64](a, plane, cells, fill), nil
	default:
		return nil, fmt.Errorf("zarr: array %q: data type %s is not supported", a.Path(), d)
	}
}

func loaderOf[T number](a *zarrv3.Array, plane, cells int, fill bool) func(context.Context, []int) (*chunk, error) {
	fv, _ := a.FillValue().(T) // the library parses it as the array's type
	return func(ctx context.Context, idx []int) (*chunk, error) {
		data, err := zarrv3.ReadChunk[T](ctx, a, idx)
		if err != nil {
			return nil, err
		}
		if len(data) < plane+cells {
			return nil, fmt.Errorf("chunk of %d elements, want at least %d", len(data), plane+cells)
		}
		c := &chunk{vals: make([]float32, cells)}
		src := data[plane : plane+cells]
		convert(c.vals, src)
		if fill {
			c.valid = validity(src, fv)
		}
		return c, nil
	}
}

// convert writes src into dst as float32, rounded to nearest as IEEE 754
// rounds, and ±Inf past float32's range.
func convert[T number](dst []float32, src []T) {
	switch s := any(src).(type) {
	case []float32:
		copy(dst, s)
	case []float64:
		for i, v := range s {
			dst[i] = toFloat32(v)
		}
	default:
		for i, v := range src {
			dst[i] = float32(v)
		}
	}
}

// overflow32 is the smallest float64 that rounds to +Inf as a float32:
// MaxFloat32 plus half its ulp, a tie that rounds to even, which is Inf.
const overflow32 = math.MaxFloat32 + 0x1p103

// toFloat32 rounds v to float32 as IEEE 754 does. Go leaves a conversion
// of an out-of-range value implementation-dependent, so the overflow is
// spelled out rather than left to the platform, as cog does.
func toFloat32(v float64) float32 {
	switch {
	case v >= overflow32:
		return float32(math.Inf(1))
	case v <= -overflow32:
		return float32(math.Inf(-1))
	}
	return float32(v)
}

// validity returns the mask of the elements of src that are not fill,
// compared in their own type, any NaN matching a NaN fill; or nil if
// none is fill.
func validity[T number](src []T, fill T) []uint64 {
	nan := math.IsNaN(float64(fill)) // exact: float64 holds every NaN as NaN
	var valid []uint64
	for i, v := range src {
		if v == fill || (nan && math.IsNaN(float64(v))) {
			if valid == nil {
				valid = raster.NewMask(len(src))
			}
			valid[i>>6] &^= 1 << (i & 63)
		}
	}
	return valid
}
