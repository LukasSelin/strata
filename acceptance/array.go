package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/LukasSelin/strata/array"
	"github.com/LukasSelin/strata/raster"
)

// The N-dimensional array cases (DESIGN.md §10): a masked [time, y, x]
// float32 stack and an int16 label stack, put through broadcasting
// arithmetic and reductions along every combination of axes, some of
// them through views (a slice, a transpose, a broadcast), and written to
// array.json for check_array.py, which judges them with numpy
// broadcasting and exact integer arithmetic.
//
// x is 300 long, more than the reductions' block of 256 outputs, so a
// block boundary falls inside every row.

const (
	arrT = 5
	arrY = 23
	arrX = 300
)

// arrayInput is a named input, written whole.
type arrayInput struct {
	Name  string `json:"name"`
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
	File  string `json:"file"`
	Mask  string `json:"mask,omitempty"`
}

// arrayOperand is an input as an operation saw it: the named input with
// view steps applied in order, each [name, args...]: ["slice", axis,
// start, end], ["select", axis, i], ["transpose", perm...],
// ["broadcast", shape...] or ["expand", axis].
type arrayOperand struct {
	Input string  `json:"input"`
	View  [][]any `json:"view,omitempty"`
}

type arrayCase struct {
	Name   string         `json:"name"`
	Op     string         `json:"op"`
	Inputs []arrayOperand `json:"inputs"`
	Axes   []int          `json:"axes,omitempty"`
	DType  string         `json:"dtype"`
	Shape  []int          `json:"shape"`
	Out    string         `json:"out"`
	Mask   string         `json:"mask,omitempty"`
}

type arrayManifest struct {
	Inputs []arrayInput `json:"inputs"`
	Cases  []arrayCase  `json:"cases"`
}

func runArray() error {
	m := arrayManifest{}
	stack := arrayStack()
	layer := arrayLayer()
	weights := array.Wrap([]float32{1, -0.5, 3.25, 0, 1e-3}, arrT, 1, 1)
	labels := arrayLabels()
	offsets := array.New[int16](arrX)
	for i := range offsets.Data {
		offsets.Data[i] = int16(i*977%65536 - 32768)
	}
	for _, in := range []struct {
		name string
		a    any
	}{{"stack", stack}, {"layer", layer}, {"weights", weights}, {"labels", labels}, {"offsets", offsets}} {
		e, err := writeInput(in.name, in.a)
		if err != nil {
			return err
		}
		m.Inputs = append(m.Inputs, e)
	}

	emit := func(c arrayCase, out any) error {
		c.Out = "array-" + c.Name + ".bin"
		switch o := out.(type) {
		case array.Array[float32]:
			c.DType, c.Shape = "float32", o.Shape
			c.Mask = maskName(c.Name, o.Valid != nil)
			return appendCase(&m, c, writeArray(o), writeArrayMask(o))
		case array.Array[float64]:
			c.DType, c.Shape = "float64", o.Shape
			c.Mask = maskName(c.Name, o.Valid != nil)
			return appendCase(&m, c, writeArray(o), writeArrayMask(o))
		case array.Array[int16]:
			c.DType, c.Shape = "int16", o.Shape
			c.Mask = maskName(c.Name, o.Valid != nil)
			return appendCase(&m, c, writeArray(o), writeArrayMask(o))
		case array.Array[int64]:
			c.DType, c.Shape = "int64", o.Shape
			c.Mask = maskName(c.Name, o.Valid != nil)
			return appendCase(&m, c, writeArray(o), writeArrayMask(o))
		}
		return fmt.Errorf("case %s: unexpected output %T", c.Name, out)
	}
	whole := func(name string) arrayOperand { return arrayOperand{Input: name} }

	// Elementwise, broadcasting.
	{
		d := array.NewMasked[float32](arrT, arrY, arrX)
		array.Sub(d, stack, layer) // [T,Y,X] − [Y,X]
		if err := emit(arrayCase{Name: "sub-layer", Op: "sub", Inputs: []arrayOperand{whole("stack"), whole("layer")}}, d); err != nil {
			return err
		}
	}
	{
		d := array.NewMasked[float32](arrT, arrY, arrX)
		array.Mul(d, stack, weights) // [T,Y,X] × [T,1,1]
		if err := emit(arrayCase{Name: "mul-weights", Op: "mul", Inputs: []arrayOperand{whole("stack"), whole("weights")}}, d); err != nil {
			return err
		}
	}
	{
		// A transposed view against a column of the layer, broadcast
		// along the new last axis: [X,Y,T] max [Y,1].
		tv := stack.Transpose(2, 1, 0)
		col := layer.Select(1, 7).ExpandDims(1)
		d := array.NewMasked[float32](arrX, arrY, arrT)
		array.Max(d, tv, col)
		if err := emit(arrayCase{Name: "max-transposed", Op: "max", Inputs: []arrayOperand{
			{Input: "stack", View: [][]any{{"transpose", 2, 1, 0}}},
			{Input: "layer", View: [][]any{{"select", 1, 7}, {"expand", 1}}},
		}}, d); err != nil {
			return err
		}
	}
	{
		// Written through a window of a larger destination, from a slice.
		big := array.NewMasked[float32](arrT, arrY+4, arrX)
		d := big.Window([]int{0, 2, 0}, []int{arrT, arrY - 3, arrX})
		array.Min(d, stack.Slice(1, 3, arrY), layer.Slice(0, 0, arrY-3))
		if err := emit(arrayCase{Name: "min-sliced", Op: "min", Inputs: []arrayOperand{
			{Input: "stack", View: [][]any{{"slice", 1, 3, arrY}}},
			{Input: "layer", View: [][]any{{"slice", 0, 0, arrY - 3}}},
		}}, d); err != nil {
			return err
		}
	}
	{
		d := array.NewMasked[int16](arrT, arrY, arrX)
		array.Add(d, labels, offsets) // wraps
		if err := emit(arrayCase{Name: "add-int16", Op: "add", Inputs: []arrayOperand{whole("labels"), whole("offsets")}}, d); err != nil {
			return err
		}
	}
	{
		d := array.NewMasked[float32](arrT, arrY, arrX)
		array.Convert(d, labels)
		if err := emit(arrayCase{Name: "convert-int16", Op: "convert", Inputs: []arrayOperand{whole("labels")}}, d); err != nil {
			return err
		}
	}

	// Reductions over each set of axes, of the stack and of views of it.
	reduceCases := []struct {
		name string
		op   arrayOperand
		src  array.Array[float32]
		axes []int
	}{
		{"time", whole("stack"), stack, []int{0}},
		{"x", whole("stack"), stack, []int{2}},
		{"space", whole("stack"), stack, []int{1, 2}},
		{"all", whole("stack"), stack, []int{0, 1, 2}},
		{"view-time", arrayOperand{Input: "stack", View: [][]any{{"slice", 2, 10, 290}, {"transpose", 1, 0, 2}}},
			stack.Slice(2, 10, 290).Transpose(1, 0, 2), []int{1}},
	}
	for _, rc := range reduceCases {
		out := reducedShape(rc.src.Shape, rc.axes)
		in := []arrayOperand{rc.op}
		sum := array.New[float64](out...)
		array.SumOver(sum, rc.src, rc.axes...)
		mean := array.NewMasked[float64](out...)
		array.MeanOver(mean, rc.src, rc.axes...)
		mn := array.NewMasked[float32](out...)
		array.MinOver(mn, rc.src, rc.axes...)
		mx := array.NewMasked[float32](out...)
		array.MaxOver(mx, rc.src, rc.axes...)
		n := array.New[int64](out...)
		array.CountOver(n, rc.src, rc.axes...)
		for _, o := range []struct {
			op  string
			out any
		}{{"sum", sum}, {"mean", mean}, {"min", mn}, {"max", mx}, {"count", n}} {
			if err := emit(arrayCase{Name: o.op + "-" + rc.name, Op: o.op, Inputs: in, Axes: rc.axes}, o.out); err != nil {
				return err
			}
		}
	}
	{
		// Integer reductions: exact sums and means of int16 labels.
		out := []int{arrY, arrX}
		sum := array.New[float64](out...)
		array.SumOver(sum, labels, 0)
		mean := array.NewMasked[float64](out...)
		array.MeanOver(mean, labels, 0)
		mn := array.NewMasked[int16](out...)
		array.MinOver(mn, labels, 0)
		for _, o := range []struct {
			op  string
			out any
		}{{"sum", sum}, {"mean", mean}, {"min", mn}} {
			if err := emit(arrayCase{Name: o.op + "-labels", Op: o.op, Inputs: []arrayOperand{whole("labels")}, Axes: []int{0}}, o.out); err != nil {
				return err
			}
		}
	}

	f, err := os.Create(filepath.Join(*dir, "array.json"))
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(m); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	fmt.Printf("wrote %d array cases to %s\n", len(m.Cases), *dir)
	return nil
}

func reducedShape(shape, axes []int) []int {
	var out []int
	for k, n := range shape {
		drop := false
		for _, a := range axes {
			drop = drop || a == k
		}
		if !drop {
			out = append(out, n)
		}
	}
	return out
}

// arrayStack is the float32 stack: smooth fields of very different
// magnitudes per time step, so that sums over time span many exponents,
// with pixels whose whole time series is NoData, a scattered 10% of
// NoData, and a few NaNs, infinities and signed zeros in valid cells.
func arrayStack() array.Array[float32] {
	a := array.NewMasked[float32](arrT, arrY, arrX)
	scale := []float64{1, 1e-6, 3e7, 0.001, 250}
	for t := range arrT {
		for y := range arrY {
			for x := range arrX {
				v := scale[t] * (math.Sin(float64(x)*0.07+float64(t)) + 0.3*math.Cos(float64(y)*0.4) + 1e-3*float64((x*31+y*17+t*7)%101))
				a.Set(float32(v), t, y, x)
				if (x*7+y*13+t*29)%10 == 0 || (y == 5 && x >= 100 && x < 110) {
					a.SetValid(false, t, y, x)
				}
			}
		}
	}
	a.Set(float32(math.NaN()), 1, 2, 3)
	a.Set(float32(math.Inf(1)), 2, 4, 9)
	a.Set(float32(math.Inf(-1)), 3, 4, 9) // both infinities in one series
	a.Set(float32(math.Inf(-1)), 0, 6, 250)
	a.Set(float32(math.Copysign(0, -1)), 0, 8, 8)
	a.Set(0, 1, 8, 8)
	for _, idx := range [][]int{{1, 2, 3}, {2, 4, 9}, {3, 4, 9}, {0, 6, 250}, {0, 8, 8}, {1, 8, 8}} {
		a.SetValid(true, idx...)
	}
	// A series of zeros, one of them -0: min -0, max +0, sum and mean +0.
	for t := range arrT {
		a.Set(0, t, 9, 9)
		a.SetValid(true, t, 9, 9)
	}
	a.Set(float32(math.Copysign(0, -1)), 2, 9, 9)
	return a
}

// arrayLayer is a masked [y, x] field to broadcast against the stack.
func arrayLayer() array.Array[float32] {
	a := array.NewMasked[float32](arrY, arrX)
	for y := range arrY {
		for x := range arrX {
			a.Set(float32(0.5*math.Sin(float64(x+y)*0.11)), y, x)
			if (x+2*y)%17 == 0 {
				a.SetValid(false, y, x)
			}
		}
	}
	return a
}

// arrayLabels is an int16 stack near the type's limits, so sums exceed
// int16 and additions wrap.
func arrayLabels() array.Array[int16] {
	a := array.NewMasked[int16](arrT, arrY, arrX)
	for i := range a.Data {
		a.Data[i] = int16((i*7919)%65536 - 32768)
		if i%13 == 0 {
			raster.MaskSet(a.Valid, i, false)
		}
	}
	return a
}

func maskName(name string, masked bool) string {
	if !masked {
		return ""
	}
	return "array-" + name + ".mask.u8"
}

func appendCase(m *arrayManifest, c arrayCase, data []byte, mask []byte) error {
	if err := os.WriteFile(filepath.Join(*dir, c.Out), data, 0o644); err != nil {
		return err
	}
	if c.Mask != "" {
		if err := os.WriteFile(filepath.Join(*dir, c.Mask), mask, 0o644); err != nil {
			return err
		}
	}
	m.Cases = append(m.Cases, c)
	return nil
}

func writeInput(name string, a any) (arrayInput, error) {
	e := arrayInput{Name: name, File: "array-in-" + name + ".bin"}
	var data, mask []byte
	switch v := a.(type) {
	case array.Array[float32]:
		e.DType, e.Shape, data, mask = "float32", v.Shape, writeArray(v), writeArrayMask(v)
	case array.Array[int16]:
		e.DType, e.Shape, data, mask = "int16", v.Shape, writeArray(v), writeArrayMask(v)
	default:
		return e, fmt.Errorf("input %s: unexpected %T", name, a)
	}
	if err := os.WriteFile(filepath.Join(*dir, e.File), data, 0o644); err != nil {
		return e, err
	}
	if mask != nil {
		e.Mask = "array-in-" + name + ".mask.u8"
		if err := os.WriteFile(filepath.Join(*dir, e.Mask), mask, 0o644); err != nil {
			return e, err
		}
	}
	return e, nil
}

// each calls f with every index of shape in row-major order.
func each(shape []int, f func(idx []int)) {
	idx := make([]int, len(shape))
	for {
		f(idx)
		k := len(shape) - 1
		for ; k >= 0; k-- {
			idx[k]++
			if idx[k] < shape[k] {
				break
			}
			idx[k] = 0
		}
		if k < 0 {
			return
		}
	}
}

// writeArray returns a's elements in row-major order of its shape,
// little-endian, whatever its layout.
func writeArray[T array.Number](a array.Array[T]) []byte {
	out := make([]byte, 0, a.Len()*8)
	each(a.Shape, func(idx []int) {
		out, _ = binary.Append(out, binary.LittleEndian, a.At(idx...))
	})
	return out
}

// writeArrayMask returns one byte per element, 1 if valid, or nil if a
// has no mask.
func writeArrayMask[T array.Number](a array.Array[T]) []byte {
	if a.Valid == nil {
		return nil
	}
	out := make([]byte, 0, a.Len())
	each(a.Shape, func(idx []int) {
		b := byte(0)
		if a.IsValid(idx...) {
			b = 1
		}
		out = append(out, b)
	})
	return out
}
