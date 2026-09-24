package array

import (
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// opcode names an elementwise operation. The run loops switch on it once
// per run, so each case is a tight loop with no call per element.
type opcode int

const (
	opAdd opcode = iota
	opSub
	opMul
	opMin
	opMax
)

// Add computes dst = a + b elementwise, with a and b broadcast to dst's
// shape.
func Add[T Number](dst, a, b Array[T]) { binary("array.Add", opAdd, dst, a, b) }

// Sub computes dst = a - b elementwise, with a and b broadcast to dst's
// shape.
func Sub[T Number](dst, a, b Array[T]) { binary("array.Sub", opSub, dst, a, b) }

// Mul computes dst = a * b elementwise, with a and b broadcast to dst's
// shape.
func Mul[T Number](dst, a, b Array[T]) { binary("array.Mul", opMul, dst, a, b) }

// Min computes dst = min(a, b) elementwise with Go's builtin min, with a
// and b broadcast to dst's shape.
func Min[T Number](dst, a, b Array[T]) { binary("array.Min", opMin, dst, a, b) }

// Max computes dst = max(a, b) elementwise with Go's builtin max, with a
// and b broadcast to dst's shape.
func Max[T Number](dst, a, b Array[T]) { binary("array.Max", opMax, dst, a, b) }

// Copy writes src, broadcast to dst's shape, into dst, values and
// validity.
func Copy[T Number](dst, src Array[T]) { Convert(dst, src) }

// Convert writes src, broadcast to dst's shape, into dst with Go's
// conversion T(v), and copies validity. Converting to a narrower integer
// wraps; converting a float to an integer truncates toward zero, and is
// implementation-specific in Go when the value does not fit, NaN
// included, so convert only values the target type can hold.
func Convert[T, U Number](dst Array[T], src Array[U]) {
	const op = "array.Convert"
	dst.must(op, "dst")
	src.must(op, "src")
	requireWritable(op, dst)
	ss := requireBroadcast(op, "src", src, dst.Shape)
	requireMask(op, dst.Valid != nil, src.Valid != nil)
	requireApart(op, "src", dst, src, ss)
	l := newLoop(dst.Shape, true, dst.Stride, ss)
	for ; !l.done; l.next() {
		n, step := l.run()
		convertRun(dst.Data[l.off[0]:], step[0], src.Data[l.off[1]:], step[1], n)
	}
	validity(dst, operand{src.Valid, src.ValidOffset, ss})
}

// Fill sets every element of dst to v and marks it valid.
func Fill[T Number](dst Array[T], v T) {
	const op = "array.Fill"
	dst.must(op, "dst")
	requireWritable(op, dst)
	l := newLoop(dst.Shape, true, dst.Stride)
	for ; !l.done; l.next() {
		n, step := l.run()
		d := dst.Data[l.off[0]:]
		for i := range n {
			d[i*step[0]] = v
		}
	}
	validity(dst)
}

func binary[T Number](op string, code opcode, dst, a, b Array[T]) {
	dst.must(op, "dst")
	a.must(op, "a")
	b.must(op, "b")
	requireWritable(op, dst)
	as := requireBroadcast(op, "a", a, dst.Shape)
	bs := requireBroadcast(op, "b", b, dst.Shape)
	requireMask(op, dst.Valid != nil, a.Valid != nil || b.Valid != nil)
	requireApart(op, "a", dst, a, as)
	requireApart(op, "b", dst, b, bs)
	// Values first, then validity, as package algebra does.
	l := newLoop(dst.Shape, true, dst.Stride, as, bs)
	for ; !l.done; l.next() {
		n, step := l.run()
		binaryRun(code, dst.Data[l.off[0]:], step[0], a.Data[l.off[1]:], step[1], b.Data[l.off[2]:], step[2], n)
	}
	validity(dst, operand{a.Valid, a.ValidOffset, as}, operand{b.Valid, b.ValidOffset, bs})
}

// binaryRun computes n elements of d from a and b, stepping each by its
// own stride. Contiguous float32 runs go to the internal/vec kernels,
// which pick up the SIMD backend.
func binaryRun[T Number](code opcode, d []T, ds int, a []T, as int, b []T, bs int, n int) {
	if ds == 1 && as == 1 && bs == 1 {
		if d32, ok := any(d).([]float32); ok {
			a32, b32 := any(a).([]float32), any(b).([]float32)
			kernel32[code](d32[:n], a32[:n], b32[:n])
			return
		}
		d, a, b = d[:n], a[:n], b[:n]
		switch code {
		case opAdd:
			for i := range d {
				d[i] = a[i] + b[i]
			}
		case opSub:
			for i := range d {
				d[i] = a[i] - b[i]
			}
		case opMul:
			for i := range d {
				d[i] = a[i] * b[i]
			}
		case opMin:
			for i := range d {
				d[i] = min(a[i], b[i])
			}
		case opMax:
			for i := range d {
				d[i] = max(a[i], b[i])
			}
		}
		return
	}
	switch code {
	case opAdd:
		for i := range n {
			d[i*ds] = a[i*as] + b[i*bs]
		}
	case opSub:
		for i := range n {
			d[i*ds] = a[i*as] - b[i*bs]
		}
	case opMul:
		for i := range n {
			d[i*ds] = a[i*as] * b[i*bs]
		}
	case opMin:
		for i := range n {
			d[i*ds] = min(a[i*as], b[i*bs])
		}
	case opMax:
		for i := range n {
			d[i*ds] = max(a[i*as], b[i*bs])
		}
	}
}

// kernel32 holds the internal/vec kernel of each opcode.
var kernel32 = [...]func(dst, a, b []float32){
	opAdd: vec.Add,
	opSub: vec.Sub,
	opMul: vec.Mul,
	opMin: vec.Min,
	opMax: vec.Max,
}

func convertRun[T, U Number](d []T, ds int, s []U, ss int, n int) {
	if ds == 1 && ss == 1 {
		if d2, ok := any(d).([]U); ok {
			copy(d2[:n], s[:n])
			return
		}
		d, s = d[:n], s[:n]
		for i := range d {
			d[i] = T(s[i])
		}
		return
	}
	for i := range n {
		d[i*ds] = T(s[i*ss])
	}
}

// operand is an input's validity, with its strides broadcast to dst's
// shape.
type operand struct {
	valid  []uint64
	off    int
	stride []int
}

// validity writes dst's mask from its inputs': an element is valid iff
// it is valid in every input, where an input without a mask is valid
// everywhere. With no input masks dst's elements are marked valid, and
// if dst has no mask either nothing is done (DESIGN.md §31, rule 4).
func validity[T Number](dst Array[T], ins ...operand) {
	if dst.Valid == nil {
		return
	}
	masked := ins[:0:0]
	for _, in := range ins {
		if in.valid != nil {
			masked = append(masked, in)
		}
	}
	strides := [][]int{dst.Stride}
	for _, in := range masked {
		strides = append(strides, in.stride)
	}
	l := newLoop(dst.Shape, true, strides...)
	for ; !l.done; l.next() {
		n, step := l.run()
		d := dst.ValidOffset + l.off[0]
		if len(masked) == 0 {
			if step[0] == 1 {
				raster.MaskFillRange(dst.Valid, d, n, true)
				continue
			}
			for i := range n {
				raster.MaskSet(dst.Valid, d+i*step[0], true)
			}
			continue
		}
		if contiguous(step[:l.ops]) {
			in := masked[0]
			raster.MaskCopyRange(dst.Valid, d, in.valid, in.off+l.off[1], n)
			for j, in := range masked[1:] {
				raster.MaskAndRange(dst.Valid, d, dst.Valid, d, in.valid, in.off+l.off[j+2], n)
			}
			continue
		}
		for i := range n {
			v := true
			for j, in := range masked {
				v = v && raster.MaskGet(in.valid, in.off+l.off[j+1]+i*step[j+1])
			}
			raster.MaskSet(dst.Valid, d+i*step[0], v)
		}
	}
}

func contiguous(steps []int) bool {
	for _, s := range steps {
		if s != 1 {
			return false
		}
	}
	return true
}
