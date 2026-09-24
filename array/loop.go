package array

import "sort"

// maxOperands is how many arrays one loop walks together: a destination
// and two inputs.
const maxOperands = 3

// loop walks a shape as runs along its innermost dimension, carrying the
// Data offset of each operand at the start of every run. Operations call
// a kernel once per run, never per element (DESIGN.md §19).
//
// Before walking, length-1 dimensions are dropped and neighbouring
// dimensions merged wherever every operand steps through them as one, so
// a compact array, or any layout the operands share, is a single run.
type loop struct {
	shape   []int
	strides [maxOperands][]int
	ops     int

	idx  []int
	off  [maxOperands]int
	done bool
}

// newLoop plans a walk over shape for operands with the given strides,
// one slice per operand and each as long as shape. With reorder the
// dimensions are first sorted by the first operand's strides, largest
// outermost, which is only right for an elementwise operation: the
// visiting order changes, not what is computed.
func newLoop(shape []int, reorder bool, strides ...[]int) *loop {
	l := &loop{ops: len(strides)}
	dims := make([]int, 0, len(shape))
	for k, n := range shape {
		if n > 1 {
			dims = append(dims, k)
		}
	}
	if reorder {
		sort.SliceStable(dims, func(i, j int) bool { return strides[0][dims[i]] > strides[0][dims[j]] })
	}
	for _, k := range dims {
		last := len(l.shape) - 1
		if last >= 0 && l.mergeable(last, shape[k], strides, k) {
			l.shape[last] *= shape[k]
			for j := range l.ops {
				l.strides[j][last] = strides[j][k]
			}
			continue
		}
		l.shape = append(l.shape, shape[k])
		for j := range l.ops {
			l.strides[j] = append(l.strides[j], strides[j][k])
		}
	}
	if len(l.shape) == 0 {
		l.shape = []int{1}
		for j := range l.ops {
			l.strides[j] = []int{0}
		}
	}
	l.idx = make([]int, len(l.shape))
	return l
}

// mergeable reports whether dimension k of the source, of length n,
// continues the planned dimension last for every operand: each steps
// from the last element of one of last's rows to the next row's first
// exactly as it steps along k.
func (l *loop) mergeable(last, n int, strides [][]int, k int) bool {
	for j := range l.ops {
		if l.strides[j][last] != strides[j][k]*n {
			return false
		}
	}
	return true
}

// run returns the length of every run and each operand's step along it.
func (l *loop) run() (n int, step [maxOperands]int) {
	last := len(l.shape) - 1
	for j := range l.ops {
		step[j] = l.strides[j][last]
	}
	return l.shape[last], step
}

// restart returns to the first run.
func (l *loop) restart() {
	clear(l.idx)
	l.off = [maxOperands]int{}
	l.done = false
}

// next moves to the next run's offsets, setting done after the last.
func (l *loop) next() {
	for k := len(l.shape) - 2; k >= 0; k-- {
		l.idx[k]++
		for j := range l.ops {
			l.off[j] += l.strides[j][k]
		}
		if l.idx[k] < l.shape[k] {
			return
		}
		for j := range l.ops {
			l.off[j] -= l.strides[j][k] * l.shape[k]
		}
		l.idx[k] = 0
	}
	l.done = true
}
