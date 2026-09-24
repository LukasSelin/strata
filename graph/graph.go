package graph

import (
	"fmt"

	"reflect"
	"strings"

	"github.com/LukasSelin/strata/focal"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/opkernel"
	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/terrain"
)

// Graph is a lazy description of a raster workflow: named inputs,
// operations on them, and the outputs and statistics wanted. Building one
// computes nothing. Plan turns it into passes over the data, and the
// Plan runs them.
//
// A Graph is not safe for concurrent use while it is being built. Its
// Plans are independent of it and of each other.
type Graph struct {
	nodes []node
	// values maps a value id to the node that produces it: values are
	// numbered in one sequence, each node appending its outputs, as a
	// Pipeline numbers its own (DESIGN.md §52).
	values []int
	inputs map[string]int
	// cse maps a node's key to its index, so that building the same
	// operation on the same values twice returns the first node.
	cse   map[string]int
	roots []root
	names map[string]bool
}

// Node is one value of a Graph: an input, or one output of an operation.
// The zero Node is not a value, and operations on it panic.
type Node struct {
	g *Graph
	v int
}

// kind is how the planner treats a node (DESIGN.md §55).
type kind uint8

const (
	// kindInput is a named input: read from a raster or a source.
	kindInput kind = iota
	// kindKernel is an operation with a kernel: a pipeline stage, or,
	// for a ScratchKernel, a pass of its own.
	kindKernel
	// kindNormalize is algebra.Normalize: a global operation, whose
	// kernel needs the range of its input from an earlier pass.
	kindNormalize
)

type node struct {
	kind  kind
	label string
	in    []int
	first int
	nout  int
	// kernel is the node's kernel, for kindKernel.
	kernel exec.Kernel
	// alone is a kernel that computes the node from its gradient's DEM
	// directly: the standalone terrain kernel, for a product that reads
	// a Horn gradient. The planner uses it when nothing else reads that
	// gradient, so a lone Slope runs the kernel SlopeTiled runs.
	alone exec.Kernel
}

// root is a value the caller asked for: an output raster, or statistics.
type root struct {
	name  string
	v     int
	stats bool
}

// New returns an empty Graph.
func New() *Graph {
	return &Graph{inputs: map[string]int{}, cse: map[string]int{}, names: map[string]bool{}}
}

// Input returns the input named name, declaring it on first use. Every
// input a plan reads must be bound, by this name, when it runs.
func (g *Graph) Input(name string) Node {
	if name == "" {
		panic("graph: an input needs a name")
	}
	if v, ok := g.inputs[name]; ok {
		return Node{g, v}
	}
	g.nodes = append(g.nodes, node{kind: kindInput, label: fmt.Sprintf("input %q", name), first: len(g.values), nout: 1})
	g.values = append(g.values, len(g.nodes)-1)
	g.inputs[name] = len(g.values) - 1
	return Node{g, len(g.values) - 1}
}

// Output asks for n to be written to the raster or sink bound to name
// when the plan runs. A value may be written under several names.
func (g *Graph) Output(name string, n Node) { g.root(name, n, false) }

// Stats asks for the reduce.Summary of n's valid cells, returned under
// name when the plan runs. The planner folds it into the pass that
// computes n (DESIGN.md §55): asking for statistics of a value that is
// not otherwise written writes no raster at all.
func (g *Graph) Stats(name string, n Node) { g.root(name, n, true) }

func (g *Graph) root(name string, n Node, stats bool) {
	g.check("Output", n)
	if name == "" {
		panic("graph: an output needs a name")
	}
	// Outputs and statistics are bound and returned apart, so each has
	// its own names: an output and a summary may share one.
	key := fmt.Sprintf("%v/%s", stats, name)
	if g.names[key] {
		panic(fmt.Sprintf("graph: %q is already asked for", name))
	}
	g.names[key] = true
	g.roots = append(g.roots, root{name: name, v: n.v, stats: stats})
}

func (g *Graph) check(op string, ns ...Node) {
	for _, n := range ns {
		if n.g != g {
			if n.g == nil {
				panic(fmt.Sprintf("graph: %s of a zero Node", op))
			}
			panic(fmt.Sprintf("graph: %s of Nodes from different Graphs", op))
		}
	}
}

// add appends a node of kind k over ins, or returns the one already
// built with the same key, and returns its first value.
func add(op string, k kind, key string, kernel, alone exec.Kernel, nout int, ins ...Node) Node {
	if len(ins) == 0 || ins[0].g == nil {
		panic(fmt.Sprintf("graph: %s of a zero Node", op))
	}
	g := ins[0].g
	g.check(op, ins...)
	in := make([]int, len(ins))
	for i, n := range ins {
		in[i] = n.v
	}
	full := fmt.Sprintf("%s|%s|%v", op, key, in)
	if i, ok := g.cse[full]; ok {
		return Node{g, g.nodes[i].first}
	}
	label := op
	if key != "" {
		label += "(" + key + ")"
	}
	g.nodes = append(g.nodes, node{kind: k, label: label, in: in, first: len(g.values), nout: nout, kernel: kernel, alone: alone})
	for range nout {
		g.values = append(g.values, len(g.nodes)-1)
	}
	g.cse[full] = len(g.nodes) - 1
	return Node{g, len(g.values) - nout}
}

// kernelNode adds an operation whose kernel is registered as op.
func kernelNode(op string, opts any, key string, ins ...Node) Node {
	return namedKernelNode(op, op, opts, key, ins...)
}

// namedKernelNode adds the operation op, whose kernel is registered as
// reg: the binary algebra operations share one constructor.
func namedKernelNode(op, reg string, opts any, key string, ins ...Node) Node {
	k := opkernel.New(reg, opts)
	_, nout := k.Arity()
	return add(op, kindKernel, key, k, nil, nout, ins...)
}

// The algebra operations. Each is the operation of the same name in
// package algebra, with its validity: a cell is valid iff it is valid in
// every operand.

// Add is algebra.Add: a + b.
func Add(a, b Node) Node { return binary(vec.OpAdd, a, b) }

// Sub is algebra.Sub: a - b.
func Sub(a, b Node) Node { return binary(vec.OpSub, a, b) }

// Mul is algebra.Mul: a * b.
func Mul(a, b Node) Node { return binary(vec.OpMul, a, b) }

// Min is algebra.Min.
func Min(a, b Node) Node { return binary(vec.OpMin, a, b) }

// Max is algebra.Max.
func Max(a, b Node) Node { return binary(vec.OpMax, a, b) }

func binary(op vec.Op, a, b Node) Node {
	return namedKernelNode("algebra."+op.String(), "algebra.Binary", op, "", a, b)
}

// Clamp is algebra.Clamp: min(max(src, lo), hi).
func Clamp(src Node, lo, hi float32) Node {
	return kernelNode("algebra.Clamp", [2]float32{lo, hi}, fmt.Sprintf("%v, %v", lo, hi), src)
}

// Mask is algebra.Mask: src, valid where src and mask are both valid.
func Mask(src, mask Node) Node { return kernelNode("algebra.Mask", nil, "", src, mask) }

// Normalize is algebra.Normalize: src mapped onto [0, 1] by the smallest
// and largest of its valid cells. It is a global operation: no cell can
// be computed before every cell of src has been seen, so the planner ends
// a pass at src, folds the range into it, and computes the result in a
// later pass (DESIGN.md §55).
func Normalize(src Node) Node {
	return add("algebra.Normalize", kindNormalize, "", nil, nil, 1, src)
}

// The transfer operations.

// Reclass is transfer.Reclass. breaks and values are copied.
func Reclass(src Node, breaks, values []float32) Node {
	t := [2][]float32{append([]float32(nil), breaks...), append([]float32(nil), values...)}
	return kernelNode("transfer.Reclass", t, fmt.Sprintf("%v → %v", t[0], t[1]), src)
}

// Lookup is transfer.Lookup. xs and ys are copied.
func Lookup(src Node, xs, ys []float32) Node {
	t := [2][]float32{append([]float32(nil), xs...), append([]float32(nil), ys...)}
	return kernelNode("transfer.Lookup", t, fmt.Sprintf("%v → %v", t[0], t[1]), src)
}

// Rescale is transfer.Rescale: a·src + b.
func Rescale(src Node, a, b float32) Node {
	return kernelNode("transfer.Rescale", [2]float32{a, b}, fmt.Sprintf("%v, %v", a, b), src)
}

// RescaleRange is transfer.RescaleRange.
func RescaleRange(src Node, inLo, inHi, outLo, outHi float32) Node {
	return kernelNode("transfer.RescaleRange", [4]float32{inLo, inHi, outLo, outHi},
		fmt.Sprintf("%v..%v → %v..%v", inLo, inHi, outLo, outHi), src)
}

// The terrain operations. Slope, Aspect and Hillshade of one DEM with the
// same cell geometry share one Horn gradient, as terrain.Surface does, and
// write the bits of the standalone functions (DESIGN.md §52).

// Gradient is terrain.Gradient: the Horn gradient, dx and dy.
func Gradient(dem Node, opts terrain.GradientOptions) (dx, dy Node) {
	dx = kernelNode("terrain.Gradient", opts, brief(opts), dem)
	return dx, Node{dx.g, dx.v + 1}
}

// Slope is terrain.Slope.
func Slope(dem Node, opts terrain.SlopeOptions) Node {
	return product("terrain.Slope", opts, dem, terrain.GradientOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor})
}

// Aspect is terrain.Aspect.
func Aspect(dem Node, opts terrain.AspectOptions) Node {
	return product("terrain.Aspect", opts, dem, terrain.GradientOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor})
}

// Hillshade is terrain.Hillshade.
func Hillshade(dem Node, opts terrain.HillshadeOptions) Node {
	return product("terrain.Hillshade", opts, dem, terrain.GradientOptions{CellSize: opts.CellSize, CellSizeY: opts.CellSizeY, ZFactor: opts.ZFactor})
}

// product adds a terrain product as a pointwise stage over the shared
// gradient, keeping the standalone kernel for when it reads the gradient
// alone.
func product(op string, opts any, dem Node, gopts terrain.GradientOptions) Node {
	alone := opkernel.New(op, opts)
	dx, dy := Gradient(dem, gopts)
	k := opkernel.New(op+"FromGradient", opts)
	return add(op, kindKernel, brief(opts), k, alone, 1, dx, dy)
}

// Curvature is terrain.Curvature.
func Curvature(dem Node, opts terrain.CurvatureOptions) Node {
	return kernelNode("terrain.Curvature", opts, brief(opts), dem)
}

// Ruggedness is terrain.Ruggedness.
func Ruggedness(dem Node, opts terrain.RuggednessOptions) Node {
	return kernelNode("terrain.Ruggedness", opts, brief(opts), dem)
}

// The focal operations. Mean, Min, Max and CorrelateSeparable need working
// memory of their own, which a pipeline stage cannot have yet (DESIGN.md
// §52), so each runs as a pass of its own over a stored input; Correlate
// and Convolve fuse like any other stencil.

// FocalMean is focal.Mean.
func FocalMean(src Node, opts focal.BoxOptions) Node {
	return kernelNode("focal.Mean", opts, brief(opts), src)
}

// FocalMin is focal.Min.
func FocalMin(src Node, opts focal.BoxOptions) Node {
	return kernelNode("focal.Min", opts, brief(opts), src)
}

// FocalMax is focal.Max.
func FocalMax(src Node, opts focal.BoxOptions) Node {
	return kernelNode("focal.Max", opts, brief(opts), src)
}

// Correlate is focal.Correlate. The weights are copied.
func Correlate(src Node, opts focal.WeightsOptions) Node {
	opts.Weights = append([]float32(nil), opts.Weights...)
	return kernelNode("focal.Correlate", opts, brief(opts), src)
}

// Convolve is focal.Convolve. The weights are copied.
func Convolve(src Node, opts focal.WeightsOptions) Node {
	opts.Weights = append([]float32(nil), opts.Weights...)
	return kernelNode("focal.Convolve", opts, brief(opts), src)
}

// CorrelateSeparable is focal.CorrelateSeparable. The taps are copied.
func CorrelateSeparable(src Node, opts focal.SeparableOptions) Node {
	opts.Row = append([]float32(nil), opts.Row...)
	opts.Col = append([]float32(nil), opts.Col...)
	return kernelNode("focal.CorrelateSeparable", opts, brief(opts), src)
}

// brief describes an options struct by its fields that are not zero, as
// a plan prints it: "CellSize=10, Units=2".
func brief(opts any) string {
	v := reflect.ValueOf(opts)
	var parts []string
	for i := range v.NumField() {
		if f := v.Field(i); !f.IsZero() {
			parts = append(parts, fmt.Sprintf("%s=%v", v.Type().Field(i).Name, f.Interface()))
		}
	}
	return strings.Join(parts, ", ")
}
