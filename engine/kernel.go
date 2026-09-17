package engine

import "strata/raster"

// Kernel is a raster operation the engine can run over any part of a
// raster: a pointwise operation (radius 0) or a neighbourhood operation
// (radius r reads the (2r+1)×(2r+1) cells around each output cell).
//
// The engine decides which cells a call covers, in what order, and
// supplies each call's halo; the kernel only fills the span it is given.
// So that splitting a raster cannot change the result, a kernel must
// compute each output cell from the input cells of its neighbourhood (and
// its own parameters and the cell position) alone. Implementations must
// be safe for concurrent calls on disjoint spans: no mutable state, no
// goroutines (DESIGN.md §26), and no references kept to the span or window
// after Process returns.
type Kernel interface {
	// Radius is how many cells beyond an output cell the kernel reads in
	// each direction: 0 for a pointwise kernel, 1 for a 3×3 stencil.
	Radius() int
	// Arity is the number of input and output rasters Process expects.
	// A kernel has at least one output.
	Arity() (inputs, outputs int)
	// Process writes the Data of every cell of every output view in dst,
	// reading src. It must not write validity bits, which belong to the
	// engine, or anything outside the dst views.
	Process(dst Span, src Window)
}

// EdgeKernel is a Kernel with radius > 0 that chooses the Data value the
// engine writes into output cells whose neighbourhood extends past the
// edge of the rasters passed to Process. Without it the edge value is NaN.
// Edge cells are invalid in outputs that have a mask either way.
type EdgeKernel interface {
	Kernel
	Edge() float32
}

// Span is the rectangle of output cells one Process call writes.
type Span struct {
	// X and Y locate the span's top-left cell in the output rasters, for
	// kernels whose result depends on position.
	X, Y int
	// Width and Height are the span's size in cells.
	Width, Height int
	// Dst holds one Width×Height view per output, in the order given to
	// ProcessN. The views share the outputs' memory and keep their
	// strides, so Row(y) and Stride == Width (a compact view, one
	// contiguous run of cells) work as on any raster.
	Dst []raster.Float32Raster
}

// Window is the input neighbourhood of a span.
type Window struct {
	// Radius is the kernel's radius.
	Radius int
	// Src holds one (Width+2·Radius)×(Height+2·Radius) view per input,
	// in the order given to ProcessN. Cell (x+Radius, y+Radius) of a view
	// is under span cell (x, y), so for radius 1 output row y reads view
	// rows y, y+1 and y+2 and output column x reads view columns x to x+2.
	// Every view cell exists: the engine never calls a kernel for an
	// output cell whose neighbourhood leaves the input.
	Src []raster.Float32Raster
}
