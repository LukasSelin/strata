package engine

import "fmt"

// bytesPerCell is the size of one float32 cell. Validity is not counted;
// see Stats.
const bytesPerCell = 4

// Stats records how many bytes a call moved, stage by stage, and what the
// same work would have cost a pipeline that touched every byte once
// (DESIGN.md §51, and §27–§28 for the limits it exists inside). Pass a
// pointer to one in Options to have a call
// report into it.
//
// It exists because throughput alone cannot say why a call is slow. The
// chunked path runs Slope, Hillshade and Clamp at the same 770–845 M
// cells/s although their compute costs differ fivefold
// (benchmarks/chunked/RESULTS.md), which is the signature of a fixed
// per-cell tax rather than of compute or IO. Reading that off three
// throughput figures that happen to coincide is an inference; Amplification
// makes it a number, and makes any change that claims to remove the tax
// falsifiable.
//
// # What is counted
//
// Each field is the Data a stage moved, in bytes, summed over every
// operand and every worker:
//
//	SourceRead     sources delivered into the workers' input buffers
//	KernelRead     kernels read from their input windows, halo included
//	KernelWritten  kernels wrote into their output spans, edge cells included
//	SinkWritten    sinks took from the workers' output buffers
//
// A Tiled or plain call has no buffers, so its sources and sinks are zero
// and the kernel reads and writes the rasters directly. A Chunked call
// moves every cell through a buffer, so it reports all four, and that is
// the difference the counters exist to show.
//
// # What is not counted
//
// Validity bits. A mask is one bit per cell against a float32's 32, so it
// is at most 3% of the traffic here, and counting it exactly through
// ErodeBox's scratch and the word-level range operations would thread
// bookkeeping into every branch of halo.go for a term smaller than the
// run-to-run variance of the benchmarks it would inform. Being exact about
// the 97% and explicit about the rest is worth more than being approximate
// about all of it. The mask's cost is measured directly instead, by
// running the same case with and without one (benchmarks/terrain/RESULTS.md).
//
// # Logical traffic, not DRAM traffic
//
// These are the bytes the engine moves between stages, not the bytes that
// reach memory: a tile buffer that stays in L2 is counted when the source
// fills it and again when the kernel reads it, though DRAM saw it once.
// That is the intended reading. How many times the engine handles each
// byte is what its structure decides and what fusion or a windowed tile
// would change; how much of that reaches DRAM is what a hardware profiler
// (AMD uProf, VTune, perf's uncore counters) measures. The two answer
// different questions and are most useful together: Amplification says how
// much traffic the design asks for, a profiler says how much of it the
// cache absorbed.
//
// # Accumulating
//
// A call adds to the Stats it is given rather than overwriting it, so one
// Stats can total a whole pipeline of calls of different arities — Ideal
// grows with the work, so Amplification stays meaningful across them. Use
// a fresh Stats, or zero it, to measure one call.
//
// A call writes its Stats once, after every worker has stopped, so no
// synchronisation is needed to read it afterwards. One Stats must not be
// given to two calls that run at the same time.
//
// A cancelled or failed call reports the work it actually did: Cells
// counts output cells written, not the raster's size, so the ratios stay
// meaningful for a partial run.
type Stats struct {
	// Cells is the number of output cells written, counted once however
	// many outputs the kernel has.
	Cells int64
	// Ideal is what the work counted here would cost a pipeline that read
	// each input cell once and wrote each output cell once: for a kernel
	// of nin inputs and nout outputs, Cells × 4 × (nin + nout). It is the
	// denominator of Amplification and accumulates with the rest, so it
	// stays right when one Stats totals calls of different arities.
	Ideal int64
	// Tiles and Bands are the units of work completed: tiles for a
	// Chunked call and 0 otherwise, bands for every call. A band is the
	// unit between cancellation checks; see the package documentation.
	Tiles int64
	Bands int64

	// SourceRead and SinkWritten are the bytes that crossed the
	// RasterSource and RasterSink interfaces. Both are 0 for a call that
	// is not Chunked.
	SourceRead  int64
	SinkWritten int64
	// KernelRead and KernelWritten are the bytes the kernels read from
	// their windows and wrote into their spans. KernelRead exceeds Cells ×
	// 4 × inputs by the halo, which neighbouring bands and tiles each read
	// again; KernelWritten is exactly Cells × 4 × outputs.
	KernelRead    int64
	KernelWritten int64
}

// Total is the bytes moved at every stage.
func (s Stats) Total() int64 {
	return s.SourceRead + s.KernelRead + s.KernelWritten + s.SinkWritten
}

// Amplification is Total over Ideal: how many times the call moved each
// byte that the work strictly needed. 1.0 is a pipeline that touches every
// byte once; a Chunked call that copies every cell into a buffer and out
// again is about 2.0. It returns 0 for an empty Stats.
//
// This is the number to watch. Fusion (DESIGN.md §29) lowers it by
// removing whole passes, a windowed tile by removing a copy, and a wider
// halo raises it. Throughput moves for all of those reasons and for
// several more.
func (s Stats) Amplification() float64 {
	if s.Ideal == 0 {
		return 0
	}
	return float64(s.Total()) / float64(s.Ideal)
}

// BytesPerCell is Total over Cells: the bytes moved for each output cell
// written. It returns 0 for an empty Stats.
func (s Stats) BytesPerCell() float64 {
	if s.Cells == 0 {
		return 0
	}
	return float64(s.Total()) / float64(s.Cells)
}

// Halo is the bytes of KernelRead that are halo: cells read by one band or
// tile that belong to another's span, and so are read more than once. It
// is 0 for a radius-0 kernel, and grows as tiles get smaller. It returns 0
// if inputs is 0.
//
// inputs is the kernel's input count, which Stats does not store because
// it accumulates across calls that may differ in arity.
func (s Stats) Halo(inputs int) int64 {
	if inputs <= 0 {
		return 0
	}
	return s.KernelRead - s.Cells*bytesPerCell*int64(inputs)
}

// Add totals another Stats into s.
func (s *Stats) Add(o Stats) {
	s.Cells += o.Cells
	s.Ideal += o.Ideal
	s.Tiles += o.Tiles
	s.Bands += o.Bands
	s.SourceRead += o.SourceRead
	s.KernelRead += o.KernelRead
	s.KernelWritten += o.KernelWritten
	s.SinkWritten += o.SinkWritten
}

// String is a one-line summary for benchmark output and test failures.
func (s Stats) String() string {
	return fmt.Sprintf("%d cells, %d tiles, %d bands, %.1f B/cell, %.2f× "+
		"(src %d, kr %d, kw %d, sink %d)",
		s.Cells, s.Tiles, s.Bands, s.BytesPerCell(), s.Amplification(),
		s.SourceRead, s.KernelRead, s.KernelWritten, s.SinkWritten)
}
