package exec

import (
	"fmt"

	"github.com/LukasSelin/strata/internal/vec"
	"github.com/LukasSelin/strata/raster"
)

// Pipeline is a chain of kernels run as one kernel (DESIGN.md §52). It
// satisfies Kernel, so every driver, tiling rule, halo, worker,
// cancellation path and counter keeps working: nothing above Kernel
// learns that a pipeline exists, and a chain that used to reload its
// tile once per operation now loads it once.
//
// Values are numbered in one sequence. Ids [0, Inputs) are the
// pipeline's own inputs, the views of its Window; each stage then
// appends its kernel's outputs in order. A stage may only name values
// already defined, so a Pipeline is a directed acyclic graph by
// construction and there is no cycle to look for. Stages run in
// declaration order, every one of them, whether or not its outputs are
// read.
//
// Everything between the stages lives in the scratch the engine lends
// (see ScratchKernel), except the pipeline's own output, which the last
// stage that produces it writes straight into the Span. So a chain of n
// stages allocates nothing per band and writes its result once.
//
// # Register-level fusion
//
// A pipeline whose stages are all FusableKernel and form a left-deep
// chain needs no scratch at all: it lowers to a vec.Chain, which
// evaluates every stage for a few cells before loading the next few, so
// the value between two stages is carried in a register rather than
// written to a buffer (DESIGN.md §29). That is what takes a chain from
// "loads its inputs once" to "touches nothing but its inputs and its
// output" — the buffers a staged pipeline keeps are span-sized, up to a
// whole L2, and the traffic to them is what §51's counter cannot see.
//
// Lowering is an optimisation and never a requirement. A pipeline off
// the cut — a stage that is not fusable, a shape that is not left-deep,
// an output that is not the last stage's — runs staged, which is the
// reference the fused form is tested against.
//
// # This is the radius-0 cut
//
// Every stage must have radius 0 and the pipeline must have exactly one
// output. §52 specifies the general form — stage k reads a window grown
// by the suffix sum of the radii after it, and scratch grows to match —
// but a stencil chain grows its halo quickly enough that the pointwise
// case is worth having on its own, and it is the whole of a weighted
// factor product. NewPipeline panics on anything outside this cut, so
// nothing silently takes a path that has not been written.
//
// # Validity
//
// A Pipeline computes none. Its stages write Data only, as every kernel
// does, and the engine derives the output's validity from the
// pipeline's declared inputs exactly as it would for any radius-0
// kernel: the AND of the masked ones.
//
// That is correct rather than convenient, and it is why NewPipeline
// requires every input to be reachable from the output. Run as separate
// calls, an intermediate's validity is the AND of its own inputs, so the
// result depends on the inputs the output actually reads. ANDing an
// input the output does not depend on would invalidate cells the unfused
// chain keeps. With every input reachable the two are the same AND, so
// the fused chain does one mask pass where the unfused one did n.
type Pipeline struct {
	// stages are the operations, in the order they run, with their
	// inputs named by value id.
	stages []Stage
	// inputs is how many values are the pipeline's own.
	inputs int
	// out is the value id the pipeline writes.
	out int

	// first[i] is the value id of stage i's first output.
	first []int
	// slot[id] is the scratch slot holding value id, or -1 for a value
	// that is an input or the output and so lives in a view the engine
	// supplied. Indexed by value id.
	slot []int
	// slots is how many scratch slots the values need, and maxIn and
	// maxOut the widest stage, which together size Scratch.
	slots, maxIn, maxOut int

	// chain is the fused form of the stages, or nil when they are off
	// the cut and Process runs them one at a time.
	chain *vec.Chain
}

// Stage is one operation of a Pipeline: a kernel, and where its inputs
// come from.
type Stage struct {
	// Kernel is the operation. It must have radius 0.
	Kernel Kernel
	// In names the kernel's inputs by value id, in the order its Window
	// expects them, and must have exactly the kernel's input count.
	In []int
}

// NewPipeline returns a Pipeline of inputs inputs that runs stages in
// order and writes value out.
//
// It panics on programming errors, as the rest of the engine does: no
// stages, a nil kernel, a kernel of non-zero radius, a stage whose In
// does not match its kernel's arity, a value id that is not defined
// before it is used, an output that is not produced by a stage, or an
// input the output cannot reach (see Pipeline for why that last one is
// a correctness rule and not a tidiness one).
//
// stages and their In slices are copied, so a caller may reuse them.
func NewPipeline(inputs int, stages []Stage, out int) *Pipeline {
	if inputs < 1 {
		panic(fmt.Sprintf("engine: pipeline has %d inputs; it needs at least one", inputs))
	}
	if len(stages) == 0 {
		panic("engine: pipeline has no stages")
	}

	p := &Pipeline{inputs: inputs, out: out}
	p.stages = make([]Stage, len(stages))
	p.first = make([]int, len(stages))

	next := inputs // the next value id a stage's outputs take
	for i, st := range stages {
		if st.Kernel == nil {
			panic(fmt.Sprintf("engine: pipeline stage %d has a nil kernel", i))
		}
		if r := st.Kernel.Radius(); r != 0 {
			panic(fmt.Sprintf("engine: pipeline stage %d has radius %d; "+
				"this cut takes radius 0 only (DESIGN.md §52)", i, r))
		}
		nin, nout := st.Kernel.Arity()
		if len(st.In) != nin {
			panic(fmt.Sprintf("engine: pipeline stage %d names %d inputs, its kernel takes %d",
				i, len(st.In), nin))
		}
		for j, id := range st.In {
			if id < 0 || id >= next {
				panic(fmt.Sprintf("engine: pipeline stage %d input %d names value %d, "+
					"which is not defined before it (values 0 to %d are)", i, j, id, next-1))
			}
		}
		p.stages[i] = Stage{Kernel: st.Kernel, In: append([]int(nil), st.In...)}
		p.first[i] = next
		next += nout
		p.maxIn = max(p.maxIn, nin)
		p.maxOut = max(p.maxOut, nout)
	}

	if out < inputs || out >= next {
		panic(fmt.Sprintf("engine: pipeline writes value %d, which is not produced by a stage "+
			"(stages produce values %d to %d)", out, inputs, next-1))
	}
	p.checkReachable(next)

	// Every value a stage produces needs a scratch slot except the
	// output, which the stage writes into the Span directly.
	p.slot = make([]int, next)
	for id := range p.slot {
		p.slot[id] = -1
	}
	for id := inputs; id < next; id++ {
		if id == out {
			continue
		}
		p.slot[id] = p.slots
		p.slots++
	}
	if fusePipelines {
		p.chain = p.lower()
	}
	return p
}

// fusePipelines is whether NewPipeline tries to lower a pipeline to a
// vec.Chain. It is a variable so that tests can build the same pipeline
// both ways and compare them; nothing else turns it off.
var fusePipelines = true

// lower returns the vec.Chain that computes what the stages compute, or
// nil when they are off the cut register-level fusion takes: every stage
// a FusableKernel with one output, and the stages a left-deep chain —
// stage 0 starting from a pipeline input, every later stage taking the
// stage before it as its *first* operand, every second operand a
// pipeline input, and the last stage producing the output.
//
// Left-deep is the whole restriction, and it is what makes the running
// value a register: a chain with one live intermediate needs nowhere to
// put the others. A shape with two — (a·b)·(c·d) — is a legal pipeline
// and simply runs staged.
func (p *Pipeline) lower() *vec.Chain {
	steps := make([]vec.Step, len(p.stages))
	first := -1
	for i := range p.stages {
		st := &p.stages[i]
		fk, ok := st.Kernel.(FusableKernel)
		if !ok {
			return nil
		}
		if _, nout := st.Kernel.Arity(); nout != 1 {
			return nil
		}
		step, fusable := fk.Fuse()
		if !fusable {
			return nil
		}
		want := 1 // the accumulator, plus one operand for a binary op
		if step.Op.Binary() {
			want = 2
		}
		if len(st.In) != want {
			panic(fmt.Sprintf("engine: pipeline stage %d fuses to %s, which takes %d inputs, "+
				"but its kernel takes %d", i, step.Op, want, len(st.In)))
		}
		// The accumulator: a pipeline input for stage 0, the stage
		// before for every other stage.
		acc := st.In[0]
		if i == 0 {
			if acc >= p.inputs {
				return nil
			}
			first = acc
		} else if acc != p.first[i-1] {
			return nil
		}
		step.Src = -1
		if step.Op.Binary() {
			if st.In[1] >= p.inputs {
				return nil
			}
			step.Src = st.In[1]
		}
		steps[i] = step
	}
	if p.first[len(p.stages)-1] != p.out {
		return nil // the output is not the last stage's
	}
	return vec.NewChain(p.inputs, first, steps)
}

// checkReachable panics unless every input is read, directly or through
// other stages, by the stage that produces the output. See Pipeline.
func (p *Pipeline) checkReachable(values int) {
	seen := make([]bool, values)
	seen[p.out] = true
	// Stages are in dependency order, so one backwards sweep is enough:
	// a stage's inputs are always lower ids than its outputs.
	for i := len(p.stages) - 1; i >= 0; i-- {
		used := false
		_, nout := p.stages[i].Kernel.Arity()
		for id := p.first[i]; id < p.first[i]+nout; id++ {
			used = used || seen[id]
		}
		if !used {
			continue
		}
		for _, id := range p.stages[i].In {
			seen[id] = true
		}
	}
	for id := range p.inputs {
		if !seen[id] {
			panic(fmt.Sprintf("engine: pipeline input %d cannot be reached from its output; "+
				"an unread input would still narrow the result's validity, so the fused chain "+
				"would not equal the unfused one (DESIGN.md §52)", id))
		}
	}
}

// Radius is 0: every stage of this cut is pointwise, so the chain is.
func (Pipeline) Radius() int { return 0 }

// Arity is the pipeline's inputs and its one output.
func (p *Pipeline) Arity() (inputs, outputs int) { return p.inputs, 1 }

// Fused reports whether the stages lowered to a fused chain (§29) rather
// than running one at a time. Lowering never changes what a pipeline
// computes, so this is for tests and benchmarks that mean to measure or
// pin one path; a caller building a pipeline needs no branch on it.
func (p *Pipeline) Fused() bool { return p.chain != nil }

// Scratch is the working memory one Process call needs: one w×h buffer
// per value that is neither an input nor the output, and views for the
// widest stage, reused down the chain.
//
// A fused pipeline asks for no cells at all — that is the claim §29
// makes — only the slice headers its operands are handed to vec.Chain
// in, which a band may not allocate for itself (DESIGN.md §26).
func (p *Pipeline) Scratch(w, h int) ScratchSize {
	if p.chain != nil {
		return ScratchSize{Runs: p.inputs}
	}
	return ScratchSize{Cells: p.slots * w * h, Views: p.maxIn + p.maxOut}
}

// Process runs the chain over the span: as one fused pass when the
// stages lowered, otherwise one stage at a time. See Kernel.
func (p *Pipeline) Process(dst Span, src Window) {
	if p.chain != nil {
		p.processFused(dst, src)
		return
	}
	w, h := dst.Width, dst.Height
	out := dst.Dst[0]
	cells := dst.Scratch.Cells
	in := dst.Scratch.Views[:p.maxIn]
	outs := dst.Scratch.Views[p.maxIn : p.maxIn+p.maxOut]

	for i := range p.stages {
		st := &p.stages[i]
		for j, id := range st.In {
			in[j] = p.view(id, w, h, src, out, cells)
		}
		_, nout := st.Kernel.Arity()
		for j := range nout {
			outs[j] = p.view(p.first[i]+j, w, h, src, out, cells)
		}
		st.Kernel.Process(
			Span{X: dst.X, Y: dst.Y, Width: w, Height: h, Dst: outs[:nout]},
			Window{Radius: 0, Src: in[:len(st.In)]},
		)
	}
}

// processFused runs the whole chain in one pass. Whole operands at once
// when every one of them is compact, as algebra's own functions do, and
// a row at a time otherwise: a windowed raster's row padding is its
// parent's cells, so a span may never be treated as one run of memory
// unless every view in it is one (DESIGN.md §21, internal/pointwise).
func (p *Pipeline) processFused(dst Span, src Window) {
	out := dst.Dst[0]
	runs := dst.Scratch.Runs[:p.inputs]

	whole := out.Stride == out.Width
	for i := range src.Src {
		whole = whole && src.Src[i].Stride == src.Src[i].Width
	}
	if whole {
		n := out.Width * out.Height
		for i := range src.Src {
			runs[i] = src.Src[i].Data[:n]
		}
		p.chain.Run(out.Data[:n], runs)
		return
	}
	for y := range out.Height {
		for i := range src.Src {
			runs[i] = src.Src[i].Row(y)
		}
		p.chain.Run(out.Row(y), runs)
	}
}

// view returns the w×h raster holding value id: one of the pipeline's
// own input views, its output view, or a compact scratch buffer. Scratch
// views carry no mask, because a kernel writes no validity and reads
// none (DESIGN.md §31).
func (p *Pipeline) view(id, w, h int, src Window, out raster.Float32Raster, cells []float32) raster.Float32Raster {
	switch {
	case id < p.inputs:
		return src.Src[id]
	case id == p.out:
		return out
	default:
		n := w * h
		off := p.slot[id] * n
		return raster.Float32Raster{
			Data: cells[off : off+n : off+n], Width: w, Height: h, Stride: w,
		}
	}
}
