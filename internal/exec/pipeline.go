package exec

import (
	"fmt"
	"math"

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
// declaration order, except that a stage no output depends on does not
// run at all: a kernel computes nothing but its outputs, so skipping it
// changes nothing, and running it could need a window wider than the
// pipeline's own.
//
// Everything between the stages lives in the scratch the engine lends
// (see ScratchKernel), except the pipeline's own outputs, which the
// stages that produce them write straight into the Span. So a chain of n
// stages allocates nothing per band and writes each result once.
//
// # Radius
//
// A stage with a radius reads the values before it beyond the cells it
// writes. So each stage runs over the span grown by how far beyond it
// the stages after it read — the sum of their radii, for a chain, and
// the largest such sum over every path in general — and the pipeline's
// radius is how far beyond the span it reads its own inputs. A stage's
// inputs are windows cut from the middle of the larger values before
// it. Radii add along a chain, so a pipeline does not make radius free:
// it stops a tile being reloaded, at the price of a wider halo.
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
// the cut — a stage that is not fusable, which includes every stage with
// a radius, a shape that is not left-deep, an output that is not the
// last stage's — runs staged, which is the reference the fused form is
// tested against.
//
// # Several outputs
//
// A pipeline writes any number of its values, each into its own view of
// the Span, and each output keeps what it would have computed alone:
// its own reach from each input, so its own validity, and its own edge
// ring, as wide as the largest of its reaches rather than the
// pipeline's radius. An output of radius 1 beside one of radius 3 gets
// real values in the two cells the wider ring would have taken. The
// engine does that for any ReachKernel (DESIGN.md §52): it runs the
// pipeline over the cells outside the narrowest ring, with windows
// padded with NaN where they leave the rasters. By the definition of
// reach no kept cell reads the padding.
//
// # What is not in the cut
//
// An output's stage must run over the span itself, so a value that is
// an output may not also be read beyond the span by a later stage, and
// neither may another output of the same stage: either would have to be
// computed grown into scratch and copied out. No stage may ask for
// scratch of its own. Register-level fusion takes one output only.
// NewPipeline panics on anything outside this cut, so nothing silently
// takes a path that has not been written.
//
// # Validity
//
// A Pipeline computes none. Its stages write Data only, as every kernel
// does, and the engine derives each output's validity from the
// pipeline's inputs: the AND, over the masked ones, of each input's
// validity eroded by its reach — the largest distance at which that
// output depends on it, or not at all if it does not (see ReachKernel). For a pointwise pipeline every
// reach is 0, and that is the AND of the masked inputs.
//
// That is what the stages compute when run as separate calls, and it is
// why the reach is per input rather than the pipeline's radius. Run as
// separate calls, Slope(a)·b is valid where a is valid over the 3×3
// around the cell and b at the cell itself; eroding b as well would
// invalidate cells the unfused chain keeps. Erosion by r contains
// erosion by any r' ≥ r, so the AND over every path from an input is
// its erosion by the longest, and one number per input is all validity
// needs. NewPipeline still requires every input to be read by some
// output, since a pipeline input nothing reads is a wiring mistake. So
// the fused chain does one mask pass per output where the unfused one
// did one per stage.
//
// # Edges
//
// The engine gives the cells in each output's edge ring the pipeline's
// edge value, as it does for any kernel, and that value is NaN. Run as separate calls, each stage leaves a ring
// of its own edge value, NaN, which the stages after it carry through
// their arithmetic; the two agree whenever those stages carry NaN
// through, as every arithmetic kernel in the tree does, and the validity
// of the ring agrees always. A stage that declares an edge value other
// than NaN would leave a number there that later stages turn into
// another, so NewPipeline accepts one only on a stage whose values are
// all outputs that no later stage reads, and only when its radius is
// each of those outputs' ring, so that its ring is the whole of theirs.
// The pipeline has one edge value, so every output's must agree
// (DESIGN.md §52).
type Pipeline struct {
	// stages are the operations, in the order they run, with their
	// inputs named by value id.
	stages []Stage
	// inputs is how many values are the pipeline's own.
	inputs int
	// outs are the value ids the pipeline writes, in the order of the
	// Span's views, and outOf[id] is value id's index in outs, or -1.
	outs  []int
	outOf []int
	// r is the pipeline's radius, and reach[o][in] how far beyond the
	// span output o reads input in, or -1.
	r     int
	reach [][]int
	// edge is the value the engine writes in the pipeline's edge ring.
	edge float32

	// first[i] is the value id of stage i's first output.
	first []int
	// live[i] is whether stage i runs: whether the output depends on one
	// of its outputs. grow[i] is how far beyond the span it runs.
	live []bool
	grow []int
	// held[id] is how far beyond the span value id is held: the
	// pipeline's radius for an input, 0 for an output, and its stage's
	// grow for any other value.
	held []int
	// slot[id] is the scratch slot holding value id, or -1 for a value
	// that is an input or an output and so lives in a view the engine
	// supplied, or that no running stage writes. Indexed by value id.
	slot []int
	// sum1[id] and sum2[id] are the sums of held and held² over the
	// slots before value id's, which with its slot place it in scratch
	// for any span size (see cellsBefore); tot1 and tot2 are the sums
	// over every slot.
	sum1, sum2 []int
	tot1, tot2 int
	// slots is how many scratch slots the values need, and maxIn and
	// maxOut the widest running stage, which together size Scratch.
	slots, maxIn, maxOut int

	// chain is the fused form of the stages, or nil when they are off
	// the cut and Process runs them one at a time.
	chain *vec.Chain
}

// Stage is one operation of a Pipeline: a kernel, and where its inputs
// come from.
type Stage struct {
	// Kernel is the operation. It must not be a ScratchKernel.
	Kernel Kernel
	// In names the kernel's inputs by value id, in the order its Window
	// expects them, and must have exactly the kernel's input count.
	In []int
}

// NewPipeline returns a Pipeline of inputs inputs that runs stages in
// order and writes the values outs, in that order.
//
// It panics on programming errors, as the rest of the engine does: no
// stages or no outputs, a nil kernel, a kernel that asks for scratch, a
// stage whose In does not match its kernel's arity, a value id that is
// not defined before it is used, an output that is not produced by a
// stage or is named twice, an output whose stage would have to run
// beyond the span, an input no output reads, or an edge value other than
// NaN that the pipeline cannot keep (see Pipeline).
//
// stages, their In slices and outs are copied, so a caller may reuse
// them.
func NewPipeline(inputs int, stages []Stage, outs []int) *Pipeline {
	if inputs < 1 {
		panic(fmt.Sprintf("engine: pipeline has %d inputs; it needs at least one", inputs))
	}
	if len(stages) == 0 {
		panic("engine: pipeline has no stages")
	}
	if len(outs) == 0 {
		panic("engine: pipeline has no outputs")
	}

	p := &Pipeline{inputs: inputs, outs: append([]int(nil), outs...), edge: float32(math.NaN())}
	p.stages = make([]Stage, len(stages))
	p.first = make([]int, len(stages))

	next := inputs // the next value id a stage's outputs take
	for i, st := range stages {
		if st.Kernel == nil {
			panic(fmt.Sprintf("engine: pipeline stage %d has a nil kernel", i))
		}
		if _, ok := st.Kernel.(ScratchKernel); ok {
			panic(fmt.Sprintf("engine: pipeline stage %d asks for scratch of its own, "+
				"which this cut does not lend (DESIGN.md §52)", i))
		}
		if r := st.Kernel.Radius(); r < 0 {
			panic(fmt.Sprintf("engine: pipeline stage %d has radius %d", i, r))
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
	}

	p.outOf = make([]int, next)
	for id := range p.outOf {
		p.outOf[id] = -1
	}
	for o, id := range p.outs {
		if id < inputs || id >= next {
			panic(fmt.Sprintf("engine: pipeline output %d is value %d, which is not produced by a stage "+
				"(stages produce values %d to %d)", o, id, inputs, next-1))
		}
		if p.outOf[id] >= 0 {
			panic(fmt.Sprintf("engine: pipeline outputs %d and %d are both value %d", p.outOf[id], o, id))
		}
		p.outOf[id] = o
	}
	p.plan(next)
	p.checkEdges()
	p.allocSlots(next)
	if fusePipelines && p.r == 0 && len(p.outs) == 1 {
		p.chain = p.lower()
	}
	return p
}

// fusePipelines is whether NewPipeline tries to lower a pipeline to a
// vec.Chain. It is a variable so that tests can build the same pipeline
// both ways and compare them; nothing else turns it off.
var fusePipelines = true

// plan works backwards from the outputs: which stages they depend on,
// how far beyond the span each must run, and how far beyond it each
// output reads each input. It panics unless every input is read and
// every output's stage runs over the span itself. See Pipeline.
func (p *Pipeline) plan(values int) {
	// need[id] is how far beyond the span value id is read, or -1 when
	// no output depends on it. Stages are in dependency order, so one
	// backwards sweep is enough: a stage's inputs are always lower ids
	// than its outputs.
	sweep := func(outs []int) (need []int, live []bool, grow []int) {
		need = make([]int, values)
		for id := range need {
			need[id] = -1
		}
		for _, id := range outs {
			need[id] = 0
		}
		live = make([]bool, len(p.stages))
		grow = make([]int, len(p.stages))
		for i := len(p.stages) - 1; i >= 0; i-- {
			st := &p.stages[i]
			_, nout := st.Kernel.Arity()
			g := -1
			for id := p.first[i]; id < p.first[i]+nout; id++ {
				g = max(g, need[id])
			}
			if g < 0 {
				continue
			}
			live[i], grow[i] = true, g
			for _, id := range st.In {
				need[id] = max(need[id], g+st.Kernel.Radius())
			}
		}
		return need, live, grow
	}

	need, live, grow := sweep(p.outs)
	p.live, p.grow = live, grow
	for id := range p.inputs {
		if need[id] < 0 {
			panic(fmt.Sprintf("engine: pipeline input %d is read by no output; "+
				"a pipeline input nothing reads is a wiring mistake (DESIGN.md §52)", id))
		}
		p.r = max(p.r, need[id])
	}
	for o, id := range p.outs {
		if i := p.stageOf(id); p.grow[i] > 0 {
			panic(fmt.Sprintf("engine: pipeline output %d (value %d) is written by stage %d, which a later "+
				"stage reads %d beyond the span; an output's stage must run over the span itself, "+
				"so neither it nor another output of its stage may feed a later stage with a radius "+
				"(DESIGN.md §52)", o, id, i, p.grow[i]))
		}
	}

	p.reach = make([][]int, len(p.outs))
	if len(p.outs) == 1 {
		p.reach[0] = need[:p.inputs:p.inputs]
		return
	}
	for o, id := range p.outs {
		n, _, _ := sweep([]int{id})
		p.reach[o] = n[:p.inputs:p.inputs]
	}
}

// stageOf returns the index of the stage that produces value id, which
// must not be an input.
func (p *Pipeline) stageOf(id int) int {
	i := len(p.first) - 1
	for p.first[i] > id {
		i--
	}
	return i
}

// ring is the width of output o's edge ring: its largest reach.
func (p *Pipeline) ring(o int) int {
	b := 0
	for _, n := range p.reach[o] {
		b = max(b, n)
	}
	return b
}

// checkEdges panics on a stage whose declared edge value the pipeline
// cannot reproduce, or on outputs whose edge values differ, and sets the
// pipeline's own. See Pipeline.
func (p *Pipeline) checkEdges() {
	// read[id] is whether a running stage reads value id.
	read := make([]bool, len(p.outOf))
	for i := range p.stages {
		if p.live[i] {
			for _, id := range p.stages[i].In {
				read[id] = true
			}
		}
	}
	edges := make([]float32, len(p.outs))
	for o := range edges {
		edges[o] = float32(math.NaN())
	}
	for i := range p.stages {
		k := p.stages[i].Kernel
		ek, ok := k.(EdgeKernel)
		if !ok || !p.live[i] || k.Radius() == 0 {
			continue
		}
		e := ek.Edge()
		if math.IsNaN(float64(e)) {
			continue
		}
		_, nout := k.Arity()
		for id := p.first[i]; id < p.first[i]+nout; id++ {
			o := p.outOf[id]
			if o < 0 || read[id] || k.Radius() != p.ring(o) {
				panic(fmt.Sprintf("engine: pipeline stage %d declares edge value %v, but its edge ring "+
					"is not its outputs' own, and the stages after it would compute with that value; "+
					"only a stage whose values are all outputs no later stage reads, with a radius equal "+
					"to their rings, may declare an edge value other than NaN (DESIGN.md §52)", i, e))
			}
			edges[o] = e
		}
	}
	p.edge = edges[0]
	for o, e := range edges {
		if math.Float32bits(e) != math.Float32bits(p.edge) && !(math.IsNaN(float64(e)) && math.IsNaN(float64(p.edge))) {
			panic(fmt.Sprintf("engine: pipeline outputs 0 and %d have edge values %v and %v; "+
				"a pipeline has one edge value (DESIGN.md §52)", o, p.edge, e))
		}
	}
}

// allocSlots gives every value a running stage writes, except the
// output, a scratch slot, and records where each lies.
func (p *Pipeline) allocSlots(values int) {
	p.held = make([]int, values)
	p.slot = make([]int, values)
	p.sum1 = make([]int, values)
	p.sum2 = make([]int, values)
	for id := range p.slot {
		p.slot[id] = -1
	}
	for id := range p.inputs {
		p.held[id] = p.r
	}
	for i := range p.stages {
		if !p.live[i] {
			continue
		}
		nin, nout := p.stages[i].Kernel.Arity()
		p.maxIn = max(p.maxIn, nin)
		p.maxOut = max(p.maxOut, nout)
		for id := p.first[i]; id < p.first[i]+nout; id++ {
			if p.outOf[id] >= 0 {
				continue // an output lives in the Span's view (see plan)
			}
			g := p.grow[i]
			p.held[id] = g
			p.slot[id], p.sum1[id], p.sum2[id] = p.slots, p.tot1, p.tot2
			p.slots++
			p.tot1 += g
			p.tot2 += g * g
		}
	}
}

// cellsBefore is how many scratch cells the first n slots take for a
// w×h span, given the sums of their held values g and of g²: a value
// held g beyond the span takes (w+2g)(h+2g) = wh + 2g(w+h) + 4g² cells.
func cellsBefore(n, sum1, sum2, w, h int) int {
	return n*w*h + 2*sum1*(w+h) + 4*sum2
}

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
	if p.first[len(p.stages)-1] != p.outs[0] {
		return nil // the output is not the last stage's
	}
	return vec.NewChain(p.inputs, first, steps)
}

// Radius is how far beyond the span the pipeline reads its inputs: for
// a chain, the sum of its stages' radii. See Pipeline.
func (p *Pipeline) Radius() int { return p.r }

// Arity is the pipeline's inputs and its outputs.
func (p *Pipeline) Arity() (inputs, outputs int) { return p.inputs, len(p.outs) }

// Reach is how far beyond the span output out reads input in, or -1 if
// it does not: what the engine erodes that input's validity by for that
// output, and the largest of them is the output's edge ring. See
// Pipeline.
func (p *Pipeline) Reach(out, in int) int { return p.reach[out][in] }

// Edge is the value the engine writes in the outputs' edge rings: NaN,
// unless the stages that write the outputs declare another. See
// Pipeline.
func (p *Pipeline) Edge() float32 { return p.edge }

// Fused reports whether the stages lowered to a fused chain (§29) rather
// than running one at a time. Lowering never changes what a pipeline
// computes, so this is for tests and benchmarks that mean to measure or
// pin one path; a caller building a pipeline needs no branch on it.
func (p *Pipeline) Fused() bool { return p.chain != nil }

// Scratch is the working memory one Process call needs: one buffer per
// value that is neither an input nor the output, each the span grown by
// how far beyond it that value is held, and views for the widest stage,
// reused down the chain.
//
// A fused pipeline asks for no cells at all — that is the claim §29
// makes — only the slice headers its operands are handed to vec.Chain
// in, which a band may not allocate for itself (DESIGN.md §26).
func (p *Pipeline) Scratch(w, h int) ScratchSize {
	if p.chain != nil {
		return ScratchSize{Runs: p.inputs}
	}
	return ScratchSize{Cells: cellsBefore(p.slots, p.tot1, p.tot2, w, h), Views: p.maxIn + p.maxOut}
}

// noScratch is every stage's Span.Scratch: stages ask for none, and a
// Span's Scratch is never nil.
var noScratch Scratch

// Process runs the chain over the span: as one fused pass when the
// stages lowered, otherwise one stage at a time, each over the span
// grown by how far beyond it the stages after it read. See Kernel.
func (p *Pipeline) Process(dst Span, src Window) {
	if p.chain != nil {
		p.processFused(dst, src)
		return
	}
	w, h := dst.Width, dst.Height
	cells := dst.Scratch.Cells
	in := dst.Scratch.Views[:p.maxIn]
	outs := dst.Scratch.Views[p.maxIn : p.maxIn+p.maxOut]

	for i := range p.stages {
		if !p.live[i] {
			continue
		}
		st := &p.stages[i]
		g, r := p.grow[i], st.Kernel.Radius()
		for j, id := range st.In {
			// The value is held at least g+r beyond the span; the stage's
			// window is the middle of it.
			v := p.view(id, w, h, src, dst.Dst, cells)
			if off := p.held[id] - g - r; off > 0 {
				v = v.Window(off, off, w+2*(g+r), h+2*(g+r))
			}
			in[j] = v
		}
		_, nout := st.Kernel.Arity()
		for j := range nout {
			outs[j] = p.view(p.first[i]+j, w, h, src, dst.Dst, cells)
		}
		st.Kernel.Process(
			Span{X: dst.X - g, Y: dst.Y - g, Width: w + 2*g, Height: h + 2*g,
				Dst: outs[:nout], Scratch: &noScratch},
			Window{Radius: r, Src: in[:len(st.In)]},
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

// view returns the raster holding value id for a w×h span, which is the
// span grown by held[id]: one of the pipeline's own input views, one of
// its output views, or a compact scratch buffer. Scratch views carry no
// mask, because a kernel writes no validity and reads none (DESIGN.md
// §31).
func (p *Pipeline) view(id, w, h int, src Window, dst []raster.Float32Raster, cells []float32) raster.Float32Raster {
	switch {
	case id < p.inputs:
		return src.Src[id]
	case p.outOf[id] >= 0:
		return dst[p.outOf[id]]
	default:
		g := p.held[id]
		vw, vh := w+2*g, h+2*g
		off := cellsBefore(p.slot[id], p.sum1[id], p.sum2[id], w, h)
		n := vw * vh
		return raster.Float32Raster{
			Data: cells[off : off+n : off+n], Width: vw, Height: vh, Stride: vw,
		}
	}
}
