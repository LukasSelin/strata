package graph

import (
	"fmt"
	"slices"
	"strings"

	"github.com/LukasSelin/strata/internal/exec"
)

// Boundary is how a plan carries a value across a pass boundary, when a
// later pass needs a value an earlier one computed: Normalize's input,
// which the later pass maps through the range the earlier one found
// (DESIGN.md §55).
type Boundary uint8

const (
	// BoundaryAuto recomputes the value when that reads at most one
	// stored raster and runs no neighbourhood operation, and stores it
	// otherwise. See DESIGN.md §55 for the costs behind the rule.
	BoundaryAuto Boundary = iota
	// BoundaryRecompute always computes the value again in the later
	// pass, from the inputs and stored values it was computed from: it
	// reads them again and stores nothing.
	BoundaryRecompute
	// BoundaryCache always stores the value in the pass that computes it
	// and reads it back in the later one: in memory, a temporary raster;
	// in a chunked run, a temporary raw file.
	BoundaryCache
)

func (b Boundary) String() string {
	switch b {
	case BoundaryAuto:
		return "auto"
	case BoundaryRecompute:
		return "recompute"
	case BoundaryCache:
		return "cache"
	}
	return fmt.Sprintf("Boundary(%d)", uint8(b))
}

// PlanOptions configures Plan.
type PlanOptions struct {
	// Boundary is how values cross pass boundaries. The zero value is
	// BoundaryAuto.
	Boundary Boundary
}

// Plan is a Graph cut into passes, each of which reads its inputs once,
// runs every operation it can on a tile while the tile is loaded, and
// writes only what is asked for or needed later (DESIGN.md §55). It is
// immutable, so one Plan can run any number of times, concurrently, over
// different data: the batch case, one workflow over many files.
type Plan struct {
	labels  []string // per value, for String
	inputOf map[int]string
	inputs  []string // the inputs the plan reads, in declaration order
	outputs []root
	stats   []root
	passes  []pass
	notes   []string
	opts    PlanOptions
}

type passKind uint8

const (
	// passFused is a Pipeline of stages over the pass's sources.
	passFused passKind = iota
	// passAlone is one kernel that cannot be a stage (a ScratchKernel),
	// run over stored values.
	passAlone
)

// pass is one run of the engine over the whole raster.
type pass struct {
	kind  passKind
	phase int
	// sources are the graph values the pass reads: inputs and stored
	// values, in value order. They are the pipeline's inputs.
	sources []int
	// stages are the fused pass's operations, in value order; a passAlone
	// has one.
	stages []stage
	// outs are the graph values the pass writes, in the order of the
	// kernel's outputs, and dests what becomes of each.
	outs  []int
	dests []dest
}

// stage is one operation of a pass.
type stage struct {
	// kernel is the operation, or nil for a Normalize, whose kernel is
	// built when the range of norm is known, and for a copy.
	kernel exec.Kernel
	// norm is the value a Normalize stage maps by its range, or -1.
	norm int
	// copy marks a stage that copies in[0] to an output: a value the
	// pipeline cannot write directly (DESIGN.md §55, "Lowering").
	copy bool
	// in are the graph values the stage reads, and first and nout the
	// values it produces (unused by a copy).
	in          []int
	first, nout int
	label       string
}

// dest is what becomes of one value a pass writes.
type dest struct {
	// outputs and stats name the caller's rasters or sinks and summaries.
	outputs []string
	stats   []string
	// rng marks a value whose range a Normalize in a later pass needs.
	rng bool
	// keep marks a value a later pass reads back.
	keep bool
}

func (d *dest) empty() bool { return len(d.outputs) == 0 && len(d.stats) == 0 && !d.rng && !d.keep }

// planner is the state of one Plan call.
type planner struct {
	g     *Graph
	nodes []node // the graph's nodes, with the gradient rewrite applied
	live  []bool // per node
	phase []int  // per node
	keep  []bool // per value
	// consumers counts, per value, the live nodes that read it.
	consumers []int
	dests     map[int]*dest
	notes     []string
	opts      PlanOptions
}

// Plan cuts g into passes. It panics if g asks for nothing: no Output and
// no Stats. The Plan does not change if g is built on afterwards.
func (g *Graph) Plan(opts PlanOptions) *Plan {
	if len(g.roots) == 0 {
		panic("graph: Plan of a graph with no outputs and no statistics")
	}
	if opts.Boundary > BoundaryCache {
		panic(fmt.Sprintf("graph: unknown %v", opts.Boundary))
	}
	pl := &planner{g: g, nodes: slices.Clone(g.nodes), opts: opts, dests: map[int]*dest{}}
	pl.markLive()
	pl.shareGradients()
	pl.markLive()
	pl.phases()
	p := &Plan{opts: opts, inputOf: map[int]string{}}
	p.labels = make([]string, len(g.values))
	for v := range g.values {
		p.labels[v] = pl.label(v)
	}
	for name, v := range g.inputs {
		p.inputOf[v] = name
	}
	for v := range g.values {
		if name, ok := p.inputOf[v]; ok && pl.live[g.values[v]] {
			p.inputs = append(p.inputs, name)
		}
	}
	for _, r := range g.roots {
		if r.stats {
			p.stats = append(p.stats, r)
		} else {
			p.outputs = append(p.outputs, r)
		}
		d := pl.dest(r.v)
		if r.stats {
			d.stats = append(d.stats, r.name)
		} else {
			d.outputs = append(d.outputs, r.name)
		}
	}
	p.passes = pl.passes()
	p.notes = pl.notes
	return p
}

func (pl *planner) dest(v int) *dest {
	d := pl.dests[v]
	if d == nil {
		d = &dest{}
		pl.dests[v] = d
	}
	return d
}

// markLive marks the nodes some root depends on, and counts each value's
// live consumers.
func (pl *planner) markLive() {
	g := pl.g
	pl.live = make([]bool, len(pl.nodes))
	pl.consumers = make([]int, len(g.values))
	for _, r := range g.roots {
		pl.live[g.values[r.v]] = true
	}
	for n := len(pl.nodes) - 1; n >= 0; n-- {
		if !pl.live[n] {
			continue
		}
		for _, v := range pl.nodes[n].in {
			pl.live[g.values[v]] = true
			pl.consumers[v]++
		}
	}
}

// shareGradients gives a terrain product its standalone kernel when it is
// the only reader of its gradient: then there is nothing to share, and
// the fused kernel computes the product without storing dx and dy
// between two stages. Products that share a gradient stay pointwise
// stages over it, as in terrain.Surface. Both forms write the same bits
// (DESIGN.md §52).
func (pl *planner) shareGradients() {
	g := pl.g
	roots := map[int]bool{}
	for _, r := range g.roots {
		roots[r.v] = true
	}
	shared := map[int]bool{}
	for n := range pl.nodes {
		nd := &pl.nodes[n]
		if !pl.live[n] || nd.alone == nil {
			continue
		}
		dx, dy := nd.in[0], nd.in[1]
		if pl.consumers[dx] != 1 || pl.consumers[dy] != 1 || roots[dx] || roots[dy] {
			if grad := g.values[dx]; !shared[grad] {
				shared[grad] = true
				pl.notes = append(pl.notes, fmt.Sprintf("the Horn gradient %s, %s is computed once for every product that reads it",
					pl.label(dx), pl.label(dy)))
			}
			continue
		}
		grad := pl.nodes[g.values[dx]]
		nd.in = slices.Clone(grad.in)
		nd.kernel = nd.alone
		nd.alone = nil
	}
}

// phases numbers the passes. A value's phase is the fused pass that
// computes it: a stage value's is the latest of its inputs', a pass of
// its own runs before the fused pass of its phase and reads only stored
// values, and a Normalize runs a phase after its input, whose range that
// input's pass finds.
func (pl *planner) phases() {
	g := pl.g
	pl.phase = make([]int, len(pl.nodes))
	pl.keep = make([]bool, len(g.values))
	for n := range pl.nodes {
		nd := &pl.nodes[n]
		if !pl.live[n] {
			continue
		}
		switch {
		case nd.kind == kindInput:
			pl.phase[n] = 0
		case nd.kind == kindNormalize:
			// The range of a pass of its own's output is known before
			// the fused pass of the same phase; any other value's only
			// after the fused pass that computes it.
			x := nd.in[0]
			pl.phase[n] = pl.phase[g.values[x]]
			if !pl.alone(g.values[x]) {
				pl.phase[n]++
			}
			pl.dest(x).rng = true
			pl.boundary(nd, x)
		case pl.alone(n):
			ph := 0
			for _, v := range nd.in {
				ph = max(ph, pl.ready(v))
			}
			pl.phase[n] = ph
		default:
			ph := 0
			for _, v := range nd.in {
				ph = max(ph, pl.phase[g.values[v]])
			}
			pl.phase[n] = ph
		}
	}
}

// alone reports whether node n must run as a pass of its own: its kernel
// asks for scratch, which a pipeline stage cannot have (DESIGN.md §52).
func (pl *planner) alone(n int) bool {
	_, ok := pl.nodes[n].kernel.(exec.ScratchKernel)
	return ok
}

// stored reports whether value v can be read, rather than computed, by
// any pass after the one that produces it: an input, the output of a
// pass of its own, or a value kept.
func (pl *planner) stored(v int) bool {
	n := pl.g.values[v]
	return pl.nodes[n].kind == kindInput || pl.alone(n) || pl.keep[v]
}

// ready returns the first phase whose passes can read v stored, and
// marks v kept if it must be stored for that.
func (pl *planner) ready(v int) int {
	n := pl.g.values[v]
	switch {
	case pl.nodes[n].kind == kindInput:
		return 0
	case pl.alone(n):
		return pl.phase[n]
	}
	if !pl.keep[v] {
		pl.keep[v] = true
		pl.notes = append(pl.notes, fmt.Sprintf("%s is stored: %s reads it and cannot be a pipeline stage",
			pl.label(v), pl.consumerLabel(v)))
	}
	return pl.phase[n] + 1
}

// boundary decides whether Normalize's input x is recomputed by the pass
// after its own or stored by its own pass for that one to read.
func (pl *planner) boundary(nd *node, x int) {
	if pl.stored(x) {
		pl.notes = append(pl.notes, fmt.Sprintf("%s = %s ends a pass at %s, which is read stored",
			pl.label(nd.first), nd.label, pl.label(x)))
		return
	}
	sources, radius := pl.recomputeCost(x)
	var cache bool
	var why string
	switch pl.opts.Boundary {
	case BoundaryRecompute:
		why = "PlanOptions.Boundary is recompute"
	case BoundaryCache:
		cache, why = true, "PlanOptions.Boundary is cache"
	default:
		cache = sources > 1 || radius
		switch {
		case sources > 1:
			why = fmt.Sprintf("recomputing it would read %d stored rasters again", sources)
		case radius:
			why = "recomputing it would run a neighbourhood operation again"
		default:
			why = "recomputing it reads one raster again and runs only per-cell operations"
		}
	}
	if cache {
		pl.keep[x] = true
		pl.notes = append(pl.notes, fmt.Sprintf("%s = %s ends a pass at %s: stored for the next pass (%s)", pl.label(nd.first), nd.label, pl.label(x), why))
	} else {
		pl.notes = append(pl.notes, fmt.Sprintf("%s = %s ends a pass at %s: recomputed in the next pass (%s)", pl.label(nd.first), nd.label, pl.label(x), why))
	}
}

// recomputeCost returns how many stored rasters computing v again reads,
// and whether that runs an operation of radius > 0.
func (pl *planner) recomputeCost(v int) (sources int, radius bool) {
	seen := map[int]bool{}
	var walk func(v int)
	walk = func(v int) {
		if seen[v] {
			return
		}
		seen[v] = true
		if pl.stored(v) {
			sources++
			return
		}
		nd := pl.nodes[pl.g.values[v]]
		if nd.kernel != nil && nd.kernel.Radius() > 0 {
			radius = true
		}
		for _, u := range nd.in {
			walk(u)
		}
	}
	walk(v)
	return sources, radius
}

// label names value v as a plan prints it: its number, "%3", which the
// stage that produces it spells out.
func (pl *planner) label(v int) string { return fmt.Sprintf("%%%d", v) }

func (pl *planner) consumerLabel(v int) string {
	for n := range pl.nodes {
		if pl.live[n] && slices.Contains(pl.nodes[n].in, v) && pl.alone(n) {
			return fmt.Sprintf("%s = %s", pl.label(pl.nodes[n].first), pl.nodes[n].label)
		}
	}
	return "a later pass"
}

// passes builds the passes, phase by phase: the passes of their own of a
// phase in value order, then its fused pass.
func (pl *planner) passes() []pass {
	g := pl.g
	last := 0
	for n := range pl.nodes {
		if pl.live[n] {
			last = max(last, pl.phase[n])
		}
	}
	// A value read stored by a later pass is written by the pass that
	// computes it. Values produced by passes of their own are stored
	// whenever anything reads them.
	for v := range g.values {
		n := g.values[v]
		if pl.live[n] && (pl.keep[v] || pl.alone(n) && pl.consumers[v] > 0) {
			pl.dest(v).keep = true
		}
	}
	var out []pass
	for ph := 0; ph <= last; ph++ {
		for n := range pl.nodes {
			if pl.live[n] && pl.phase[n] == ph && pl.alone(n) {
				out = append(out, pl.alonePass(n))
			}
		}
		if p, ok := pl.fusedPass(ph); ok {
			out = append(out, p)
		}
	}
	return out
}

func (pl *planner) alonePass(n int) pass {
	nd := pl.nodes[n]
	p := pass{kind: passAlone, phase: pl.phase[n], sources: slices.Clone(nd.in)}
	p.stages = []stage{{kernel: nd.kernel, norm: -1, in: nd.in, first: nd.first, nout: nd.nout, label: nd.label}}
	for v := nd.first; v < nd.first+nd.nout; v++ {
		p.outs = append(p.outs, v)
		d := pl.dests[v]
		if d == nil {
			d = &dest{}
		}
		p.dests = append(p.dests, *d)
	}
	return p
}

// fusedPass builds phase ph's pipeline: every value with a destination in
// this phase, and the stages they need back to values this pass can read
// stored. It reports false if the phase writes nothing.
func (pl *planner) fusedPass(ph int) (pass, bool) {
	g := pl.g
	var roots []int
	for v := range g.values {
		d := pl.dests[v]
		if d == nil || d.empty() || pl.alone(g.values[v]) {
			continue
		}
		n := g.values[v]
		if pl.nodes[n].kind == kindInput && ph == 0 || pl.nodes[n].kind != kindInput && pl.phase[n] == ph {
			roots = append(roots, v)
		}
	}
	if len(roots) == 0 {
		return pass{}, false
	}
	p := pass{kind: passFused, phase: ph}
	included := map[int]bool{}
	sources := map[int]bool{}
	var need func(v int)
	need = func(v int) {
		n := g.values[v]
		nd := pl.nodes[n]
		if nd.kind == kindInput || pl.alone(n) || pl.keep[v] && pl.phase[n] < ph {
			sources[v] = true
			return
		}
		if included[n] {
			return
		}
		included[n] = true
		for _, u := range nd.in {
			need(u)
		}
	}
	for _, v := range roots {
		need(v)
	}
	for v := range g.values {
		if sources[v] {
			p.sources = append(p.sources, v)
		}
	}
	// radiusReaders marks the nodes whose values a stage of radius > 0 in
	// this pass reads: the pipeline runs them over a span grown by that
	// radius, so they cannot also be written as outputs, and are copied.
	radiusReaders := map[int]bool{}
	for n := range pl.nodes {
		if !included[n] {
			continue
		}
		nd := pl.nodes[n]
		st := stage{kernel: nd.kernel, norm: -1, in: nd.in, first: nd.first, nout: nd.nout, label: nd.label}
		if nd.kind == kindNormalize {
			st.norm = nd.in[0]
		} else if nd.kernel.Radius() > 0 {
			for _, u := range nd.in {
				radiusReaders[g.values[u]] = true
			}
		}
		p.stages = append(p.stages, st)
	}
	for _, v := range roots {
		n := g.values[v]
		if sources[v] || radiusReaders[n] {
			why := "it is read by a neighbourhood operation in the same pass"
			if sources[v] {
				why = "the pass reads it rather than computing it"
			}
			pl.notes = append(pl.notes, fmt.Sprintf("pass %d copies %s to its destination: %s", ph+1, pl.label(v), why))
			p.stages = append(p.stages, stage{copy: true, norm: -1, in: []int{v}, nout: 1, label: "copy"})
		}
		p.outs = append(p.outs, v)
		p.dests = append(p.dests, *pl.dests[v])
	}
	return p, true
}

// String describes the plan: its passes, what each reads, runs and
// writes, and the decisions behind them.
func (p *Plan) String() string {
	var b strings.Builder
	reads := map[string]int{}
	for _, ps := range p.passes {
		for _, v := range ps.sources {
			if name, ok := p.inputOf[v]; ok {
				reads[name]++
			}
		}
	}
	fmt.Fprintf(&b, "plan: %d pass(es), boundary %v\n", len(p.passes), p.opts.Boundary)
	for _, name := range p.inputs {
		fmt.Fprintf(&b, "  input %q is read %d time(s)\n", name, reads[name])
	}
	for i, ps := range p.passes {
		kind := "fused"
		if ps.kind == passAlone {
			kind = "alone"
		}
		fmt.Fprintf(&b, "pass %d (%s, phase %d)\n", i+1, kind, ps.phase)
		for _, v := range ps.sources {
			if name, ok := p.inputOf[v]; ok {
				fmt.Fprintf(&b, "  read  %s = input %q\n", p.labels[v], name)
			} else {
				fmt.Fprintf(&b, "  read  %s, stored\n", p.labels[v])
			}
		}
		for _, st := range ps.stages {
			in := make([]string, len(st.in))
			for j, v := range st.in {
				in[j] = p.labels[v]
			}
			if st.copy {
				fmt.Fprintf(&b, "  run   copy of %s\n", in[0])
				continue
			}
			outs := make([]string, st.nout)
			for j := range outs {
				outs[j] = p.labels[st.first+j]
			}
			r := ""
			if st.kernel != nil && st.kernel.Radius() > 0 {
				r = fmt.Sprintf(", radius %d", st.kernel.Radius())
			}
			fmt.Fprintf(&b, "  run   %s = %s ← %s%s\n", strings.Join(outs, ", "), st.label, strings.Join(in, ", "), r)
		}
		for j, v := range ps.outs {
			d := ps.dests[j]
			var what []string
			for _, o := range d.outputs {
				what = append(what, fmt.Sprintf("output %q", o))
			}
			for _, s := range d.stats {
				what = append(what, fmt.Sprintf("stats %q", s))
			}
			if d.rng {
				what = append(what, "range for Normalize")
			}
			if d.keep {
				what = append(what, "stored")
			}
			if len(what) > 0 {
				fmt.Fprintf(&b, "  write %s → %s\n", p.labels[v], strings.Join(what, ", "))
			}
		}
	}
	for _, n := range p.notes {
		fmt.Fprintf(&b, "note: %s\n", n)
	}
	return b.String()
}
