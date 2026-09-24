package graph

import (
	"context"
	"fmt"
	"os"
	"slices"
	"sort"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/opkernel"
	"github.com/LukasSelin/strata/internal/summary"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/reduce"
)

// Result is what a run returns besides its rasters.
type Result struct {
	// Stats holds the summary asked for under each Stats name.
	Stats map[string]reduce.Summary
}

// ChunkedOptions configures RunChunked.
type ChunkedOptions struct {
	// Engine is passed to every pass.
	Engine engine.Options
	// TempDir is where values stored between passes go, in a directory
	// of their own that the run removes. Empty means os.TempDir().
	TempDir string
}

// Run runs the plan over rasters in memory. in binds every input the
// plan reads, and out every output, by name; all must have the same
// dimensions. Every output is written with the bits the separate
// operations would write (DESIGN.md §55), and every rule of those
// operations applies: an output must have a mask if an input has one, and
// must not overlap an input or another output.
//
// A value stored between passes is a temporary raster the size of the
// inputs, or the output raster itself when the value is also an output.
// Statistics are folded from the value's raster after its pass.
//
// It returns ctx.Err() if ctx is done before the run finishes, and the
// outputs then hold some of their cells. It panics on programming
// errors: a missing or unknown name, and the operations' own checks.
func (p *Plan) Run(ctx context.Context, in, out map[string]raster.Float32Raster, opts engine.Options) (*Result, error) {
	p.bind(keys(in), keys(out))
	var w, h int
	masked := false
	store := map[int]raster.Float32Raster{}
	for v, name := range p.inputOf {
		r, ok := in[name]
		if !ok {
			continue
		}
		if w == 0 {
			w, h = r.Width, r.Height
		} else if r.Width != w || r.Height != h {
			panic(fmt.Sprintf("graph: input %q is %d×%d, other inputs are %d×%d", name, r.Width, r.Height, w, h))
		}
		masked = masked || r.Valid != nil
		store[v] = r
	}
	if w == 0 {
		panic("graph: Run needs at least one input")
	}
	res := &Result{Stats: map[string]reduce.Summary{}}
	ranges := map[int][2]float32{}
	temp := func() raster.Float32Raster {
		r := raster.NewFloat32(w, h, make([]float32, w*h))
		if masked {
			r.Valid = raster.NewMask(w * h)
		}
		return r
	}
	for i := range p.passes {
		ps := &p.passes[i]
		// A value the pass reads and would copy only to fold it is
		// already a raster here: fold that, and leave the copy out.
		reads := map[int]bool{}
		for _, v := range ps.sources {
			reads[v] = true
		}
		var outs, folds []int
		var dsts []raster.Float32Raster
		for j, v := range ps.outs {
			d := ps.dests[j]
			if reads[v] && len(d.outputs) == 0 {
				folds = append(folds, j)
				continue
			}
			dst := temp
			if len(d.outputs) > 0 {
				dst = func() raster.Float32Raster { return out[d.outputs[0]] }
			}
			outs = append(outs, j)
			dsts = append(dsts, dst())
		}
		if len(outs) > 0 {
			k, srcs := p.lower(ps, outs, ranges)
			rs := make([]raster.Float32Raster, len(srcs))
			for j, v := range srcs {
				rs[j] = store[v]
			}
			if err := exec.ProcessN(ctx, dsts, rs, k, opts); err != nil {
				return nil, err
			}
		}
		for n, j := range outs {
			v, d, dst := ps.outs[j], ps.dests[j], dsts[n]
			for _, name := range d.outputs[min(1, len(d.outputs)):] {
				if err := exec.ProcessN(ctx, []raster.Float32Raster{out[name]}, []raster.Float32Raster{dst}, copyKernel{}, opts); err != nil {
					return nil, err
				}
			}
			// A value is stored, and folded, from the raster it was
			// written to: the temporary, or the first output of that name.
			store[v] = dst
			folds = append(folds, j)
		}
		for _, j := range folds {
			v, d := ps.outs[j], ps.dests[j]
			if len(d.stats) == 0 && !d.rng {
				continue
			}
			part, err := exec.Reduce(ctx, []raster.Float32Raster{store[v]}, summary.Reducer{}, opts)
			if err != nil {
				return nil, err
			}
			p.record(res, ranges, v, d, part.Summary())
		}
	}
	return res, nil
}

// RunChunked runs the plan over sources and sinks, a tile at a time, in
// memory bounded by the tile size and worker count rather than the
// rasters. in binds every input the plan reads and out every output, by
// name; all must have the same size. The sinks receive the bits Run
// would write into rasters holding the sources' data.
//
// A value stored between passes goes to a temporary file under
// opts.TempDir, removed when the run returns. Statistics are folded from
// each tile as it is written, so a value asked for only as statistics is
// never written anywhere.
//
// It returns the first error of a source, a sink or the temporary
// files, or ctx.Err(); the sinks may then hold some of their tiles.
func (p *Plan) RunChunked(ctx context.Context, in map[string]engine.RasterSource, out map[string]engine.RasterSink, opts ChunkedOptions) (res *Result, err error) {
	p.bind(keys(in), keys(out))
	var w, h int
	store := map[int]engine.RasterSource{}
	for v, name := range p.inputOf {
		s, ok := in[name]
		if !ok {
			continue
		}
		sw, sh := s.Size()
		if w == 0 {
			w, h = sw, sh
		} else if sw != w || sh != h {
			panic(fmt.Sprintf("graph: input %q is %d×%d, other inputs are %d×%d", name, sw, sh, w, h))
		}
		store[v] = s
	}
	if w == 0 {
		panic("graph: RunChunked needs at least one input")
	}
	var dir string
	var spills []*spill
	defer func() {
		for _, s := range spills {
			if e := s.close(); err == nil && e != nil {
				err = e
			}
		}
		if dir != "" {
			if e := os.RemoveAll(dir); err == nil && e != nil {
				err = e
			}
		}
		if err != nil {
			res = nil
		}
	}()
	res = &Result{Stats: map[string]reduce.Summary{}}
	ranges := map[int][2]float32{}
	for i := range p.passes {
		ps := &p.passes[i]
		all := make([]int, len(ps.outs))
		for j := range all {
			all[j] = j
		}
		k, srcs := p.lower(ps, all, ranges)
		sources := make([]engine.RasterSource, len(srcs))
		masked := false
		for j, v := range srcs {
			sources[j] = store[v]
			masked = masked || sources[j].Masked()
		}
		sinks := make([]engine.RasterSink, len(ps.outs))
		tees := make([]*teeSink, len(ps.outs))
		for j, v := range ps.outs {
			d := ps.dests[j]
			t := &teeSink{w: w, h: h, masked: masked}
			for _, name := range d.outputs {
				s := out[name]
				if masked && !s.Masked() {
					panic(fmt.Sprintf("graph: output %q is not Masked, but the value it receives has validity", name))
				}
				t.sinks = append(t.sinks, s)
			}
			if d.keep {
				if dir == "" {
					if dir, err = tempDir(opts.TempDir); err != nil {
						return nil, err
					}
				}
				s, err := newSpill(dir, fmt.Sprintf("v%d", v), w, h, masked)
				if err != nil {
					return nil, err
				}
				spills = append(spills, s)
				t.sinks = append(t.sinks, s)
				store[v] = s
			}
			if len(d.stats) > 0 || d.rng {
				t.fold = &folder{}
			}
			sinks[j], tees[j] = t, t
		}
		if err := exec.ProcessChunked(ctx, sinks, sources, k, opts.Engine); err != nil {
			return nil, err
		}
		for j, t := range tees {
			if t.fold != nil {
				p.record(res, ranges, ps.outs[j], ps.dests[j], t.fold.summary())
			}
		}
	}
	return res, nil
}

// record keeps a value's summary under its Stats names, and its range if
// a Normalize needs it: Min and Max are what reduce.MinMax returns, the
// bounds algebra.Normalize maps by.
func (p *Plan) record(res *Result, ranges map[int][2]float32, v int, d dest, s summary.Summary) {
	for _, name := range d.stats {
		res.Stats[name] = reduce.Summary(s)
	}
	if d.rng {
		ranges[v] = [2]float32{s.Min, s.Max}
	}
}

// bind panics unless the names bound are exactly the inputs the plan
// reads and the outputs it writes.
func (p *Plan) bind(ins, outs []string) {
	want := slices.Clone(p.inputs)
	sort.Strings(want)
	if !slices.Equal(ins, want) {
		panic(fmt.Sprintf("graph: the plan reads inputs %q, but %q are bound", want, ins))
	}
	var wantOut []string
	for _, r := range p.outputs {
		wantOut = append(wantOut, r.name)
	}
	sort.Strings(wantOut)
	if !slices.Equal(outs, wantOut) {
		panic(fmt.Sprintf("graph: the plan writes outputs %q, but %q are bound", wantOut, outs))
	}
}

func keys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// lower builds the kernel that writes outputs outs (indices into ps.outs)
// of pass ps, and the graph values it reads, in the kernel's input order.
// A Normalize stage gets its kernel here, from the range an earlier pass
// recorded. Stages and sources that none of outs needs are left out, so
// the kernel reads nothing it does not use.
func (p *Plan) lower(ps *pass, outs []int, ranges map[int][2]float32) (exec.Kernel, []int) {
	if ps.kind == passAlone {
		return ps.stages[0].kernel, ps.sources
	}
	// Which stages and sources the wanted outputs need, walking back.
	produced := map[int]int{} // graph value -> index into ps.stages
	copies := map[int]int{}   // graph value -> its copy stage
	for i, st := range ps.stages {
		if st.copy {
			copies[st.in[0]] = i
			continue
		}
		for v := st.first; v < st.first+st.nout; v++ {
			produced[v] = i
		}
	}
	isSource := map[int]bool{}
	for _, v := range ps.sources {
		isSource[v] = true
	}
	needStage := make([]bool, len(ps.stages))
	needSource := map[int]bool{}
	var need func(v int)
	need = func(v int) {
		if isSource[v] {
			needSource[v] = true
			return
		}
		i := produced[v]
		if needStage[i] {
			return
		}
		needStage[i] = true
		for _, u := range ps.stages[i].in {
			need(u)
		}
	}
	for _, j := range outs {
		v := ps.outs[j]
		if i, ok := copies[v]; ok {
			needStage[i] = true
		}
		need(v)
	}
	var srcs []int
	id := map[int]int{}
	for _, v := range ps.sources {
		if needSource[v] {
			id[v] = len(srcs)
			srcs = append(srcs, v)
		}
	}
	next := len(srcs)
	var stages []exec.Stage
	copyID := map[int]int{}
	for i, st := range ps.stages {
		if !needStage[i] {
			continue
		}
		in := make([]int, len(st.in))
		for j, v := range st.in {
			in[j] = id[v]
		}
		k := st.kernel
		switch {
		case st.copy:
			k = copyKernel{}
			copyID[st.in[0]] = next
		case st.norm >= 0:
			k = opkernel.New("algebra.Normalize", ranges[st.norm])
		default:
			for v := st.first; v < st.first+st.nout; v++ {
				id[v] = next + v - st.first
			}
		}
		_, nout := k.Arity()
		stages = append(stages, exec.Stage{Kernel: k, In: in})
		if st.norm >= 0 {
			id[st.first] = next
		}
		next += nout
	}
	pouts := make([]int, len(outs))
	for n, j := range outs {
		v := ps.outs[j]
		if c, ok := copyID[v]; ok {
			pouts[n] = c
		} else {
			pouts[n] = id[v]
		}
	}
	return exec.NewPipeline(len(srcs), stages, pouts), srcs
}

// copyKernel writes its input unchanged: a value a pipeline cannot write
// directly, because it reads it rather than computing it or because a
// neighbourhood stage reads it over a grown span, and a value written
// under a second name.
type copyKernel struct{}

func (copyKernel) Radius() int                  { return 0 }
func (copyKernel) Arity() (inputs, outputs int) { return 1, 1 }

func (copyKernel) Process(dst exec.Span, src exec.Window) {
	d, s := dst.Dst[0], src.Src[0]
	for y := range dst.Height {
		copy(d.Row(y), s.Row(y))
	}
}
