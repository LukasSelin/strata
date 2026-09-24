package graph

import (
	"fmt"
	"math"
	"strconv"

	"github.com/LukasSelin/strata/internal/resamp"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/resample"
)

// Grids (DESIGN.md §55). Every value lies on a grid: the one its input
// declares, or the one a grid change such as Resample produces. An
// operation that combines values requires them on one grid, and panics
// when it is built otherwise. A fused pass runs over one grid, so the
// planner gives each grid of a phase a pass of its own, and a grid change
// is a pass of its own between them.
//
// Two grids are one grid when their size, origin and resolution are
// equal, compared exactly, and their CRSs match (raster.CRS.Matches):
// grids a rounding apart are different grids, and combining their values
// needs a Resample.

// InputOn returns the input named name, declaring it on first use as a
// raster laid out on grid. The raster or source bound to it must have
// grid's size. InputOn of a name already declared panics unless it was
// declared on the same grid.
func (g *Graph) InputOn(name string, grid raster.Grid) Node {
	if name == "" {
		panic("graph: an input needs a name")
	}
	checkGrid(fmt.Sprintf("input %q", name), grid)
	id := g.gridID(grid)
	if v, ok := g.inputs[name]; ok {
		if have := g.gridOfValue(v); have != id {
			panic(fmt.Sprintf("graph: input %q is declared on %s, not %s", name, g.describeGrid(have), g.describeGrid(id)))
		}
		return Node{g, v}
	}
	return g.newInput(name, id)
}

// Grid returns the grid n lies on, and false if n lies on the undeclared
// grid of inputs built with Input.
func (n Node) Grid() (raster.Grid, bool) {
	if n.g == nil {
		panic("graph: Grid of a zero Node")
	}
	id := n.g.gridOfValue(n.v)
	return n.g.grids[id], id != 0
}

// resampling is a Resample node's operation.
type resampling struct {
	opts     resample.Options
	src, dst raster.Grid
	// uncovered reports that some cell of dst lies outside the source,
	// so the result has invalid cells even from a source without them.
	uncovered bool
}

// Resample is resample.Resample: src resampled onto grid dst. It is a
// grid change: its value lies on dst, and can be combined with values on
// that grid, such as an input declared on it. src must lie on a declared
// grid (InputOn), in dst's CRS.
//
// It runs as a pass of its own, which reads src stored and stores its
// result for the passes on dst after it: a tile of the result reads a
// window of src of its own shape, which a pipeline stage cannot
// (DESIGN.md §55).
func Resample(src Node, dst raster.Grid, opts resample.Options) Node {
	if src.g == nil {
		panic("graph: resample.Resample of a zero Node")
	}
	g := src.g
	sg, ok := src.Grid()
	if !ok {
		panic("graph: resample.Resample of a value on the undeclared grid; declare its input's grid with InputOn")
	}
	checkGrid("resample.Resample's dst", dst)
	if !sg.CRS.Matches(dst.CRS) {
		panic(fmt.Sprintf("graph: resample.Resample from CRS %s to CRS %s; reprojection is not supported",
			sg.CRS.Describe(), dst.CRS.Describe()))
	}
	if opts.Method > resample.Average {
		panic(fmt.Sprintf("graph: resample.Resample with unknown %v", opts.Method))
	}
	id := g.gridID(dst)
	key := fmt.Sprintf("Method=%v", opts.Method)
	full := fmt.Sprintf("resample.Resample|%s|%s|[%d]", key, gridKey(dst), src.v)
	if i, ok := g.cse[full]; ok {
		return Node{g, g.nodes[i].first}
	}
	rs := &resampling{opts: opts, src: sg, dst: dst}
	x := resamp.NewAxis(resamp.Method(opts.Method), resamp.Spec{N: dst.Width, Origin: dst.OriginX, Res: dst.ResolutionX,
		SrcN: sg.Width, SrcOrigin: sg.OriginX, SrcRes: sg.ResolutionX})
	y := resamp.NewAxis(resamp.Method(opts.Method), resamp.Spec{N: dst.Height, Origin: dst.OriginY, Res: dst.ResolutionY,
		SrcN: sg.Height, SrcOrigin: sg.OriginY, SrcRes: sg.ResolutionY})
	rs.uncovered = x.Lo != 0 || x.Hi != dst.Width || y.Lo != 0 || y.Hi != dst.Height
	g.nodes = append(g.nodes, node{kind: kindResample, label: "resample.Resample(" + key + ")", in: []int{src.v},
		first: len(g.values), nout: 1, grid: id, rs: rs})
	g.values = append(g.values, len(g.nodes)-1)
	g.cse[full] = len(g.nodes) - 1
	return Node{g, len(g.values) - 1}
}

// gridOfValue returns the id of the grid value v lies on.
func (g *Graph) gridOfValue(v int) int { return g.nodes[g.values[v]].grid }

// gridID returns the id of grid, registering it if it is new. A grid
// that matches a registered one with a CRS the registered one lacks
// gives it that CRS.
func (g *Graph) gridID(grid raster.Grid) int {
	for id := 1; id < len(g.grids); id++ {
		if sameGrid(g.grids[id], grid) {
			if g.grids[id].CRS.Code == "" {
				g.grids[id].CRS = grid.CRS
			}
			return id
		}
	}
	g.grids = append(g.grids, grid)
	return len(g.grids) - 1
}

func sameGrid(a, b raster.Grid) bool {
	return a.Width == b.Width && a.Height == b.Height &&
		a.OriginX == b.OriginX && a.OriginY == b.OriginY &&
		a.ResolutionX == b.ResolutionX && a.ResolutionY == b.ResolutionY &&
		a.CRS.Matches(b.CRS)
}

// describeGrid names grid id for a message or a plan.
func (g *Graph) describeGrid(id int) string { return describeGrid(g.grids, id) }

func describeGrid(grids []raster.Grid, id int) string {
	if id == 0 {
		return "the undeclared grid"
	}
	gr := grids[id]
	f := func(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }
	s := fmt.Sprintf("grid %d×%d from (%s, %s) by (%s, %s)", gr.Width, gr.Height, f(gr.OriginX), f(gr.OriginY), f(gr.ResolutionX), f(gr.ResolutionY))
	if gr.CRS.Code != "" {
		s += " in " + strconv.Quote(gr.CRS.Code)
	}
	return s
}

// gridKey is grid's geometry exactly, for a common-subexpression key.
func gridKey(gr raster.Grid) string {
	return fmt.Sprintf("%d,%d,%x,%x,%x,%x", gr.Width, gr.Height,
		math.Float64bits(gr.OriginX), math.Float64bits(gr.OriginY), math.Float64bits(gr.ResolutionX), math.Float64bits(gr.ResolutionY))
}

// checkGrid panics on a grid resample would reject: a size that is not
// positive, or an origin or resolution that is not finite, or a zero
// resolution.
func checkGrid(what string, gr raster.Grid) {
	if gr.Width <= 0 || gr.Height <= 0 {
		panic(fmt.Sprintf("graph: %s is on a %d×%d grid; sizes must be positive", what, gr.Width, gr.Height))
	}
	for _, v := range []float64{gr.OriginX, gr.OriginY, gr.ResolutionX, gr.ResolutionY} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			panic(fmt.Sprintf("graph: %s is on a grid with a non-finite origin or resolution", what))
		}
	}
	if gr.ResolutionX == 0 || gr.ResolutionY == 0 {
		panic(fmt.Sprintf("graph: %s is on a grid with a zero resolution", what))
	}
}
