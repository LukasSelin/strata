package raster

import "fmt"

// CRS identifies a coordinate reference system. It is a placeholder: the
// engine carries it along with a Grid but never interprets or transforms
// it (DESIGN.md §30).
type CRS struct {
	// Code is an opaque identifier such as "EPSG:25833", or empty if
	// unknown.
	Code string
}

// Grid is the spatial metadata of a raster, kept separate from its cell
// data. Cell (x, y) covers
//
//	[OriginX + x*ResolutionX, OriginX + (x+1)*ResolutionX]
//
// horizontally, and likewise vertically with OriginY and ResolutionY. The
// origin is the outer corner of cell (0, 0). Resolutions are signed, so a
// north-up grid has a negative ResolutionY, as in a GDAL geotransform.
type Grid struct {
	Width  int
	Height int

	ResolutionX float64
	ResolutionY float64

	OriginX float64
	OriginY float64

	CRS CRS
}

// Window returns the grid of the w×h region of g whose top-left cell is
// (x, y), matching Float32Raster.Window on a raster laid out on g. It
// panics under the same conditions.
func (g Grid) Window(x, y, w, h int) Grid {
	if w <= 0 || h <= 0 {
		panic(fmt.Sprintf("raster: window dimensions must be positive, got %d×%d", w, h))
	}
	if x < 0 || y < 0 || x > g.Width-w || y > g.Height-h {
		panic(fmt.Sprintf("raster: window %d×%d at (%d, %d) outside %d×%d grid",
			w, h, x, y, g.Width, g.Height))
	}
	g.OriginX += float64(x) * g.ResolutionX
	g.OriginY += float64(y) * g.ResolutionY
	g.Width = w
	g.Height = h
	return g
}

// Dataset pairs a raster with the grid it is laid out on.
type Dataset struct {
	Grid   Grid
	Raster Float32Raster
}

// NewDataset pairs g and r. It panics if their dimensions differ.
func NewDataset(g Grid, r Float32Raster) Dataset {
	if g.Width != r.Width || g.Height != r.Height {
		panic(fmt.Sprintf("raster: grid is %d×%d but raster is %d×%d",
			g.Width, g.Height, r.Width, r.Height))
	}
	return Dataset{Grid: g, Raster: r}
}

// Window returns the w×h region of d whose top-left cell is (x, y), with
// the raster as a zero-copy view and the grid origin shifted to match.
func (d Dataset) Window(x, y, w, h int) Dataset {
	return Dataset{
		Grid:   d.Grid.Window(x, y, w, h),
		Raster: d.Raster.Window(x, y, w, h),
	}
}
