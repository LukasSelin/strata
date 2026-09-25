package zarr

import (
	"errors"
	"fmt"

	zarrv3 "github.com/LukasSelin/zarr"
)

// georef is what an array's attributes say about its grid.
type georef struct {
	ok               bool // a transform was read
	originX, originY float64
	resX, resY       float64
	crs              string
}

// The attributes read; see the package documentation.
const (
	attrTransform    = "spatial:transform"
	attrRegistration = "spatial:registration"
	attrCRS          = "proj:code"
)

// readGeoref reads a's georeferencing attributes. Each is optional; one
// that is present must be well formed.
func readGeoref(a *zarrv3.Array) (georef, error) {
	var g georef
	if _, err := a.Attribute(attrCRS, &g.crs); err != nil {
		return georef{}, fmt.Errorf("%s: %w", attrCRS, err)
	}
	var t []float64
	has, err := a.Attribute(attrTransform, &t)
	if err != nil {
		return georef{}, fmt.Errorf("%s: %w", attrTransform, err)
	}
	var reg string
	if _, err := a.Attribute(attrRegistration, &reg); err != nil {
		return georef{}, fmt.Errorf("%s: %w", attrRegistration, err)
	}
	if reg != "" && reg != "pixel" && reg != "node" {
		return georef{}, fmt.Errorf("%s %q: want \"pixel\" or \"node\"", attrRegistration, reg)
	}
	if !has {
		return g, nil
	}
	if len(t) != 6 {
		return georef{}, fmt.Errorf("%s has %d numbers, want 6", attrTransform, len(t))
	}
	a0, b, c, d, e, f := t[0], t[1], t[2], t[3], t[4], t[5]
	if b != 0 || d != 0 {
		return georef{}, fmt.Errorf("%s %v is rotated or sheared, which raster.Grid cannot hold", attrTransform, t)
	}
	if a0 == 0 || e == 0 {
		return georef{}, errors.New(attrTransform + " has a resolution of 0")
	}
	g.ok = true
	g.resX, g.resY, g.originX, g.originY = a0, e, c, f
	if reg == "node" {
		// The transform places cell centres; a Grid's origin is a corner.
		g.originX -= a0 / 2
		g.originY -= e / 2
	}
	return g, nil
}
