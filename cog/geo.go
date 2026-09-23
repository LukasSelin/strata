package cog

import (
	"errors"
	"fmt"
	"math"
	"strconv"

	"github.com/LukasSelin/strata/raster"
)

// GeoTIFF georeferencing, read as GDAL reads it: a geotransform from
// ModelTransformation, or from ModelPixelScale and one ModelTiepoint,
// shifted half a cell for PixelIsPoint; and an EPSG code from the
// GeoKeyDirectory. strata's Grid has no rotation terms, so a rotated or
// sheared geotransform is an error rather than silently dropped.

// GeoKeys this reader uses.
const (
	keyModelType      = 1024
	keyRasterType     = 1025
	keyGeographicType = 2048
	keyProjectedType  = 3072
	modelProjected    = 1
	modelGeographic   = 2
	rasterPixelIsPt   = 2
	userDefined       = 32767
)

// georef is a level-0 image's georeferencing.
type georef struct {
	ok            bool // false: the file has no geotransform
	originX, resX float64
	originY, resY float64
	crs           raster.CRS
}

func (c *container) georef(ifd rawIFD) (georef, error) {
	keys, err := c.geoKeys(ifd)
	if err != nil {
		return georef{}, err
	}
	var g georef
	if code := epsg(keys); code != 0 {
		g.crs = raster.CRS{Code: "EPSG:" + strconv.FormatUint(code, 10)}
	}

	var gt [6]float64 // GDAL's order: x0, dx/dcol, dx/drow, y0, dy/dcol, dy/drow
	if e, ok := ifd.entries[tagModelTransform]; ok {
		m, err := c.floats(e)
		if err != nil {
			return georef{}, err
		}
		if len(m) != 16 {
			return georef{}, fmt.Errorf("ModelTransformation has %d values, want 16", len(m))
		}
		gt = [6]float64{m[3], m[0], m[1], m[7], m[4], m[5]}
	} else {
		se, okS := ifd.entries[tagModelPixelScale]
		te, okT := ifd.entries[tagModelTiepoint]
		if !okS || !okT {
			return g, nil // not georeferenced, or by GCPs only
		}
		s, err := c.floats(se)
		if err != nil {
			return georef{}, err
		}
		t, err := c.floats(te)
		if err != nil {
			return georef{}, err
		}
		if len(s) < 2 || len(t) < 6 {
			return georef{}, errors.New("ModelPixelScale or ModelTiepoint too short")
		}
		if len(t) > 6 {
			return g, nil // several tiepoints are GCPs, not a geotransform
		}
		gt = [6]float64{t[3] - t[0]*s[0], s[0], 0, t[4] + t[1]*s[1], 0, -s[1]}
	}
	if gt[2] != 0 || gt[4] != 0 {
		return georef{}, fmt.Errorf("a rotated or sheared geotransform %v is not supported", gt)
	}
	if keys[keyRasterType] == rasterPixelIsPt {
		// GDAL's default (GTIFF_POINT_GEO_IGNORE=NO): the tiepoint is
		// a cell centre, so the corner is half a cell up and left.
		gt[0] -= 0.5 * gt[1]
		gt[3] -= 0.5 * gt[5]
	}
	for _, v := range gt {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return georef{}, fmt.Errorf("geotransform %v is not finite", gt)
		}
	}
	g.ok = true
	g.originX, g.resX, g.originY, g.resY = gt[0], gt[1], gt[3], gt[5]
	return g, nil
}

// geoKeys reads the GeoKeyDirectory's short-valued keys. Keys stored in
// the double or ASCII parameter tags are not needed and are skipped.
func (c *container) geoKeys(ifd rawIFD) (map[uint64]uint64, error) {
	keys := map[uint64]uint64{}
	e, ok := ifd.entries[tagGeoKeyDirectory]
	if !ok {
		return keys, nil
	}
	d, err := c.uints(e)
	if err != nil {
		return nil, err
	}
	if len(d) < 4 {
		return nil, errors.New("GeoKeyDirectory too short")
	}
	n := int(min(d[3], uint64(len(d)-4)/4)) // #nosec G115 -- len(d) >= 4
	for i := range n {
		k := d[4+4*i : 8+4*i]
		if k[1] == 0 && k[2] == 1 { // stored inline, as a SHORT
			keys[k[0]] = k[3]
		}
	}
	return keys, nil
}

// epsg returns the CRS's EPSG code, or 0 if it is user-defined or absent.
func epsg(keys map[uint64]uint64) uint64 {
	var code uint64
	switch keys[keyModelType] {
	case modelProjected:
		code = keys[keyProjectedType]
	case modelGeographic:
		code = keys[keyGeographicType]
	default:
		// No model type: whichever CRS key is there.
		code = keys[keyProjectedType]
		if code == 0 {
			code = keys[keyGeographicType]
		}
	}
	if code == userDefined {
		return 0
	}
	return code
}
