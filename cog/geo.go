package cog

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/LukasSelin/strata/raster"
)

// GeoTIFF georeferencing, read as GDAL reads it: a geotransform from
// ModelPixelScale and the first ModelTiepoint, or else from
// ModelTransformation, shifted half a cell for PixelIsPoint; and an EPSG
// code from the GeoKeyDirectory. strata's Grid has no rotation terms, so
// a rotated or sheared geotransform is an error rather than silently
// dropped.

// GeoKeys this reader uses.
const (
	keyModelType        = 1024
	keyRasterType       = 1025
	keyCitation         = 1026
	keyGeographicType   = 2048
	keyGeodeticDatum    = 2050
	keyGeogAngularUnits = 2054
	keyGeogEllipsoid    = 2056
	keyProjectedType    = 3072
	keyProjCitation     = 3073
	keyProjection       = 3074
	keyProjCoordTrans   = 3075
	keyProjLinearUnits  = 3076
	modelProjected      = 1
	modelGeographic     = 2
	rasterPixelIsPt     = 2
	userDefined         = 32767
	unitMetre           = 9001
	unitDegree          = 9102
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

	var gt [6]float64 // GDAL's order: x0, dx/dcol, dx/drow, y0, dy/dcol, dy/drow
	var tiepoints []float64
	if te, ok := ifd.entries[tagModelTiepoint]; ok {
		if tiepoints, err = c.floats(te); err != nil {
			return georef{}, err
		}
	}
	var scale []float64
	if se, ok := ifd.entries[tagModelPixelScale]; ok {
		if scale, err = c.floats(se); err != nil {
			return georef{}, err
		}
	}
	switch {
	case len(scale) >= 2 && scale[0] != 0 && scale[1] != 0:
		// The scale wins over ModelTransformation, and the first
		// tiepoint is used however many there are. A negative Y scale
		// is taken to mean north-up, as GDAL does by default
		// (GTIFF_HONOUR_NEGATIVE_SCALEY), contrary to the specification.
		if len(tiepoints) < 6 {
			break
		}
		gt = [6]float64{0, scale[0], 0, 0, 0, -math.Abs(scale[1])}
		gt[0] = tiepoints[3] - tiepoints[0]*gt[1]
		gt[3] = tiepoints[4] - tiepoints[1]*gt[5]
		g.ok = true
	default:
		me, ok := ifd.entries[tagModelTransform]
		if !ok {
			break
		}
		m, err := c.floats(me)
		if err != nil {
			return georef{}, err
		}
		if len(m) != 16 {
			break // GDAL ignores a matrix of any other length
		}
		gt = [6]float64{m[3], m[0], m[1], m[7], m[4], m[5]}
		g.ok = true
	}
	// Tiepoints without a geotransform are GCPs, and GDAL gives the CRS
	// to them, not to the raster's pixel grid.
	gcps := !g.ok && len(tiepoints) >= 6
	if !gcps {
		if code := epsg(keys); code != 0 {
			g.crs = raster.CRS{Code: "EPSG:" + strconv.FormatUint(code, 10)}
		}
	}
	if !g.ok {
		return g, nil
	}
	if gt[2] != 0 || gt[4] != 0 {
		return georef{}, fmt.Errorf("a rotated or sheared geotransform %v is not supported", gt)
	}
	if keys.shorts[keyRasterType] == rasterPixelIsPt {
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
	g.originX, g.resX, g.originY, g.resY = gt[0], gt[1], gt[3], gt[5]
	return g, nil
}

// geoKeySet is the GeoKeyDirectory's keys: the short-valued ones, and
// the ASCII ones, which hold citations. Keys stored as doubles are not
// needed.
type geoKeySet struct {
	shorts map[uint64]uint64
	ascii  map[uint64]string
}

func (k geoKeySet) has(key uint64) bool {
	_, ok := k.shorts[key]
	return ok
}

func (c *container) geoKeys(ifd rawIFD) (geoKeySet, error) {
	keys := geoKeySet{shorts: map[uint64]uint64{}, ascii: map[uint64]string{}}
	e, ok := ifd.entries[tagGeoKeyDirectory]
	if !ok {
		return keys, nil
	}
	d, err := c.uints(e)
	if err != nil {
		return geoKeySet{}, err
	}
	if len(d) < 4 {
		return geoKeySet{}, errors.New("GeoKeyDirectory too short")
	}
	var params string
	if pe, ok := ifd.entries[tagGeoASCIIParams]; ok && pe.typ == typeASCII {
		if b, err := c.bytes(pe); err == nil {
			params = string(b)
		}
	}
	n := int(min(d[3], uint64(len(d)-4)/4)) // #nosec G115 -- len(d) >= 4
	for i := range n {
		k := d[4+4*i : 8+4*i]
		switch {
		case k[1] == 0 && k[2] == 1: // stored inline, as a SHORT
			keys.shorts[k[0]] = k[3]
		case k[1] == tagGeoASCIIParams && k[3] < uint64(len(params)): // #nosec G115 -- a length
			end := min(uint64(len(params)), k[3]+k[2])
			keys.ascii[k[0]] = strings.TrimRight(params[k[3]:end], "|\x00")
		}
	}
	return keys, nil
}

// epsg returns the EPSG code of the CRS the GeoKeys define, or 0 when
// they define none, or one that differs from the code they name: a
// user-defined or geocentric model, keys that give the projected CRS its
// own projection, datum or ellipsoid, or units other than the code's own
// in the unit keys or in an ERDAS IMAGINE or GDAL citation. GDAL
// identifies some of those with an EPSG code through PROJ's database;
// cog reports them as having none rather than risk a wrong one. A code
// that EPSG has deprecated in favour of another is reported as its
// replacement, as GDAL does.
func epsg(keys geoKeySet) uint64 {
	k := keys.shorts
	var code uint64
	switch k[keyModelType] {
	case modelProjected:
		code = k[keyProjectedType]
		for _, redefines := range []uint64{keyProjection, keyProjCoordTrans, keyGeographicType,
			keyGeodeticDatum, keyGeogEllipsoid} {
			if keys.has(redefines) {
				return 0
			}
		}
	case modelGeographic:
		code = k[keyGeographicType]
	default:
		return 0
	}
	if code == 0 || code == userDefined || code > math.MaxUint16 {
		return 0
	}
	if r, ok := lookup(epsgReplaced[:], code); ok {
		code = r
	}
	if k[keyModelType] == modelProjected {
		unit, ok := lookup(epsgProjectedUnits[:], code)
		if !ok {
			unit = unitMetre
		}
		if u, set := k[keyProjLinearUnits]; set && u != unit {
			return 0
		}
		for _, key := range []uint64{keyCitation, keyProjCitation} {
			if name, ok := citationUnits(keys.ascii[key]); ok && (unit != unitMetre || !isMetre(name)) {
				return 0
			}
		}
		return code
	}
	unit, ok := lookup(epsgGeographicUnits[:], code)
	if !ok {
		unit = unitDegree
	}
	if u, set := k[keyGeogAngularUnits]; set && u != unit {
		return 0
	}
	return code
}

// lookup finds code in a table of code<<16 | value, sorted.
func lookup(table []uint32, code uint64) (uint64, bool) {
	i, found := slices.BinarySearchFunc(table, code, func(e uint32, c uint64) int {
		return int(uint64(e>>16)) - int(c) // #nosec G115 -- both below 2^16
	})
	if !found {
		return 0, false
	}
	return uint64(table[i] & 0xffff), true
}

// citationUnits finds the linear unit a citation names: "Units = " in an
// ERDAS IMAGINE citation, or "LUnits = " in one GDAL wrote.
func citationUnits(s string) (string, bool) {
	for _, tag := range []string{"LUnits = ", "Units = "} {
		i := strings.Index(s, tag)
		if i < 0 || (tag == "Units = " && !strings.HasPrefix(s, "IMAGINE GeoTIFF Support")) {
			continue
		}
		v := s[i+len(tag):]
		if j := strings.IndexAny(v, "\n|"); j >= 0 {
			v = v[:j]
		}
		return strings.TrimSpace(v), true
	}
	return "", false
}

func isMetre(name string) bool {
	switch strings.ToLower(name) {
	case "m", "meter", "meters", "metre", "metres":
		return true
	}
	return false
}
