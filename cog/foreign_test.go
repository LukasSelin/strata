package cog

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/LukasSelin/strata/raster"
)

// Tests for GeoTIFFs that other software writes. Each is the minimal
// form of a file in acceptance/corpus that GDAL read and the reader did
// not, or read differently; the file it comes from is named. The
// expected behaviour is GDAL's, read from its source and confirmed by
// acceptance/cogcorpus.sh, not the reader's own.

// open writes fs and opens it.
func open(t *testing.T, fs fileSpec) *File {
	t.Helper()
	f, err := Open(bytes.NewReader(fs.write()))
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// readLevel reads a whole band of a level, with validity.
func readLevel(t *testing.T, f *File, band, level int) (raster.Float32Raster, *Source) {
	t.Helper()
	src, err := f.Source(SourceOptions{Band: band, Level: level})
	if err != nil {
		t.Fatal(err)
	}
	w, h := src.Size()
	dst := raster.NewFloat32(w, h, make([]float32, w*h))
	dst.Valid = raster.NewMask(w * h)
	if err := src.ReadWindow(context.Background(), dst, 0, 0); err != nil {
		t.Fatal(err)
	}
	return dst, src
}

func le() fileSpec { return fileSpec{order: binary.LittleEndian} }

// TestOldStyleLZW reads LZW in libtiff's pre-5.0 style, least significant
// bit first (libtiff-test/quad-lzw-compat.tiff, libtiff-pics/quad-lzw.tif,
// gdal-autotest quad-lzw-old-style.tif).
func TestOldStyleLZW(t *testing.T) {
	for _, lay := range layouts {
		sp := grid16()
		sp.tiled, sp.blockW, sp.blockH, sp.planar = lay.tiled, lay.blockW, lay.blockH, planarChunky
		sp.compression = compressionLZW
		fs := le()
		fs.images = []imageSpec{sp}
		f := open(t, fs)
		src, _ := f.Source(SourceOptions{})
		checkWindow(t, lay.name, src, sp.vals[0], sp.w, 0, 0, sp.w, sp.h, false, 0)
	}
}

// TestShortEdgeTile reads a last row of tiles that stops at the image's
// last row (gdal-autotest contig_tiled.tif, separate_tiled.tif), and
// still refuses a tile shorter than that.
func TestShortEdgeTile(t *testing.T) {
	for _, comp := range []uint64{compressionNone, compressionPackBits, compressionDeflate} {
		sp := grid16()
		sp.h, sp.vals[0] = 10, sp.vals[0][:160] // the last tile row has 2 of 4 rows
		sp.compression, sp.cropTiles = comp, true
		fs := le()
		fs.images = []imageSpec{sp}
		f := open(t, fs)
		src, _ := f.Source(SourceOptions{})
		checkWindow(t, compressionName(comp), src, sp.vals[0], sp.w, 0, 0, sp.w, sp.h, false, 0)
	}
	sp := grid16()
	sp.cropTiles = true
	sp.h, sp.vals[0] = 10, sp.vals[0][:160]
	sp.edit = func(t []tagValue) []tagValue { return setTag(t, tagImageLength, 12) }
	fs := le()
	fs.images = []imageSpec{sp}
	f := open(t, fs)
	src, _ := f.Source(SourceOptions{})
	dst := raster.NewFloat32(16, 12, make([]float32, 16*12))
	if err := src.ReadWindow(context.Background(), dst, 0, 0); err == nil || !strings.Contains(err.Error(), "decodes to") {
		t.Errorf("a tile short of the image's rows: %v", err)
	}
}

// setTag replaces the values of tag.
func setTag(tags []tagValue, tag uint16, vals ...uint64) []tagValue {
	for i := range tags {
		if tags[i].tag == tag {
			tags[i].uints = vals
		}
	}
	return tags
}

// dropTag removes tag.
func dropTag(tags []tagValue, tag uint16) []tagValue {
	return slices.DeleteFunc(tags, func(v tagValue) bool { return v.tag == tag })
}

// TestBlockTags reads the block tags as libtiff does: tiles listed under
// StripOffsets (libtiff-pics/cramps-tile.tif, quad-tile.tif); offsets
// as SLONG8 (byte_bigtiff_invalid_slong8_for_stripoffsets.tif); lists
// shorter than the block count, padded with 0, and longer, cut
// (size_of_stripbytecount_*.tif, stripbytecounts_count_not_same_*.tif);
// a block of 0 bytes read as absent whatever its offset; and byte counts
// missing from an uncompressed file, computed (one_strip_nobytecount.tif).
func TestBlockTags(t *testing.T) {
	type edited struct {
		name   string
		edit   func([]tagValue) []tagValue
		big    bool
		absent []int // blocks that read as 0
	}
	toStripTags := func(tags []tagValue) []tagValue {
		for i := range tags {
			switch tags[i].tag {
			case tagTileOffsets:
				tags[i].tag = tagStripOffsets
			case tagTileByteCounts:
				tags[i].tag = tagStripByteCounts
			}
		}
		return tags
	}
	cut := func(n int, which ...uint16) func([]tagValue) []tagValue {
		return func(tags []tagValue) []tagValue {
			for i := range tags {
				if slices.Contains(which, tags[i].tag) {
					tags[i].uints = tags[i].uints[:n]
				}
			}
			return tags
		}
	}
	extend := func(tags []tagValue) []tagValue {
		for i := range tags {
			if tags[i].tag == tagTileOffsets || tags[i].tag == tagTileByteCounts {
				tags[i].uints = append(tags[i].uints, 12345, 678)
			}
		}
		return tags
	}
	zeroCount := func(tags []tagValue) []tagValue {
		for i := range tags {
			if tags[i].tag == tagTileByteCounts {
				tags[i].uints[3] = 0 // the offset stays
			}
		}
		return tags
	}
	signed := func(tags []tagValue) []tagValue {
		for i := range tags {
			if tags[i].tag == tagTileOffsets {
				tags[i].typ = typeSLong8
			}
		}
		return tags
	}
	for _, c := range []edited{
		{name: "tiles in strip tags", edit: toStripTags},
		{name: "short lists", edit: cut(4, tagTileOffsets, tagTileByteCounts), absent: []int{4, 5}},
		{name: "short counts", edit: cut(5, tagTileByteCounts), absent: []int{5}},
		{name: "long lists", edit: extend},
		{name: "0 bytes at an offset", edit: zeroCount, absent: []int{3}},
		{name: "SLONG8 offsets", edit: signed, big: true},
	} {
		sp := grid16()
		sp.edit = c.edit
		fs := fileSpec{order: binary.BigEndian, big: c.big, images: []imageSpec{sp}}
		f := open(t, fs)
		dst, _ := readLevel(t, f, 0, 0)
		for y := range 12 {
			for x := range 16 {
				want := float32(x + 100*y)
				if slices.Contains(c.absent, (y/4)*2+x/8) {
					want = 0
				}
				if got := dst.Data[dst.Index(x, y)]; got != want || !dst.IsValid(x, y) {
					t.Fatalf("%s: cell (%d, %d) = %v, want %v", c.name, x, y, got, want)
				}
			}
		}
	}

	// Missing byte counts: computed for uncompressed data, refused for
	// compressed data, whose size the layout does not give.
	for _, comp := range []uint64{compressionNone, compressionDeflate} {
		for _, lay := range layouts {
			sp := grid16()
			sp.tiled, sp.blockW, sp.blockH, sp.planar = lay.tiled, lay.blockW, lay.blockH, lay.planar
			sp.compression = comp
			sp.edit = func(t []tagValue) []tagValue {
				return dropTag(dropTag(t, tagStripByteCounts), tagTileByteCounts)
			}
			fs := le()
			fs.images = []imageSpec{sp}
			f, err := Open(bytes.NewReader(fs.write()))
			if comp != compressionNone {
				if err == nil {
					t.Errorf("%s: Open accepted compressed data without byte counts", lay.name)
				}
				continue
			}
			if err != nil {
				t.Fatalf("%s: %v", lay.name, err)
			}
			src, _ := f.Source(SourceOptions{})
			checkWindow(t, "no byte counts/"+lay.name, src, sp.vals[0], sp.w, 0, 0, sp.w, sp.h, false, 0)
		}
	}
}

// TestIFDChain reads the first image of a file whose IFD chain loops or
// points past the end, as libtiff does (libtiff-test/test_ifd_loop_*.tif).
func TestIFDChain(t *testing.T) {
	ov := halve(grid16())
	ov.subfile = subfileReduced
	fs := le()
	fs.images = []imageSpec{grid16(), ov}
	file := fs.write()
	first := binary.LittleEndian.Uint32(file[4:])
	n := binary.LittleEndian.Uint16(file[first:])
	nextOf0 := first + 2 + 12*uint32(n)
	second := binary.LittleEndian.Uint32(file[nextOf0:])
	n2 := binary.LittleEndian.Uint16(file[second:])
	nextOf1 := second + 2 + 12*uint32(n2)
	for _, c := range []struct {
		name   string
		at     uint32 // the next pointer to patch
		to     uint32
		levels int
	}{
		{"loop to self", nextOf0, first, 1},
		{"loop to first", nextOf1, first, 2},
		{"past the end", nextOf1, uint32(len(file)) + 1000, 2},
		{"into garbage", nextOf0, 9, 1}, // an entry count of 0x... in pixel data
	} {
		b := slices.Clone(file)
		binary.LittleEndian.PutUint32(b[c.at:], c.to)
		f, err := Open(bytes.NewReader(b))
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if f.Levels() > c.levels {
			t.Errorf("%s: %d levels, want at most %d", c.name, f.Levels(), c.levels)
		}
		src, _ := f.Source(SourceOptions{})
		checkWindow(t, c.name, src, grid16().vals[0], 16, 0, 0, 16, 12, false, 0)
	}
}

// TestSubIFDs reads overviews that the first image lists as SubIFDs, as
// GDAL does (gdal-autotest tiff_with_subifds.tif,
// libtiff-test/tiff_with_subifd_chain.tif): they come before overviews
// in the IFD chain.
func TestSubIFDs(t *testing.T) {
	full := grid16()
	ov1 := halve(full)
	ov1.subfile = subfileReduced
	ov2 := halve(ov1)
	ov2.subfile = subfileReduced
	page := grid16() // a second page, not an overview
	fs := le()
	fs.images, fs.subIFDs = []imageSpec{full, ov1, ov2, page}, 2
	f := open(t, fs)
	if f.Levels() != 3 {
		t.Fatalf("%d levels, want 3", f.Levels())
	}
	for lvl, sp := range []imageSpec{full, ov1, ov2} {
		src, _ := f.Source(SourceOptions{Level: lvl})
		checkWindow(t, "subIFD level", src, sp.vals[0], sp.w, 0, 0, sp.w, sp.h, false, 0)
		// Without a geotransform, GDAL gives every level the grid of
		// its own cells, not one scaled from the full resolution's.
		want := raster.Grid{Width: sp.w, Height: sp.h, ResolutionX: 1, ResolutionY: 1}
		if g := f.Grid(lvl); g != want {
			t.Errorf("level %d: grid %+v, want %+v", lvl, g, want)
		}
	}
}

// TestFillOrder reads FillOrder 2, whose stored bytes libtiff reverses
// bit by bit before decoding them, for every codec cog reads
// (generated-tiffcp gray16-fillorder2.tif, rgb8-lzw-fillorder2.tif).
func TestFillOrder(t *testing.T) {
	for _, comp := range []uint64{compressionNone, compressionLZW, compressionDeflate, compressionPackBits, compressionZSTD} {
		sp := grid16()
		sp.compression, sp.lsbFirst = comp, true
		fs := le()
		fs.images = []imageSpec{sp}
		src, _ := open(t, fs).Source(SourceOptions{})
		checkWindow(t, "FillOrder 2/"+compressionName(comp), src, sp.vals[0], sp.w, 0, 0, sp.w, sp.h, false, 0)
	}
}

// maskOf is a mask image of sp's size: 1-bit or 8-bit, with 0 where
// hole says.
func maskOf(sp imageSpec, bits int, hole func(x, y int) bool) imageSpec {
	v := make([]float64, sp.w*sp.h)
	for i := range v {
		if !hole(i%sp.w, i/sp.w) {
			v[i] = 255
			if bits == 1 {
				v[i] = 1
			}
		}
	}
	m := imageSpec{w: sp.w, h: sp.h, bands: 1, format: sampleUint, size: bits / 8, tiled: true,
		blockW: 16, blockH: 16, planar: planarChunky, compression: compressionDeflate,
		vals: [][]float64{v}, subfile: subfileMask}
	return m
}

// TestMasks reads GDAL's internal masks (gdal-autotest test_with_mask_*.tif,
// test3_with_mask_*.tif, generated-gdal int16-internal-mask*.tif,
// generated-rasterio uint8-dataset-mask.tif): a transparency-mask IFD of
// 1- or 8-bit samples gives the validity of the level of its size, cells
// where it is 0 invalid, for every band; with a mask, NoData is ignored;
// an overview without a mask of its own falls back to NoData.
func TestMasks(t *testing.T) {
	for _, bits := range []int{1, 8} {
		full := grid16(tagValue{tag: tagGDALNoData, typ: typeASCII, str: "3"})
		full.bands, full.vals = 2, [][]float64{full.vals[0], full.vals[0]}
		hole := func(x, y int) bool { return x > 10 && y < 5 }
		mask := maskOf(full, bits, hole)
		ov := halve(full)
		ov.bands, ov.vals, ov.subfile = 2, [][]float64{ov.vals[0], ov.vals[0]}, subfileReduced
		ov.vals[0][3], ov.vals[1][3] = 3, 3 // NoData, which counts at this level
		ovMaskless := halve(ov)
		ovMaskless.bands, ovMaskless.vals = 2, [][]float64{ovMaskless.vals[0], ovMaskless.vals[0]}
		ovMaskless.vals[0][0], ovMaskless.vals[1][0] = 3, 3
		ovMask := maskOf(ov, bits, func(x, y int) bool { return x == 0 })
		ovMask.subfile = subfileReduced | subfileMask
		fs := le()
		// COG order: the image, its mask, then each overview and its mask.
		fs.images = []imageSpec{full, mask, ov, ovMask, ovMaskless}
		f := open(t, fs)
		if f.Levels() != 3 {
			t.Fatalf("%d-bit: %d levels", bits, f.Levels())
		}
		for band := range 2 {
			dst, src := readLevel(t, f, band, 0)
			if !src.Masked() {
				t.Errorf("%d-bit: level 0 not Masked", bits)
			}
			for y := range 12 {
				for x := range 16 {
					if dst.IsValid(x, y) == hole(x, y) {
						t.Fatalf("%d-bit band %d: cell (%d, %d) valid %v", bits, band, x, y, dst.IsValid(x, y))
					}
					if got, want := dst.Data[dst.Index(x, y)], float32(x+100*y); got != want {
						t.Fatalf("%d-bit: cell (%d, %d) = %v, want %v", bits, x, y, got, want)
					}
				}
			}
			// Level 1: its own mask, not NoData.
			dst, _ = readLevel(t, f, band, 1)
			for y := range ov.h {
				for x := range ov.w {
					if dst.IsValid(x, y) != (x != 0) {
						t.Fatalf("%d-bit level 1: cell (%d, %d) valid %v", bits, x, y, dst.IsValid(x, y))
					}
				}
			}
			// Level 2: no mask, so NoData.
			dst, src = readLevel(t, f, band, 2)
			if !src.Masked() || dst.IsValid(0, 0) || !dst.IsValid(1, 0) {
				t.Fatalf("%d-bit level 2: Masked %v, validity %v %v", bits, src.Masked(), dst.IsValid(0, 0), dst.IsValid(1, 0))
			}
		}
	}

	// A mask of the wrong size, or of 16-bit samples, is not a mask.
	full := grid16()
	for _, m := range []imageSpec{halve(maskOf(full, 8, func(int, int) bool { return true })),
		func() imageSpec { m := maskOf(full, 8, func(int, int) bool { return true }); m.size = 2; return m }()} {
		m.subfile = subfileMask
		fs := le()
		fs.images = []imageSpec{full, m}
		f := open(t, fs)
		if _, src := readLevel(t, f, 0, 0); src.Masked() {
			t.Errorf("a %d×%d mask of %d-byte samples was used", m.w, m.h, m.size)
		}
	}
}

// TestPhotometric refuses the colour spaces GDAL converts to RGB rather
// than reading as stored (gdal-autotest ycbcr_*.tif, rgbsmall_cmyk.tif,
// cielab.tif, libtiff-pics/ycbcr-cat.tif, flower-separated-*-08.tif), and
// reads 16-bit CMYK, which GDAL reads as stored.
func TestPhotometric(t *testing.T) {
	for _, c := range []struct {
		photometric uint64
		size        int
		refused     bool
	}{
		{photometricYCbCr, 1, true},
		{photometricYCbCr, 2, true},
		{photometricSeparated, 1, true},
		{photometricSeparated, 2, false},
		{photometricCIELab, 1, true},
		{2, 1, false}, // RGB
		{3, 1, false}, // palette: GDAL reads the indices
	} {
		sp := grid16()
		sp.format, sp.size = sampleUint, c.size
		for i := range sp.vals[0] {
			sp.vals[0][i] = math.Mod(sp.vals[0][i], 256)
		}
		sp.edit = func(t []tagValue) []tagValue { return setTag(t, tagPhotometric, c.photometric) }
		fs := le()
		fs.images = []imageSpec{sp}
		_, err := Open(bytes.NewReader(fs.write()))
		if c.refused != (err != nil) || (err != nil && !strings.Contains(err.Error(), "not supported")) {
			t.Errorf("photometric %d, %d-byte: %v", c.photometric, c.size, err)
		}
	}
}

// TestNoDataValues refuses GDAL's NODATA_VALUES metadata, a NoData value
// per band that marks a cell only where all bands hold theirs
// (gdal-autotest test_nodatavalues.tif), and ignores it when it has the
// wrong number of values, as GDAL does.
func TestNoDataValues(t *testing.T) {
	for _, c := range []struct {
		values  string
		refused bool
	}{{"0 0 0", true}, {"0 0", false}} {
		sp := grid16(tagValue{tag: tagGDALMetadata, typ: typeASCII,
			str: `<GDALMetadata><Item name="NODATA_VALUES">` + c.values + `</Item></GDALMetadata>`})
		sp.bands, sp.vals = 3, [][]float64{sp.vals[0], sp.vals[0], sp.vals[0]}
		fs := le()
		fs.images = []imageSpec{sp}
		_, err := Open(bytes.NewReader(fs.write()))
		if c.refused != (err != nil) {
			t.Errorf("NODATA_VALUES %q: %v", c.values, err)
		}
	}
}

// TestAtofM checks GDAL_NODATA parsing against GDAL's CPLAtofM
// (gdal-autotest empty_nodata.tif, stats_nodata_*_msvc.tif).
func TestAtofM(t *testing.T) {
	for _, c := range []struct {
		s    string
		want float64
	}{
		{"-9999", -9999}, {" 12.5 ", 12.5}, {"1e-9", 1e-9}, {"+3", 3},
		{"12abc", 12}, {"abc", 0}, {"1,5", 1.5}, {"1e", 1}, {"-", 0},
		{"1e400", math.Inf(1)}, {"-1e400", math.Inf(-1)},
		{"inf", math.Inf(1)}, {"-inf", math.Inf(-1)}, {"Infinity", math.Inf(1)}, {"-Infinity", math.Inf(-1)},
		{"1.#INF", math.Inf(1)}, {"-1.#INF", math.Inf(-1)}, {"1.#inf", math.Inf(1)},
		{"nan", math.NaN()}, {"NaN", math.NaN()}, {"1.#QNAN", math.NaN()}, {"-1.#IND", math.NaN()},
		{"-nan", 0}, {"0x10", 0}, {"1_000", 1},
	} {
		got := atofM(c.s)
		if got != c.want && !(math.IsNaN(got) && math.IsNaN(c.want)) {
			t.Errorf("atofM(%q) = %v, want %v", c.s, got, c.want)
		}
	}
	sp := grid16(tagValue{tag: tagGDALNoData, typ: typeASCII, str: ""})
	fs := le()
	fs.images = []imageSpec{sp}
	if _, has := open(t, fs).NoData(); has {
		t.Error("an empty GDAL_NODATA is a NoData value")
	}
}

// TestGeoreferencingGDAL follows GDAL where the GeoTIFF tags are odd:
// a negative Y scale taken as north-up (gdal-autotest negative_scaley.tif);
// the scale and tiepoint before ModelTransformation; the first of several
// tiepoints with a scale; tiepoints without one are GCPs, with no
// geotransform and no CRS on the pixel grid (byte_gcp.tif); a matrix of
// the wrong length ignored.
func TestGeoreferencingGDAL(t *testing.T) {
	tie := func(vals ...float64) tagValue {
		return tagValue{tag: tagModelTiepoint, typ: typeDouble, floats: vals}
	}
	scale := func(sx, sy float64) tagValue {
		return tagValue{tag: tagModelPixelScale, typ: typeDouble, floats: []float64{sx, sy, 0}}
	}
	matrix := tagValue{tag: tagModelTransform, typ: typeDouble, floats: []float64{
		2, 0, 0, 7, 0, -2, 0, 8, 0, 0, 0, 0, 0, 0, 0, 1}}
	crs := geoTags(modelGeographic, 1, 4326)
	for _, c := range []struct {
		name string
		tags []tagValue
		want raster.Grid
		geo  bool
	}{
		{"negative scale", []tagValue{scale(10, -10), tie(0, 0, 0, 100, 200, 0), crs},
			raster.Grid{ResolutionX: 10, ResolutionY: -10, OriginX: 100, OriginY: 200, CRS: raster.CRS{Code: "EPSG:4326"}}, true},
		{"scale over matrix", []tagValue{scale(1, 1), tie(0, 0, 0, 5, 6, 0), matrix},
			raster.Grid{ResolutionX: 1, ResolutionY: -1, OriginX: 5, OriginY: 6}, true},
		{"first tiepoint", []tagValue{scale(1, 1), tie(2, 3, 0, 5, 6, 0, 9, 9, 0, 1, 1, 0)},
			raster.Grid{ResolutionX: 1, ResolutionY: -1, OriginX: 3, OriginY: 9}, true},
		{"GCPs", []tagValue{tie(0, 0, 0, 5, 6, 0, 9, 9, 0, 1, 1, 0), crs},
			raster.Grid{ResolutionX: 1, ResolutionY: 1}, false},
		{"zero scale", []tagValue{scale(0, 1), tie(0, 0, 0, 5, 6, 0), matrix},
			raster.Grid{ResolutionX: 2, ResolutionY: -2, OriginX: 7, OriginY: 8}, true},
		{"short matrix", []tagValue{{tag: tagModelTransform, typ: typeDouble, floats: []float64{1, 0, 0, 1}}, crs},
			raster.Grid{ResolutionX: 1, ResolutionY: 1, CRS: raster.CRS{Code: "EPSG:4326"}}, false},
	} {
		fs := le()
		fs.images = []imageSpec{grid16(c.tags...)}
		f := open(t, fs)
		c.want.Width, c.want.Height = 16, 12
		if g := f.Grid(0); g != c.want || f.Georeferenced() != c.geo {
			t.Errorf("%s: grid %+v georeferenced %v, want %+v %v", c.name, g, f.Georeferenced(), c.want, c.geo)
		}
	}
}

// keys is a GeoKeyDirectory of the given keys, with citations stored in
// GeoAsciiParams.
func keys(shorts [][2]uint64, citations map[uint64]string) []tagValue {
	dir := []uint64{1, 1, 0, uint64(len(shorts) + len(citations))}
	var ascii string
	var entries [][4]uint64
	for _, k := range shorts {
		entries = append(entries, [4]uint64{k[0], 0, 1, k[1]})
	}
	for k, s := range citations {
		entries = append(entries, [4]uint64{k, tagGeoASCIIParams, uint64(len(s) + 1), uint64(len(ascii))})
		ascii += s + "|"
	}
	slices.SortFunc(entries, func(a, b [4]uint64) int { return int(a[0]) - int(b[0]) })
	for _, e := range entries {
		dir = append(dir, e[:]...)
	}
	tags := []tagValue{{tag: tagGeoKeyDirectory, typ: typeShort, uints: dir}}
	if ascii != "" {
		tags = append(tags, tagValue{tag: tagGeoASCIIParams, typ: typeASCII, str: ascii})
	}
	return tags
}

// TestCRS checks that an EPSG code is named only when the GeoKeys do not
// redefine it, and a deprecated one as its replacement, as GDAL names it.
func TestCRS(t *testing.T) {
	proj := func(code uint64, more ...[2]uint64) [][2]uint64 {
		return append([][2]uint64{{keyModelType, modelProjected}, {keyRasterType, 1}, {keyProjectedType, code}}, more...)
	}
	geog := func(code uint64, more ...[2]uint64) [][2]uint64 {
		return append([][2]uint64{{keyModelType, modelGeographic}, {keyRasterType, 1}, {keyGeographicType, code}}, more...)
	}
	imagine := "IMAGINE GeoTIFF Support\nProjection Name = x\nUnits = %s\nGeoTIFF Units = %s"
	for _, c := range []struct {
		name      string
		keys      [][2]uint64
		citations map[uint64]string
		want      string
	}{
		{"projected", proj(32633, [2]uint64{keyProjLinearUnits, unitMetre}), nil, "EPSG:32633"},
		{"geographic, GDAL's keys", geog(4326, [2]uint64{keyGeogAngularUnits, unitDegree}), nil, "EPSG:4326"},
		// WGS_1984_Web_Mercator_Auxiliary_Sphere.tif, pci_eg/meter.tif, intergraph/geoc.tif
		{"user-defined model", [][2]uint64{{keyModelType, userDefined}, {keyGeographicType, 4326}}, nil, ""},
		{"geocentric model", [][2]uint64{{keyModelType, 3}, {keyGeographicType, 4322}}, nil, ""},
		{"no model type", [][2]uint64{{keyProjectedType, 32633}}, nil, ""},
		// other/erdas_spnad83.tif, epsg_2853_with_us_feet.tif, pci_eg/spif83.tif
		{"metre CRS in feet", proj(26966, [2]uint64{keyProjLinearUnits, 9003}), nil, ""},
		// pci_eg/spcs27.tif: EPSG:26746 is in US survey feet
		{"feet CRS in metres", proj(26746, [2]uint64{keyProjLinearUnits, unitMetre}), nil, ""},
		{"feet CRS in feet", proj(26746, [2]uint64{keyProjLinearUnits, 9003}), nil, "EPSG:26746"},
		{"grads CRS in degrees", geog(4807, [2]uint64{keyGeogAngularUnits, unitDegree}), nil, ""},
		{"degrees CRS in grads", geog(4326, [2]uint64{keyGeogAngularUnits, 9105}), nil, ""},
		// citation_mixedcase.tif
		{"IMAGINE citation in feet", proj(2838), map[uint64]string{keyCitation: strings.ReplaceAll(imagine, "%s", "international_FeeT")}, ""},
		{"IMAGINE citation in metres", proj(2838), map[uint64]string{keyCitation: strings.ReplaceAll(imagine, "%s", "meters")}, "EPSG:2838"},
		{"GDAL citation in feet", proj(32633), map[uint64]string{keyProjCitation: "LUnits = foot"}, ""},
		{"plain citation", proj(32633), map[uint64]string{keyProjCitation: "WGS 84 / UTM zone 33N"}, "EPSG:32633"},
		// projection_3856.tif
		{"own projection key", proj(3857, [2]uint64{keyProjection, 3856}), nil, ""},
		{"geographic key on a projected CRS", proj(32633, [2]uint64{keyGeographicType, 4326}), nil, ""},
		// made_up/bogota.tif, made_up/ntf_nord.tif
		{"deprecated", proj(21892), nil, "EPSG:21897"},
		{"deprecated", proj(27591), nil, "EPSG:27561"},
	} {
		fs := le()
		fs.images = []imageSpec{grid16(append(scaleTiepoint(1, 1, 0, 0), keys(c.keys, c.citations)...)...)}
		if g := open(t, fs).Grid(0); g.CRS.Code != c.want {
			t.Errorf("%s: CRS %q, want %q", c.name, g.CRS.Code, c.want)
		}
	}
}
