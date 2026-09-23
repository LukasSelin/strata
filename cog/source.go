package cog

import (
	"context"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strings"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/raster"
)

// File is an open GeoTIFF: its resolution levels and georeferencing. It
// holds the tags it needs and no cell data; Source reads cells.
type File struct {
	c      *container
	levels []*image
	masks  []*image // each level's transparency mask, or nil
	geo    georef
	nodata float64
	hasND  bool
}

// maxOverviews is GDAL's limit on the overviews it reads from one file.
const maxOverviews = 30

// Open reads the header and every image directory of the GeoTIFF that r
// holds. It reads no cell data. r must stay open while the File and its
// sources are in use, and must be safe for concurrent ReadAt calls, as
// *os.File and engine.RawFile are.
//
// The first image is level 0, the full resolution. The other images are
// read as GDAL reads them: those of the first image's SubIFDs, then the
// rest of the IFD chain, in that order. Each reduced-resolution image
// (NewSubfileType bit 0) with the same number of bands is an overview, a
// level after the ones before it. A transparency mask (NewSubfileType
// bit 2) of 1- or 8-bit samples becomes the validity of the level of its
// size, as GDAL's internal mask does; other images are skipped.
//
// It returns an error for a file that is not a TIFF, is corrupt, or uses
// something outside what the package supports (see the package
// documentation). It never panics on a file's contents.
func Open(r io.ReaderAt) (*File, error) {
	c, first, err := readHeader(r)
	if err != nil {
		return nil, fmt.Errorf("cog: %w", err)
	}
	ifds, err := c.readIFDs(first)
	if err != nil {
		return nil, fmt.Errorf("cog: %w", err)
	}
	f := &File{c: c}
	main, err := c.parseImage(ifds[0], false)
	if err != nil {
		return nil, fmt.Errorf("cog: image 0: %w", err)
	}
	f.levels, f.masks = []*image{main}, []*image{nil}
	if err := f.scan(ifds); err != nil {
		return nil, fmt.Errorf("cog: %w", err)
	}
	if err := c.checkNoDataValues(ifds[0], main.bands); err != nil {
		return nil, fmt.Errorf("cog: %w", err)
	}
	if f.geo, err = c.georef(ifds[0]); err != nil {
		return nil, fmt.Errorf("cog: georeferencing: %w", err)
	}
	if f.nodata, f.hasND, err = c.noData(ifds[0]); err != nil {
		return nil, fmt.Errorf("cog: %w", err)
	}
	return f, nil
}

// scan finds the overviews and masks among the images after the first,
// as GDAL's GTiffDataset::ScanDirectories does.
func (f *File) scan(ifds []rawIFD) error {
	c, main := f.c, f.levels[0]
	var rest []rawIFD
	if e, ok := ifds[0].entries[tagSubIFDs]; ok {
		offs, err := c.uints(e)
		if err != nil {
			return fmt.Errorf("SubIFDs: %w", err)
		}
		for _, off := range offs {
			ifd, _, err := c.readIFD(off)
			if err != nil {
				break // as GDAL: the SubIFDs end at one it cannot read
			}
			rest = append(rest, ifd)
		}
	}
	for _, ifd := range ifds[1:] {
		if _, ok := ifd.entries[tagSubIFDs]; !ok {
			rest = append(rest, ifd)
		}
	}
	// A file whose first image is itself reduced is an external .ovr,
	// and its mask is flagged reduced too.
	_, hasSub := ifds[0].entries[tagSubIFDs]
	ovrFile := !hasSub && c.subfileType(ifds[0]) == subfileReduced
	for i, ifd := range rest {
		t := c.subfileType(ifd)
		switch {
		case t&subfileReduced != 0 && t&subfileMask == 0:
			if len(f.levels)-1 >= maxOverviews {
				continue
			}
			im, err := c.parseImage(ifd, false)
			if err != nil {
				return fmt.Errorf("image %d (an overview): %w", i+1, err)
			}
			if im.bands != main.bands {
				continue // not an overview of this image
			}
			if im.format != main.format || im.bits != main.bits {
				return fmt.Errorf("image %d: an overview of a different sample type is not supported", i+1)
			}
			f.levels, f.masks = append(f.levels, im), append(f.masks, nil)
		case t&subfileMask != 0:
			im, err := c.parseImage(ifd, true)
			if err != nil || !im.isMask(main.bands) {
				continue // GDAL ignores a mask it cannot use
			}
			if f.masks[0] == nil && (t&subfileReduced == 0) != ovrFile &&
				im.width == main.width && im.height == main.height {
				f.masks[0] = im
				continue
			}
			if t&subfileReduced == 0 {
				continue
			}
			for lvl := 1; lvl < len(f.levels); lvl++ {
				if f.masks[lvl] == nil && f.levels[lvl].width == im.width && f.levels[lvl].height == im.height {
					f.masks[lvl] = im
					break
				}
			}
		}
	}
	return nil
}

// isMask reports whether im can serve as the mask of an image of bands
// bands: one band, or one per band, of 1- or 8-bit unsigned samples.
func (im *image) isMask(bands int) bool {
	return (im.bands == 1 || im.bands == bands) && im.format == sampleUint && (im.bits == 1 || im.bits == 8)
}

// nodataValues finds GDAL's dataset-level NODATA_VALUES metadata item.
var nodataValues = regexp.MustCompile(`<Item name="NODATA_VALUES"(?: domain="")?>([^<]*)</Item>`)

// checkNoDataValues refuses a file whose GDAL metadata holds one
// NODATA_VALUES entry per band: GDAL then masks a cell only where every
// band holds its value, which a single-band source cannot express.
func (c *container) checkNoDataValues(ifd rawIFD, bands int) error {
	e, ok := ifd.entries[tagGDALMetadata]
	if !ok {
		return nil
	}
	s, err := c.ascii(e)
	if err != nil {
		return nil // metadata is optional: GDAL would warn and go on
	}
	if m := nodataValues.FindStringSubmatch(s); m != nil && len(strings.Fields(m[1])) == bands {
		return errors.New("GDAL's NODATA_VALUES, a NoData value across all bands, is not supported")
	}
	return nil
}

// Levels returns the number of resolution levels: 1 for the full
// resolution, plus one per overview.
func (f *File) Levels() int { return len(f.levels) }

// Size returns the width and height in cells of a level. It panics if
// the level does not exist.
func (f *File) Size(level int) (width, height int) {
	im := f.level("Size", level)
	return im.width, im.height
}

// Bands returns the number of bands, which TIFF calls samples per pixel.
func (f *File) Bands() int { return f.levels[0].bands }

// NoData returns GDAL's NoData value for the file, and whether it has
// one. A source over a level with neither a mask nor a NoData value the
// sample type can hold is not Masked; see the package documentation for
// how cells compare with NoData.
func (f *File) NoData() (float64, bool) { return f.nodata, f.hasND }

// Georeferenced reports whether the file has a geotransform:
// ModelTransformation, or ModelPixelScale with one ModelTiepoint.
func (f *File) Georeferenced() bool { return f.geo.ok }

// Grid returns the grid of a level, as GDAL reports it. An overview has
// the full resolution's origin and its resolution scaled by the size
// ratio, as GDAL computes it. Without a geotransform (see Georeferenced)
// the grid of every level is GDAL's default: origin (0, 0), resolution
// (1, 1), the grid of the level's own cells. The CRS
// is the EPSG code of the GeoKeys, or empty if it is user-defined or
// absent. It panics if the level does not exist.
func (f *File) Grid(level int) raster.Grid {
	im := f.level("Grid", level)
	full := f.levels[0]
	g := raster.Grid{Width: im.width, Height: im.height, ResolutionX: 1, ResolutionY: 1, CRS: f.geo.crs}
	if f.geo.ok {
		g.OriginX, g.OriginY = f.geo.originX, f.geo.originY
		g.ResolutionX, g.ResolutionY = f.geo.resX, f.geo.resY
	}
	if level > 0 && f.geo.ok {
		g.ResolutionX *= float64(full.width) / float64(im.width)
		g.ResolutionY *= float64(full.height) / float64(im.height)
	}
	return g
}

func (f *File) level(fn string, level int) *image {
	if level < 0 || level >= len(f.levels) {
		panic(fmt.Sprintf("cog: File.%s: level %d of %d", fn, level, len(f.levels)))
	}
	return f.levels[level]
}

// DefaultCacheBytes is the decoded-block cache of a source whose
// SourceOptions.CacheBytes is 0: 64 MiB.
const DefaultCacheBytes = 64 << 20

// SourceOptions selects what a Source reads.
type SourceOptions struct {
	// Band is the band to read, from 0.
	Band int
	// Level is the resolution level, 0 for full resolution; see
	// File.Levels.
	Level int
	// CacheBytes bounds the memory of decoded blocks the source keeps,
	// so blocks that several windows touch are decoded once: 0 means
	// DefaultCacheBytes, a negative value no cache. The cache always
	// holds at least the most recent block.
	CacheBytes int64
}

// Source is an engine.RasterSource reading one band of one level of a
// File. It is safe for concurrent ReadWindow calls.
type Source struct {
	f     *File
	im    *image
	level int
	band  int
	nd    noData
	cache *cache
	mask  *Source // the level's transparency mask, which gives validity, or nil
}

var _ engine.RasterSource = (*Source)(nil)

// Source returns a source reading the band and level that opts select.
// It returns an error if either does not exist.
func (f *File) Source(opts SourceOptions) (*Source, error) {
	if opts.Level < 0 || opts.Level >= len(f.levels) {
		return nil, fmt.Errorf("cog: level %d, and the file has %d", opts.Level, len(f.levels))
	}
	im := f.levels[opts.Level]
	if opts.Band < 0 || opts.Band >= im.bands {
		return nil, fmt.Errorf("cog: band %d, and the file has %d", opts.Band, im.bands)
	}
	newSource := func(im *image, band int, nd noData) *Source {
		s := &Source{f: f, im: im, level: opts.Level, band: band, nd: nd}
		switch {
		case opts.CacheBytes == 0:
			s.cache = newCache(DefaultCacheBytes)
		case opts.CacheBytes > 0:
			s.cache = newCache(opts.CacheBytes)
		}
		return s
	}
	mask := f.masks[opts.Level]
	if mask == nil {
		return newSource(im, opts.Band, prepareNoData(f.nodata, f.hasND, im.format, im.bits)), nil
	}
	// With a mask, GDAL ignores NoData: the mask alone says which cells
	// are valid, those where it is not 0.
	s := newSource(im, opts.Band, noData{})
	band := 0
	if mask.bands > 1 {
		band = opts.Band
	}
	s.mask = newSource(mask, band, noData{set: true, cmp: 0})
	return s, nil
}

// Size returns the level's width and height.
func (s *Source) Size() (width, height int) { return s.im.width, s.im.height }

// Masked reports whether some cells may be invalid: the level has a
// transparency mask, or the file a NoData value that the band's sample
// type can hold.
func (s *Source) Masked() bool { return s.nd.set || s.mask != nil }

// Grid returns the grid of the level the source reads.
func (s *Source) Grid() raster.Grid { return s.f.Grid(s.level) }

// ReadWindow reads the region at (x, y) into dst; see
// engine.RasterSource. It decodes each block the region touches, or
// takes it from the cache. It returns ctx.Err() if ctx is done before a
// block is read, and errors from the file wrapped with the block that
// failed.
func (s *Source) ReadWindow(ctx context.Context, dst raster.Float32Raster, x, y int) error {
	if err := dst.Validate(); err != nil {
		panic(fmt.Sprintf("cog: Source.ReadWindow: dst: %v", err))
	}
	im := s.im
	if x < 0 || y < 0 || x > im.width-dst.Width || y > im.height-dst.Height {
		panic(fmt.Sprintf("cog: Source.ReadWindow: %d×%d region at (%d, %d) outside %d×%d raster",
			dst.Width, dst.Height, x, y, im.width, im.height))
	}
	if s.Masked() && dst.Valid == nil {
		panic("cog: Source.ReadWindow: the source is Masked and dst has no validity mask")
	}
	if err := s.read(ctx, dst, x, y, true); err != nil {
		return err
	}
	if s.mask != nil {
		if err := s.mask.read(ctx, dst, x, y, false); err != nil {
			return fmt.Errorf("the mask: %w", err)
		}
	}
	return nil
}

// read copies the blocks that the region at (x, y) touches into dst: the
// cells and their validity, or only the validity when values is false.
func (s *Source) read(ctx context.Context, dst raster.Float32Raster, x, y int, values bool) error {
	im := s.im
	x1, y1 := x+dst.Width, y+dst.Height
	for by := y / im.blockH; by*im.blockH < y1; by++ {
		for bx := x / im.blockW; bx*im.blockW < x1; bx++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			b, err := s.block(ctx, bx, by)
			if err != nil {
				return fmt.Errorf("cog: level %d, band %d, block (%d, %d): %w", s.level, s.band, bx, by, err)
			}
			s.copyBlock(dst, x, y, b, bx*im.blockW, by*im.blockH, values)
		}
	}
	return nil
}

// block returns the decoded block at (bx, by), through the cache.
func (s *Source) block(ctx context.Context, bx, by int) (*block, error) {
	idx := s.im.blockIndex(bx, by, s.band)
	load := func() (*block, error) { return s.f.c.decodeBlock(s.im, idx, by, s.band, s.nd) }
	if s.cache == nil {
		return load()
	}
	return s.cache.get(ctx, idx, load)
}

// copyBlock copies the part of b, whose top-left cell is (bx0, by0) in
// the raster, that lies inside dst, whose top-left is (x, y): its cells
// unless values is false, and their validity.
func (s *Source) copyBlock(dst raster.Float32Raster, x, y int, b *block, bx0, by0 int, values bool) {
	cx0, cx1 := max(x, bx0), min(x+dst.Width, bx0+b.w, s.im.width)
	cy0, cy1 := max(y, by0), min(y+dst.Height, by0+b.rows)
	n := cx1 - cx0
	for yy := cy0; yy < cy1; yy++ {
		src := (yy-by0)*b.w + cx0 - bx0
		d := (yy-y)*dst.Stride + cx0 - x
		if values {
			copy(dst.Data[d:d+n], b.vals[src:src+n])
		}
		if dst.Valid == nil {
			continue
		}
		if b.valid == nil {
			raster.MaskFillRange(dst.Valid, dst.ValidOffset+d, n, true)
		} else {
			raster.MaskCopyRange(dst.Valid, dst.ValidOffset+d, b.valid, src, n)
		}
	}
}
