package cog

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// TIFF and GeoTIFF tags this reader uses.
const (
	tagNewSubfileType   = 254
	tagImageWidth       = 256
	tagImageLength      = 257
	tagBitsPerSample    = 258
	tagCompression      = 259
	tagStripOffsets     = 273
	tagSamplesPerPixel  = 277
	tagRowsPerStrip     = 278
	tagStripByteCounts  = 279
	tagPlanarConfig     = 284
	tagPredictor        = 317
	tagTileWidth        = 322
	tagTileLength       = 323
	tagTileOffsets      = 324
	tagTileByteCounts   = 325
	tagSampleFormat     = 339
	tagModelPixelScale  = 33550
	tagModelTiepoint    = 33922
	tagModelTransform   = 34264
	tagGeoKeyDirectory  = 34735
	tagGeoDoubleParams  = 34736
	tagGeoASCIIParams   = 34737
	tagGDALNoData       = 42113
	subfileReduced      = 1 // NewSubfileType: a reduced-resolution image (an overview)
	subfileMask         = 4 // NewSubfileType: a transparency mask
	planarChunky        = 1
	planarSeparate      = 2
	sampleUint          = 1
	sampleInt           = 2
	sampleFloat         = 3
	predictorNone       = 1
	predictorHorizontal = 2
	predictorFloat      = 3
)

// Compression schemes.
const (
	compressionNone     = 1
	compressionLZW      = 5
	compressionJPEGOld  = 6
	compressionJPEG     = 7
	compressionDeflate  = 8
	compressionPackBits = 32773
	compressionDeflate2 = 32946 // the pre-standard Adobe Deflate code
	compressionLERC     = 34887
	compressionLZMA     = 34925
	compressionZSTD     = 50000
	compressionWebP     = 50001
	compressionJXL      = 50002
)

func compressionName(c uint64) string {
	switch c {
	case compressionNone:
		return "none"
	case compressionLZW:
		return "LZW"
	case compressionJPEGOld, compressionJPEG:
		return "JPEG"
	case compressionDeflate, compressionDeflate2:
		return "Deflate"
	case compressionPackBits:
		return "PackBits"
	case compressionLERC:
		return "LERC"
	case compressionLZMA:
		return "LZMA"
	case compressionZSTD:
		return "ZSTD"
	case compressionWebP:
		return "WebP"
	case compressionJXL:
		return "JPEG XL"
	}
	return "code " + strconv.FormatUint(c, 10)
}

// maxBlockBytes bounds one tile or strip, decoded, and maxBlockCells its
// cells, which become 4-byte float32s: 256 MiB either way. GDAL writes
// 512² tiles, a few MiB at most; a stripped file larger than this must
// have more than one strip.
const (
	maxBlockBytes = 1 << 28
	maxBlockCells = maxBlockBytes / 4
)

// image is one IFD, understood: the layout of one resolution level.
type image struct {
	width, height int
	bands         int // samples per pixel
	format        int // sampleUint, sampleInt or sampleFloat
	bytes         int // bytes per sample: 1, 2, 4 or 8
	compression   uint64
	predictor     int
	planar        int

	// Blocks are tiles, or strips as wide as the image. Strips' last
	// block may hold fewer rows than blockH; tiles are always whole.
	tiled          bool
	blockW, blockH int
	across, down   int // blocks per row and per column of one band plane
	offsets        []uint64
	byteCounts     []uint64
}

// parseImage reads the tags of a full-resolution image or an overview.
func (c *container) parseImage(ifd rawIFD) (*image, error) {
	im := &image{}
	w, err := c.one(ifd, tagImageWidth, 0, true)
	if err != nil {
		return nil, err
	}
	h, err := c.one(ifd, tagImageLength, 0, true)
	if err != nil {
		return nil, err
	}
	if w == 0 || h == 0 || w > math.MaxInt32 || h > math.MaxInt32 {
		return nil, fmt.Errorf("image size %d×%d is not supported", w, h)
	}
	im.width, im.height = int(w), int(h)

	spp, err := c.one(ifd, tagSamplesPerPixel, 1, false)
	if err != nil {
		return nil, err
	}
	if spp == 0 || spp > 1<<16 {
		return nil, fmt.Errorf("%d samples per pixel", spp)
	}
	im.bands = int(spp)

	bits, err := c.same(ifd, tagBitsPerSample, 1, im.bands, "bits per sample")
	if err != nil {
		return nil, err
	}
	format, err := c.same(ifd, tagSampleFormat, sampleUint, im.bands, "sample formats")
	if err != nil {
		return nil, err
	}
	switch format {
	case sampleUint, sampleInt:
		if bits != 8 && bits != 16 && bits != 32 {
			return nil, fmt.Errorf("%d-bit integer samples are not supported (8, 16 and 32 are)", bits)
		}
	case sampleFloat:
		if bits != 32 && bits != 64 {
			return nil, fmt.Errorf("%d-bit floating-point samples are not supported (32 and 64 are)", bits)
		}
	default:
		return nil, fmt.Errorf("sample format %d is not supported (1 unsigned, 2 signed and 3 float are)", format)
	}
	im.format, im.bytes = int(format), int(bits/8) // #nosec G115 -- both checked above

	if im.compression, err = c.one(ifd, tagCompression, compressionNone, false); err != nil {
		return nil, err
	}
	switch im.compression {
	case compressionNone, compressionLZW, compressionDeflate, compressionDeflate2,
		compressionPackBits, compressionZSTD:
	default:
		return nil, fmt.Errorf("%s compression is not supported (none, LZW, Deflate, PackBits and ZSTD are)",
			compressionName(im.compression))
	}

	pred, err := c.one(ifd, tagPredictor, predictorNone, false)
	if err != nil {
		return nil, err
	}
	switch pred {
	case predictorNone, predictorHorizontal:
	case predictorFloat:
		if im.format != sampleFloat {
			return nil, errors.New("the floating-point predictor on integer samples")
		}
	default:
		return nil, fmt.Errorf("predictor %d is not supported (1, 2 and 3 are)", pred)
	}
	im.predictor = int(pred)

	planar, err := c.one(ifd, tagPlanarConfig, planarChunky, false)
	if err != nil {
		return nil, err
	}
	if planar != planarChunky && planar != planarSeparate {
		return nil, fmt.Errorf("planar configuration %d", planar)
	}
	im.planar = int(planar)

	offTag, countTag := uint16(tagStripOffsets), uint16(tagStripByteCounts)
	if _, ok := ifd.entries[tagTileWidth]; ok {
		im.tiled = true
		offTag, countTag = tagTileOffsets, tagTileByteCounts
		tw, err := c.one(ifd, tagTileWidth, 0, true)
		if err != nil {
			return nil, err
		}
		th, err := c.one(ifd, tagTileLength, 0, true)
		if err != nil {
			return nil, err
		}
		if tw == 0 || th == 0 || tw > 1<<16 || th > 1<<16 {
			return nil, fmt.Errorf("tile size %d×%d", tw, th)
		}
		im.blockW, im.blockH = int(tw), int(th)
	} else {
		rps, err := c.one(ifd, tagRowsPerStrip, h, false)
		if err != nil {
			return nil, err
		}
		if rps == 0 {
			return nil, errors.New("0 rows per strip")
		}
		im.blockW, im.blockH = im.width, int(min(rps, h))
	}
	if cells := int64(im.blockW) * int64(im.blockH); cells > maxBlockCells ||
		cells*int64(im.blockSamples()*im.bytes) > maxBlockBytes {
		return nil, fmt.Errorf("a %d×%d block of %d-byte samples is over %d MiB",
			im.blockW, im.blockH, im.blockSamples()*im.bytes, maxBlockBytes>>20)
	}
	im.across = (im.width + im.blockW - 1) / im.blockW
	im.down = (im.height + im.blockH - 1) / im.blockH
	blocks := int64(im.across) * int64(im.down)
	if im.planar == planarSeparate {
		blocks *= int64(im.bands)
	}
	if blocks > maxTagValues {
		return nil, fmt.Errorf("%d blocks, more than %d", blocks, maxTagValues)
	}
	if im.offsets, err = c.list(ifd, offTag, int(blocks)); err != nil {
		return nil, err
	}
	if im.byteCounts, err = c.list(ifd, countTag, int(blocks)); err != nil {
		return nil, err
	}
	return im, nil
}

// blockSamples is the number of samples per cell stored in one block:
// every band's for chunky data, one for separate planes.
func (im *image) blockSamples() int {
	if im.planar == planarSeparate {
		return 1
	}
	return im.bands
}

// blockRows is the number of rows block row by holds: blockH, except for
// the last strip, which stops at the image's last row.
func (im *image) blockRows(by int) int {
	if im.tiled {
		return im.blockH
	}
	return min(im.blockH, im.height-by*im.blockH)
}

// blockIndex is the index into offsets of the block at (bx, by) holding
// band.
func (im *image) blockIndex(bx, by, band int) int {
	i := by*im.across + bx
	if im.planar == planarSeparate {
		i += band * im.across * im.down
	}
	return i
}

// one reads a tag holding a single unsigned integer, or returns def if
// the tag is absent and not required.
func (c *container) one(ifd rawIFD, tag uint16, def uint64, required bool) (uint64, error) {
	e, ok := ifd.entries[tag]
	if !ok {
		if required {
			return 0, fmt.Errorf("tag %d is missing", tag)
		}
		return def, nil
	}
	if e.count == 0 {
		return 0, fmt.Errorf("tag %d has no value", tag)
	}
	// Only the first value counts. An inline field or an offset both
	// start with it, so reading one value from either finds it.
	e.count = 1
	v, err := c.uints(e)
	if err != nil {
		return 0, err
	}
	return v[0], nil
}

// same reads a per-sample tag whose values must all be equal, such as
// BitsPerSample, or returns def if the tag is absent.
func (c *container) same(ifd rawIFD, tag uint16, def uint64, bands int, what string) (uint64, error) {
	e, ok := ifd.entries[tag]
	if !ok {
		return def, nil
	}
	v, err := c.uints(e)
	if err != nil {
		return 0, err
	}
	if len(v) == 0 {
		return 0, fmt.Errorf("tag %d has no value", tag)
	}
	for _, x := range v[:min(len(v), bands)] {
		if x != v[0] {
			return 0, fmt.Errorf("bands with different %s (%v) are not supported", what, v)
		}
	}
	return v[0], nil
}

// list reads a tag that must hold exactly n unsigned integers.
func (c *container) list(ifd rawIFD, tag uint16, n int) ([]uint64, error) {
	e, ok := ifd.entries[tag]
	if !ok {
		return nil, fmt.Errorf("tag %d is missing", tag)
	}
	if e.count != uint64(n) { // #nosec G115 -- n is a block count, positive
		return nil, fmt.Errorf("tag %d has %d values for %d blocks", tag, e.count, n)
	}
	return c.uints(e)
}

// subfileType returns an IFD's NewSubfileType, 0 if absent or unreadable.
func (c *container) subfileType(ifd rawIFD) uint64 {
	v, err := c.one(ifd, tagNewSubfileType, 0, false)
	if err != nil {
		return 0
	}
	return v
}

// noData reads GDAL's NoData tag, an ASCII number.
func (c *container) noData(ifd rawIFD) (float64, bool, error) {
	e, ok := ifd.entries[tagGDALNoData]
	if !ok {
		return 0, false, nil
	}
	s, err := c.ascii(e)
	if err != nil {
		return 0, false, err
	}
	s = strings.TrimSpace(s)
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		var ne *strconv.NumError
		if !errors.As(err, &ne) || !errors.Is(ne.Err, strconv.ErrRange) {
			return 0, false, fmt.Errorf("GDAL_NODATA %q is not a number", s)
		}
		// Out of float64's range: ParseFloat has returned ±Inf, which
		// is what GDAL's CPLAtof would give too.
	}
	return v, true, nil
}
