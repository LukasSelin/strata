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
	tagPhotometric      = 262
	tagFillOrder        = 266
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
	tagSubIFDs          = 330
	tagSampleFormat     = 339
	tagModelPixelScale  = 33550
	tagModelTiepoint    = 33922
	tagModelTransform   = 34264
	tagGeoKeyDirectory  = 34735
	tagGeoDoubleParams  = 34736
	tagGeoASCIIParams   = 34737
	tagGDALMetadata     = 42112
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

// Photometric interpretations that GDAL converts to RGB(A) through
// libtiff when they have at most 8 bits per sample, rather than reading
// the samples as stored.
const (
	photometricSeparated = 5 // CMYK
	photometricYCbCr     = 6
	photometricCIELab    = 8
	photometricLogL      = 32844
	photometricLogLuv    = 32845
)

// Compression schemes.
const (
	compressionNone     = 1
	compressionCCITTRLE = 2
	compressionCCITT3   = 3
	compressionCCITT4   = 4
	compressionLZW      = 5
	compressionJPEGOld  = 6
	compressionJPEG     = 7
	compressionDeflate  = 8
	compressionPackBits = 32773
	compressionNeXT     = 32766
	compressionThunder  = 32809
	compressionDeflate2 = 32946 // the pre-standard Adobe Deflate code
	compressionSGILog   = 34676
	compressionSGILog24 = 34677
	compressionJPEG2000 = 34712
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
	case compressionCCITTRLE, compressionCCITT3, compressionCCITT4:
		return "CCITT fax"
	case compressionNeXT:
		return "NeXT"
	case compressionThunder:
		return "ThunderScan"
	case compressionSGILog, compressionSGILog24:
		return "SGILog"
	case compressionJPEG2000:
		return "JPEG 2000"
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

// image is one IFD, understood: the layout of one resolution level, or
// of a transparency mask.
type image struct {
	width, height int
	bands         int // samples per pixel
	format        int // sampleUint, sampleInt or sampleFloat
	bits          int // bits per sample: 8, 16, 32 or 64, or 1 in a mask
	bytes         int // bytes per sample: 1, 2, 4 or 8; 0 for 1 bit
	compression   uint64
	predictor     int
	lsbFirst      bool // FillOrder 2: libtiff reverses each byte's bits before decoding
	planar        int

	// Blocks are tiles, or strips as wide as the image. Strips' last
	// block may hold fewer rows than blockH; tiles are always whole.
	tiled          bool
	blockW, blockH int
	across, down   int // blocks per row and per column of one band plane
	offsets        []uint64
	byteCounts     []uint64
}

// parseImage reads the tags of a full-resolution image or an overview,
// or with mask set of a transparency mask, which may also hold 1-bit
// samples.
func (c *container) parseImage(ifd rawIFD, mask bool) (*image, error) {
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
		if mask && bits == 1 && format == sampleUint {
			break
		}
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
	im.format, im.bits, im.bytes = int(format), int(bits), int(bits/8) // #nosec G115 -- all checked above

	photometric, err := c.one(ifd, tagPhotometric, 1, false)
	if err != nil {
		return nil, err
	}
	switch photometric {
	case photometricYCbCr:
		// Subsampled YCbCr is laid out in macro-blocks, and GDAL
		// converts even unsubsampled YCbCr to RGB.
		return nil, errors.New("YCbCr colour is not supported: GDAL converts it to RGB")
	case photometricSeparated, photometricCIELab, photometricLogL, photometricLogLuv:
		if bits <= 8 && !mask {
			return nil, fmt.Errorf("photometric interpretation %d at %d bits is not supported: GDAL converts it to RGBA",
				photometric, bits)
		}
	}

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

	fill, err := c.one(ifd, tagFillOrder, 1, false)
	if err != nil {
		return nil, err
	}
	im.lsbFirst = fill == 2

	pred, err := c.one(ifd, tagPredictor, predictorNone, false)
	if err != nil {
		return nil, err
	}
	switch pred {
	case predictorNone:
	case predictorHorizontal:
		if im.bits == 1 {
			return nil, errors.New("the horizontal predictor on 1-bit samples")
		}
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
		// Early tiled files keep their tiles in StripOffsets and
		// StripByteCounts, which libtiff accepts.
		if _, ok := ifd.entries[tagTileOffsets]; ok {
			offTag, countTag = tagTileOffsets, tagTileByteCounts
		}
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
		cells*int64(im.blockSamples())*int64(im.bits) > 8*maxBlockBytes {
		return nil, fmt.Errorf("a %d×%d block of %d %d-bit samples per cell is over %d MiB",
			im.blockW, im.blockH, im.blockSamples(), im.bits, maxBlockBytes>>20)
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
	if _, ok := ifd.entries[countTag]; !ok && im.compression == compressionNone {
		// libtiff computes missing byte counts of uncompressed data
		// from the layout.
		im.byteCounts = make([]uint64, blocks)
		for i := range im.byteCounts {
			by := i % (im.across * im.down) / im.across
			im.byteCounts[i] = uint64(im.rowBytes() * im.blockRows(by)) // #nosec G115 -- positive, bounded above
		}
		return im, nil
	}
	if im.byteCounts, err = c.list(ifd, countTag, int(blocks)); err != nil {
		return nil, err
	}
	return im, nil
}

// rowBytes is the size of one row of a block, decoded: rows start on a
// byte boundary, which matters only for 1-bit samples.
func (im *image) rowBytes() int {
	return (im.blockW*im.blockSamples()*im.bits + 7) / 8
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

// list reads a tag that should hold n unsigned integers, one per block.
// As libtiff does, it ignores values past n and pads a short list with
// zeros, which make blocks with a 0 byte count, and so absent ones.
func (c *container) list(ifd rawIFD, tag uint16, n int) ([]uint64, error) {
	e, ok := ifd.entries[tag]
	if !ok {
		return nil, fmt.Errorf("tag %d is missing", tag)
	}
	if e.count > uint64(n) { // #nosec G115 -- n is a block count, positive
		e.count = uint64(n) // #nosec G115 -- as above
		if e.inline != nil {
			e.inline = e.inline[:n*typeSize(e.typ)]
		}
	}
	v, err := c.uints(e)
	if err != nil {
		return nil, err
	}
	if len(v) < n {
		v = append(v, make([]uint64, n-len(v))...)
	}
	return v, nil
}

// subfileType returns an IFD's NewSubfileType, 0 if absent or unreadable.
func (c *container) subfileType(ifd rawIFD) uint64 {
	v, err := c.one(ifd, tagNewSubfileType, 0, false)
	if err != nil {
		return 0
	}
	return v
}

// noData reads GDAL's NoData tag, an ASCII number. An empty tag is no
// NoData value, as in GDAL.
func (c *container) noData(ifd rawIFD) (float64, bool, error) {
	e, ok := ifd.entries[tagGDALNoData]
	if !ok {
		return 0, false, nil
	}
	s, err := c.ascii(e)
	if err != nil {
		return 0, false, err
	}
	if s == "" {
		return 0, false, nil
	}
	return atofM(s), true, nil
}

// atofM parses s as GDAL's CPLAtofM does (port/cpl_strtod.cpp), which is
// how GDAL reads GDAL_NODATA: the longest number at the start of s after
// white space, with a comma as the decimal point if one comes before any
// full stop; MSVC's spellings of NaN and infinity ("1.#QNAN", "-1.#INF");
// "inf", "infinity" and "nan" in any case; and 0 when nothing parses.
func atofM(s string) float64 {
	point := byte('.')
	for i := 0; i < len(s) && i < 50; i++ {
		if s[i] == ',' {
			point = ','
			break
		}
		if s[i] == '.' {
			break
		}
	}
	s = strings.TrimLeft(s, " \t\n\v\f\r")
	s = strings.TrimRight(s, " \t\n\v\f\r")
	switch {
	case strings.HasPrefix(s, "-1.#QNAN"), strings.HasPrefix(s, "-1.#IND"),
		strings.HasPrefix(s, "1.#QNAN"), strings.HasPrefix(s, "1.#SNAN"):
		return math.NaN()
	case hasPrefixFold(s, "-1.#INF"):
		return math.Inf(-1)
	case hasPrefixFold(s, "1.#INF"):
		return math.Inf(1)
	}
	s = strings.TrimPrefix(s, "+")
	end := 0
	for end < len(s) && (s[end] >= '0' && s[end] <= '9' || s[end] == point ||
		s[end] == '+' || s[end] == '-' || s[end] == 'e' || s[end] == 'E') {
		end++
	}
	num := s[:end]
	if point == ',' {
		num = strings.ReplaceAll(num, ",", ".")
	}
	// The longest prefix that is a number, as fast_float finds it.
	for n := len(num); n > 0; n-- {
		if !isFloatSyntax(num[:n]) {
			continue
		}
		v, err := strconv.ParseFloat(num[:n], 64)
		if err == nil || errors.Is(err, strconv.ErrRange) {
			return v // out of range: ±Inf or 0, as fast_float gives
		}
	}
	switch {
	case hasPrefixFold(s, "-inf"):
		return math.Inf(-1)
	case hasPrefixFold(s, "inf"):
		return math.Inf(1)
	case hasPrefixFold(s, "nan"):
		return math.NaN()
	}
	return 0
}

// isFloatSyntax reports whether s is a decimal number in C's syntax:
// an optional minus sign, digits with an optional point, at least one
// digit, and an optional exponent. It excludes what strconv.ParseFloat
// accepts and C does not: underscores, hexadecimal, "inf" and "nan".
func isFloatSyntax(s string) bool {
	i := 0
	if i < len(s) && s[i] == '-' {
		i++
	}
	digits := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i, digits = i+1, digits+1
	}
	if i < len(s) && s[i] == '.' {
		i++
		for i < len(s) && s[i] >= '0' && s[i] <= '9' {
			i, digits = i+1, digits+1
		}
	}
	if digits == 0 {
		return false
	}
	if i < len(s) && (s[i] == 'e' || s[i] == 'E') {
		i++
		if i < len(s) && (s[i] == '+' || s[i] == '-') {
			i++
		}
		exp := 0
		for i < len(s) && s[i] >= '0' && s[i] <= '9' {
			i, exp = i+1, exp+1
		}
		if exp == 0 {
			return false
		}
	}
	return i == len(s)
}

func hasPrefixFold(s, prefix string) bool {
	return len(s) >= len(prefix) && strings.EqualFold(s[:len(prefix)], prefix)
}
