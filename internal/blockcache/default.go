package blockcache

import "github.com/LukasSelin/strata/raster"

// The cache an adapter's source makes when its CacheBytes option is 0
// holds DefaultRows rows of the raster's blocks, but never less than
// DefaultBytes nor more than MaxDefaultBytes. cog and zarr export these
// as their DefaultCache* constants, whose doc comment in cog gives the
// reasoning (benchmarks/cog/RESULTS.md, "The cache").
const (
	DefaultRows     = 8
	DefaultBytes    = 64 << 20
	MaxDefaultBytes = 1 << 30
)

// DefaultLimit returns the default cache of a source over a raster width
// cells wide in blocks of bw by bh cells: DefaultRows rows of decoded
// blocks, clamped to [DefaultBytes, MaxDefaultBytes]. A block counts as
// the adapters' block sizes count it: 4 bytes a value, a full validity
// mask and 64 bytes over.
func DefaultLimit(width, bw, bh int) int64 {
	cells := int64(bw) * int64(bh)
	perBlock := 4*cells + 8*int64(raster.MaskWords(int(cells))) + 64
	across := int64((width + bw - 1) / bw)
	if across > MaxDefaultBytes/(DefaultRows*perBlock) {
		return MaxDefaultBytes // and no overflow on the way
	}
	return min(max(DefaultRows*across*perBlock, DefaultBytes), MaxDefaultBytes)
}
