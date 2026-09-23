// A separate module, so the format libraries it needs (TIFF LZW, zstd)
// stay out of strata's own go.mod: formats are adapters, not the core
// (DESIGN.md §34, docs/adr/0002-cog-adapter.md). It depends on strata the
// way a user would, through its published import path, resolved to the
// checkout next door until strata is tagged.
module github.com/LukasSelin/strata/cog

go 1.27.0

require (
	github.com/LukasSelin/strata v0.0.0
	github.com/klauspost/compress v1.20.0
)

replace github.com/LukasSelin/strata => ..
