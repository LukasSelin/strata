// A separate module, because the core module cannot import cog without
// taking on its format libraries (DESIGN.md §34). Like acceptance/, it
// depends on strata and cog the way a user would, through their published
// import paths, resolved to the checkouts next door.
module strata-cogbench

go 1.27.0

replace github.com/LukasSelin/strata => ../..

replace github.com/LukasSelin/strata/cog => ../../cog

require (
	github.com/LukasSelin/strata v0.0.0
	github.com/LukasSelin/strata/cog v0.0.0
)

require github.com/klauspost/compress v1.20.0 // indirect
