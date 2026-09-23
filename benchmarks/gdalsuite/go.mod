// A separate module, like benchmarks/cog, because the whole-flow cases
// read COGs and the core module cannot import cog (DESIGN.md §34). It
// depends on strata and cog the way a user would, through their
// published import paths, resolved to the checkouts next door. That also
// means it cannot reach strata's internal kernel switches: the scalar
// rows come from a second build without GOEXPERIMENT=simd, which is what
// an ordinary `go build` gets.
module strata-gdalsuite

go 1.27.0

replace github.com/LukasSelin/strata => ../..

replace github.com/LukasSelin/strata/cog => ../../cog

require (
	github.com/LukasSelin/strata v0.0.0
	github.com/LukasSelin/strata/cog v0.0.0
)

require github.com/klauspost/compress v1.20.0 // indirect
