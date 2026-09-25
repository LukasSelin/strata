// A separate module, because the core module cannot import the zarr
// adapter without taking on its format library (DESIGN.md §34). Like
// benchmarks/cog, it depends on strata and the adapter the way a user
// would, through their published import paths, resolved to the checkouts
// next door.
module strata-zarrbench

go 1.27.0

replace github.com/LukasSelin/strata => ../..

replace github.com/LukasSelin/strata/zarr => ../../zarr

require (
	github.com/LukasSelin/strata v0.0.0
	github.com/LukasSelin/strata/zarr v0.0.0
	github.com/LukasSelin/zarr v0.3.0
)
