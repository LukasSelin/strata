// A separate module, so the Zarr library it wraps stays out of strata's
// own go.mod: formats are adapters, not the core (DESIGN.md §34,
// docs/adr/0003-zarr-adapter.md). It depends on strata the way a user
// would, through its published import path, resolved to the checkout next
// door until strata is tagged.
module github.com/LukasSelin/strata/zarr

go 1.27.0

require (
	github.com/LukasSelin/strata v0.0.0
	github.com/LukasSelin/zarr v0.3.0
)

replace github.com/LukasSelin/strata => ..
