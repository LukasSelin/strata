// A separate module, so the harness stays out of the library's build,
// vet, test and lint runs. It depends on strata the way a user would,
// through its published import path, resolved to the checkout next door.
module strata-acceptance

go 1.27.0

require github.com/LukasSelin/strata v0.0.0

replace github.com/LukasSelin/strata => ..
