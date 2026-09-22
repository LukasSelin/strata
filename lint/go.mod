// A separate module, so golang.org/x/tools stays out of the library's
// go.mod: the linters here check strata's source, and nothing that
// imports strata needs them.
module strata-lint

go 1.27.0

require golang.org/x/tools v0.50.0

require (
	golang.org/x/mod v0.41.0 // indirect
	golang.org/x/sync v0.23.0 // indirect
)
