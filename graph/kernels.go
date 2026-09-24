package graph

// The operation packages register their kernels with internal/opkernel
// in init functions (DESIGN.md §55). focal and terrain are imported for
// their option types anyway; algebra and transfer have none the graph
// needs, so they are imported here for the registration alone.
import (
	_ "github.com/LukasSelin/strata/algebra"
	_ "github.com/LukasSelin/strata/transfer"
)
