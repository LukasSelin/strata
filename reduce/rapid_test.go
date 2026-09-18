package reduce_test

import (
	"testing"

	"pgregory.net/rapid"

	"github.com/LukasSelin/strata/internal/rapidsource"
)

// TestReduceRelations checks the relations of metamorphic_test.go as a
// property test. It shares its body with FuzzReduceRelations: fuzzing
// searches deeper for as long as it is given, and rapid needs no corpus,
// shrinks a failing case draw by draw and prints what is left along with
// a seed that reruns it (DESIGN.md §39).
//
// What it searches here is the reduction's own space: values, masks,
// layouts, tiles, bands, workers and which of the plain, tiled and
// chunked paths each side of a relation takes.
func TestReduceRelations(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		reduceRelations(rt, rapidsource.New(rt))
	})
}
