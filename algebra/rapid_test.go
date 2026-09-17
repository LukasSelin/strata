package algebra_test

import (
	"testing"

	"pgregory.net/rapid"

	"strata/internal/rapidsource"
)

// TestAlgebraRelations checks the relations of metamorphic_test.go as a
// property test. See terrain's TestTerrainRelations for what rapid adds
// to the fuzz target they share a body with: a search that needs no
// corpus, a failing case shrunk draw by draw, and a seed that reruns it.
func TestAlgebraRelations(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		algebraRelations(rt, rapidsource.New(rt))
	})
}
