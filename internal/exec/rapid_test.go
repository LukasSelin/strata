package exec_test

import (
	"testing"

	"pgregory.net/rapid"

	"strata/internal/rapidsource"
)

// TestProcessRelations checks the relations of metamorphic_test.go as a
// property test. See terrain's TestTerrainRelations for what rapid adds
// to the fuzz target they share a body with: a search that needs no
// corpus, a failing case shrunk draw by draw, and a seed that reruns it.
//
// The relations hold for any kernel whose cells depend only on their
// neighbourhood, so what rapid searches here is the engine's own space:
// tiles, bands, workers, halos and layouts.
func TestProcessRelations(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		processRelations(rt, rapidsource.New(rt))
	})
}
