package transfer_test

import (
	"testing"

	"pgregory.net/rapid"

	"github.com/LukasSelin/strata/internal/rapidsource"
)

// TestTransferRelations checks the relations of metamorphic_test.go as a
// property test. See terrain's TestTerrainRelations for what rapid adds
// to the fuzz target they share a body with: a search that needs no
// corpus, a failing case shrunk draw by draw, and a seed that reruns it.
func TestTransferRelations(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		transferRelations(rt, rapidsource.New(rt))
	})
}
