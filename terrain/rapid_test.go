package terrain

import (
	"testing"

	"pgregory.net/rapid"

	"strata/internal/rapidsource"
)

// TestTerrainRelations checks the metamorphic relations of
// metamorphic_test.go as a property test: rapid searches for a case that
// breaks one and then shrinks it, reporting the draws that are left
// rather than the bytes of a fuzz input, and a seed that reruns it.
// Where fuzzing needs a corpus and time, this runs with go test.
//
// The relations are the fuzz target's, driven through the same body over
// fuzzdata.Source, so there is one implementation of each relation and
// two ways of searching for a case that breaks it.
//
// Both the number of cases and the seed are flags of rapid's:
//
//	go test ./terrain -run TestTerrainRelations -rapid.checks 10000
//	go test ./terrain -run TestTerrainRelations -rapid.seed 1234567890
func TestTerrainRelations(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		terrainRelations(rt, rapidsource.New(rt))
	})
}
