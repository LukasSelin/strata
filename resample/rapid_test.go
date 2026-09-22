package resample_test

import (
	"testing"

	"pgregory.net/rapid"

	"github.com/LukasSelin/strata/internal/rapidsource"
)

// TestResampleRelations checks the relations of metamorphic_test.go as a
// property test: rapid searches for a case that breaks one and shrinks it
// to a small counterexample. The body is the fuzz target's, so each
// relation has one implementation and two ways of searching.
//
//	go test ./resample -run TestResampleRelations -rapid.checks 10000
func TestResampleRelations(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		resampleRelations(rt, rapidsource.New(rt))
	})
}
