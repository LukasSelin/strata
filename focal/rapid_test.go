package focal_test

import (
	"testing"

	"pgregory.net/rapid"

	"github.com/LukasSelin/strata/internal/rapidsource"
)

// TestFocalRelations checks the metamorphic relations of
// metamorphic_test.go as a property test: rapid searches for a case that
// breaks one and shrinks it, reporting the draws that are left and a seed
// that reruns it. It drives the same body as FuzzFocalRelations, over
// fuzzdata.Source, so each relation has one implementation.
//
//	go test ./focal -run TestFocalRelations -rapid.checks 10000
func TestFocalRelations(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		focalRelations(rt, rapidsource.New(rt))
	})
}
