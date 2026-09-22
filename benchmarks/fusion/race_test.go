//go:build race

package fusion_test

// raceEnabled reports a -race build, in which sync.Pool drops a share of
// what is put in it on purpose, so pooled scratch is reallocated at random.
const raceEnabled = true
