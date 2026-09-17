//go:build race

package chunked_test

// raceEnabled reports a -race build, whose file IO allocates.
const raceEnabled = true
