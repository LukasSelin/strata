//go:build race

package exec_test

// raceEnabled reports a -race build, in which sync.Pool drops a share of
// what is put in it on purpose.
const raceEnabled = true
