package exec

// SetBandCells sets the band size target and returns a function that
// restores it, so tests can force one-row bands.
func SetBandCells(n int) (restore func()) {
	old := bandCells
	bandCells = n
	return func() { bandCells = old }
}
