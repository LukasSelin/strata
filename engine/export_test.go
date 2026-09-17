package engine

// SetRawCallBytes sets the size of grouped raw reads and writes and
// returns a function that restores it.
func SetRawCallBytes(n int) (restore func()) {
	old := rawCallBytes
	rawCallBytes = n
	return func() { rawCallBytes = old }
}
