package engine_test

import (
	"testing"

	"go.uber.org/goleak"
)

// TestMain fails the package if any test leaves a goroutine behind.
// Sources, sinks and RawFile start none of their own; the tests that call
// them from many goroutines must join every one, even when a read or
// write fails.
func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}
