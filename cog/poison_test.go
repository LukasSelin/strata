package cog

// Every test in the package runs with released block buffers poisoned;
// see poisonVals.
func init() { poisonVals = true }
