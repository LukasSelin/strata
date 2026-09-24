// Package opkernel lets package graph build the kernels that live,
// unexported, in the operation packages (DESIGN.md §55).
//
// A graph lowers each operation onto the same kernel its own entry point
// runs, so that a graph and the separate calls write the same bits. Those
// kernels are unexported, and stay so: Kernel is internal (§52,
// "Publishing Kernel: the decision"), and an exported constructor
// returning one would put the contract in the public API by the back
// door. So each operation package registers its constructors here in an
// init function, by name, and package graph, which imports the operation
// packages for their option types, looks them up. Nothing outside the
// module can import this package, so the registry is not an extension
// point either.
package opkernel

import (
	"fmt"
	"sync"

	"github.com/LukasSelin/strata/internal/exec"
)

var (
	mu    sync.RWMutex
	ctors = map[string]func(opts any) exec.Kernel{}
)

// Register makes f the constructor for the operation name, such as
// "terrain.Slope". f receives the options New is given and must panic, as
// the operation's own entry point does, on options it rejects. It panics
// if name is already registered.
func Register(name string, f func(opts any) exec.Kernel) {
	mu.Lock()
	defer mu.Unlock()
	if _, ok := ctors[name]; ok {
		panic(fmt.Sprintf("opkernel: %s is registered twice", name))
	}
	ctors[name] = f
}

// New returns the kernel of the operation name built from opts. It panics
// if no package registered name, which is a missing import in package
// graph, not a caller's mistake.
func New(name string, opts any) exec.Kernel {
	mu.RLock()
	f, ok := ctors[name]
	mu.RUnlock()
	if !ok {
		panic(fmt.Sprintf("opkernel: no kernel is registered for %s", name))
	}
	return f(opts)
}
