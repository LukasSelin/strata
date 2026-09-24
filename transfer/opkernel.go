package transfer

import (
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/opkernel"
)

// The kernels package graph lowers transfer's operations onto (DESIGN.md
// §55), built and checked by the constructors the entry points use, so a
// graph rejects the tables they reject with the same messages.
func init() {
	opkernel.Register("transfer.Reclass", func(o any) exec.Kernel {
		t := o.([2][]float32)
		return newReclassKernel("transfer.Reclass", t[0], t[1])
	})
	opkernel.Register("transfer.Lookup", func(o any) exec.Kernel {
		t := o.([2][]float32)
		return newLookupKernel("transfer.Lookup", t[0], t[1])
	})
	opkernel.Register("transfer.Rescale", func(o any) exec.Kernel {
		k := o.([2]float32)
		return rescaleKernel{k[0], k[1]}
	})
	opkernel.Register("transfer.RescaleRange", func(o any) exec.Kernel {
		k := o.([4]float32)
		a, b := rescaleCoeffs("transfer.RescaleRange", k[0], k[1], k[2], k[3])
		return rescaleKernel{a, b}
	})
}
