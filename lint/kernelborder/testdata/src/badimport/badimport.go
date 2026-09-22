// want package:"kernel"

//strata:kernel
package badimport

import (
	"context" // want `kernel package imports context: kernels take spans`
	"sync"    // want `kernel package imports sync: kernels take spans`

	"rep" // want `kernel package imports rep: kernels take spans`
)

var (
	_ context.Context
	_ sync.Mutex
	_ rep.Raster
)
