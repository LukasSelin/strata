// Command kernelborder runs the kernelborder analyzer (DESIGN.md §12,
// §14). It speaks go vet's tool protocol, so CI runs it over strata as
//
//	go vet -vettool=/path/to/kernelborder ./...
//
// once per build configuration, since the SIMD kernels only compile with
// GOEXPERIMENT=simd.
package main

import (
	"golang.org/x/tools/go/analysis/singlechecker"

	"strata-lint/kernelborder"
)

func main() { singlechecker.Main(kernelborder.Analyzer) }
