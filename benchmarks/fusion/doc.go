// Package fusion measures what register-level operation fusion
// (DESIGN.md §29) would add on top of the tile-level Pipeline that exists
// today (§52), before anyone builds a generator for it.
//
// The workload is §52's own example, a six-factor product
// dst = ((((a0·a1)·a2)·a3)·a4)·a5, computed three ways through the engine:
//
//   - MulChained: five algebra.MulTiled calls over whole rasters, the way
//     a caller writes it today. 60 B/cell.
//   - MulPipeline: the five multiplies as one exec.Pipeline. Each tile is
//     read once and the intermediates live in the worker's scratch, so
//     28 B/cell reach the rasters, plus scratch traffic the engine's
//     counter does not see.
//   - MulFused: one hand-written kernel (Fused) that loads six values,
//     multiplies them in registers and stores once. 28 B/cell and no
//     scratch at all. It is what a fusion generator would emit for this
//     chain.
//
// All three are bit-identical (TestFormsAgree): the fused loop multiplies
// in the chain's order and rounds every product.
//
//	Benchmark<Form>/size=<1024|4096|8192>/mask=<off|on>/backend=<scalar|simd>/workers=<W>/tiles=<strips|256x256>
//
// W is 1, one per physical core and one per logical CPU (suite.Workers);
// scalar runs one worker only, as in the engine category. GB/s is the
// form's own bytesPerCell, not the ideal, so MulChained's GB/s is the
// traffic it really asks for while Mcells/s compares the three directly.
//
// The comparison that answers §29 is MulFused against MulPipeline at the
// same case. The pipeline against the chain is §52's result, measured
// again here as a control.
//
// Factors are uniform in [0.5, 2), so no product overflows or goes
// subnormal. With mask=on each input has its own mask with 2% of cells
// invalid, and the output and the chain's intermediates have masks.
// 16384² is not run: the chain needs nine rasters, 9 GiB there. At 8192²
// the fixture is 2.3 GiB, so 8192² is skipped under -short. See
// benchmarks/README.md for how to run the suite and RESULTS.md for
// numbers.
package fusion
