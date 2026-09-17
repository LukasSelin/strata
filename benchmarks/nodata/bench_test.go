package nodata

import (
	"flag"
	"fmt"
	"strconv"
	"strings"
	"testing"
)

// Run the full matrix with, e.g.:
//
//	GOEXPERIMENT=simd go test ./benchmarks/nodata -run '^$' -bench . -benchmem -count 6 -timeout 3h
//
// Without GOEXPERIMENT=simd the AVX2 variants are skipped and the vec
// variants call scalar internal/vec kernels.
//
// and turn the output into tables with ./benchmarks/nodata/cmd/nodatatable.
// RESULTS.md describes how the published numbers were taken (one pinned
// core, GOMAXPROCS=1, high priority).
var benchSizes = flag.String("nodata.sizes", "1024,4096", "comma-separated square raster sizes")

const benchCellSize = 10

type benchState struct {
	f        *Fixture
	dst      []float32
	dstValid []uint64
	scratch  []uint64
}

// Fixtures are cached for the whole run so that -count passes, which
// cycle through every size/pattern, do not rebuild them. A 4096² fixture
// is about 300 MB; all eight together need roughly 1.5 GB.
var cached = map[string]*benchState{}

func stateFor(size int, p Pattern) *benchState {
	key := fmt.Sprintf("%d/%s", size, p.Name)
	if cached[key] == nil {
		f := NewFixture(size, size, p)
		n := size * size
		dst := make([]float32, n)
		for i := range dst {
			dst[i] = 1 // fault the pages in before timing
		}
		dstValid := make([]uint64, MaskWords(n))
		for i := range dstValid {
			dstValid[i] = 1
		}
		cached[key] = &benchState{f: f, dst: dst, dstValid: dstValid,
			scratch: make([]uint64, SlopeMaskScratchWords(size))}
	}
	return cached[key]
}

func sizes(b *testing.B) []int {
	var out []int
	for _, s := range strings.Split(*benchSizes, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			b.Fatalf("bad -nodata.sizes: %v", err)
		}
		out = append(out, v)
	}
	return out
}

type variant struct {
	name string
	avx2 bool
	run  func(s *benchState)
}

func runMatrix(b *testing.B, variants []variant) {
	for _, size := range sizes(b) {
		for _, p := range Patterns {
			for _, v := range variants {
				name := fmt.Sprintf("size=%d/nodata=%s/%s", size, p.Name, v.name)
				b.Run(name, func(b *testing.B) {
					if v.avx2 && !HaveAVX2 {
						b.Skip("AVX2 not available")
					}
					s := stateFor(size, p)
					cells := float64(size * size)
					b.ReportAllocs()
					for b.Loop() {
						v.run(s)
					}
					nsPerCell := float64(b.Elapsed().Nanoseconds()) / float64(b.N) / cells
					b.ReportMetric(nsPerCell, "ns/cell")
					b.ReportMetric(1e3/nsPerCell, "Mcells/s")
				})
			}
		}
	}
}

func BenchmarkAdd(b *testing.B) {
	runMatrix(b, []variant{
		{"sentinel-scalar-branchy", false, func(s *benchState) {
			AddSentinelBranchy(s.dst, s.f.SentinelA, s.f.SentinelB, Sentinel)
		}},
		{"sentinel-scalar-select", false, func(s *benchState) {
			AddSentinelSelect(s.dst, s.f.SentinelA, s.f.SentinelB, Sentinel)
		}},
		{"sentinel-vec+fixup", false, func(s *benchState) {
			AddSentinelVecFixup(s.dst, s.f.SentinelA, s.f.SentinelB, Sentinel)
		}},
		{"sentinel-avx2-blend", true, func(s *benchState) {
			AddSentinelAVX2(s.dst, s.f.SentinelA, s.f.SentinelB, Sentinel)
		}},
		{"nan-scalar", false, func(s *benchState) {
			AddNaNScalar(s.dst, s.f.NaNA, s.f.NaNB)
		}},
		{"nan-vec", false, func(s *benchState) {
			AddNaNVec(s.dst, s.f.NaNA, s.f.NaNB)
		}},
		{"mask-scalar", false, func(s *benchState) {
			AddMaskScalar(s.dst, s.f.SentinelA, s.f.SentinelB, s.dstValid, s.f.ValidA, s.f.ValidB)
		}},
		{"mask-vec", false, func(s *benchState) {
			AddMaskVec(s.dst, s.f.SentinelA, s.f.SentinelB, s.dstValid, s.f.ValidA, s.f.ValidB)
		}},
		{"mask-vec+fill", false, func(s *benchState) {
			AddMaskVecFill(s.dst, s.f.SentinelA, s.f.SentinelB, s.dstValid, s.f.ValidA, s.f.ValidB, NaN32)
		}},
	})
}

func BenchmarkSlope(b *testing.B) {
	sz := func(s *benchState) int { return s.f.W }
	runMatrix(b, []variant{
		{"sentinel-scalar-branchy", false, func(s *benchState) {
			SlopeSentinelBranchy(s.dst, s.f.SentinelA, sz(s), sz(s), benchCellSize, Sentinel)
		}},
		{"sentinel-scalar-select", false, func(s *benchState) {
			SlopeSentinelSelect(s.dst, s.f.SentinelA, sz(s), sz(s), benchCellSize, Sentinel)
		}},
		{"sentinel-avx2-blend", true, func(s *benchState) {
			SlopeSentinelAVX2(s.dst, s.f.SentinelA, sz(s), sz(s), benchCellSize, Sentinel)
		}},
		{"nan-scalar", false, func(s *benchState) {
			SlopeNaNScalar(s.dst, s.f.NaNA, sz(s), sz(s), benchCellSize)
		}},
		{"nan-avx2", true, func(s *benchState) {
			SlopeNaNAVX2(s.dst, s.f.NaNA, sz(s), sz(s), benchCellSize)
		}},
		{"mask-scalar-branchy", false, func(s *benchState) {
			SlopeMaskBranchy(s.dst, s.f.SentinelA, s.dstValid, s.f.ValidA, sz(s), sz(s), benchCellSize)
		}},
		{"mask-scalar", false, func(s *benchState) {
			SlopeMaskScalar(s.dst, s.f.SentinelA, s.dstValid, s.f.ValidA, sz(s), sz(s), benchCellSize, s.scratch)
		}},
		{"mask-avx2", true, func(s *benchState) {
			SlopeMaskAVX2(s.dst, s.f.SentinelA, s.dstValid, s.f.ValidA, sz(s), sz(s), benchCellSize, s.scratch)
		}},
		{"mask-avx2+fill", true, func(s *benchState) {
			SlopeMaskAVX2Fill(s.dst, s.f.SentinelA, s.dstValid, s.f.ValidA, sz(s), sz(s), benchCellSize, s.scratch, NaN32)
		}},
	})
}

// BenchmarkIngest measures the one-off conversion an IO adapter pays to
// turn a fill-value chunk into each in-memory representation.
func BenchmarkIngest(b *testing.B) {
	runMatrix(b, []variant{
		{"to-nan", false, func(s *benchState) {
			SentinelToNaN(s.dst, s.f.SentinelA, Sentinel)
		}},
		{"to-mask", false, func(s *benchState) {
			MaskFromSentinel(s.dstValid, s.f.SentinelA, Sentinel)
		}},
	})
}

// TestReportMemoryOverhead prints the per-representation storage cost.
func TestReportMemoryOverhead(t *testing.T) {
	for _, size := range []int{1024, 4096, 16384} {
		n := size * size
		data := 4 * n
		mask := 8 * MaskWords(n)
		t.Logf("%5d²: data %8.1f MiB | sentinel +4 B | NaN +0 B | mask +%6.2f MiB (%.3f%% of float32, %.1f%% of uint8) | slope mask scratch %d B",
			size, float64(data)/(1<<20), float64(mask)/(1<<20),
			100*float64(mask)/float64(data), 100*float64(mask)/float64(n),
			8*SlopeMaskScratchWords(size))
	}
}
