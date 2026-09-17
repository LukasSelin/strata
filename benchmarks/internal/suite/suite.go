// Package suite is the shared harness of the project benchmark suite
// (DESIGN.md §38). Each benchmarks/<category> package describes its
// fixture and workloads, and Run expands them into one sub-benchmark per
// point of the comparison matrix:
//
//	Benchmark<Op>/size=<N>/mask=<off|on>/backend=<scalar|simd>/workers=<W>[/tiles=<T>]
//
// Every leaf reports the same metrics (see Report), so one parser
// (benchmarks/cmd/stratabench) summarises every category.
//
// Categories that run through the engine set Matrix.Workers (usually to
// Workers()) and Matrix.Tiles, and receive both through Case. The others
// run workers=1 only and have no tiles level.
package suite

import (
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"testing"

	"strata/raster"
)

// Sizes are the square raster sizes of DESIGN.md §38. Roughly: 256² fits
// in L2 or L3 cache, 1024² in L3, 4096² in neither, and 16384² is
// gigabytes per operand set.
var Sizes = []int{256, 1024, 4096, 16384}

// MaxShortSize is the largest size run under -short. At 16384² each
// float32 operand alone is 1 GiB, so larger sizes need gigabytes; each
// category documents its measured peak.
const MaxShortSize = 4096

var sizesFlag = flag.String("strata.sizes", "",
	"comma-separated square raster sizes to run instead of 256,1024,4096,16384")

// Backend names used in benchmark names.
const (
	Scalar = "scalar"
	SIMD   = "simd"
)

// Backends lists the kernel backends every workload is run with.
var Backends = []string{Scalar, SIMD}

// Workers returns the worker counts of DESIGN.md §38: 1, one per physical
// core and one per logical CPU, without duplicates. Where the machine's
// cores are unknown it uses runtime.NumCPU for both.
func Workers() []int {
	physical, logical := CPUs()
	if physical <= 0 || logical <= 0 {
		physical, logical = runtime.NumCPU(), runtime.NumCPU()
	}
	out := []int{1}
	for _, n := range []int{physical, logical} {
		if n > out[len(out)-1] {
			out = append(out, n)
		}
	}
	return out
}

// Kernels switches one kernel package between its scalar and SIMD
// backends, for example internal/vec or internal/stencil.
type Kernels struct {
	// Name identifies the package in the configuration lines, e.g. "vec".
	Name string
	// Backend names the kernels in use; "scalar" means no SIMD.
	Backend func() string
	// UseScalar forces the scalar kernels (true) or restores the best
	// available ones (false).
	UseScalar func(scalar bool)
}

// Case is one point of the matrix.
type Case struct {
	Size    int
	Masked  bool
	Backend string // Scalar or SIMD
	Workers int
	Tiles   string // one of Matrix.Tiles, or "" without a tiles level
}

// Cells is the number of cells one operation processes.
func (c Case) Cells() int { return c.Size * c.Size }

// Name is the sub-benchmark path below the Benchmark function.
func (c Case) Name() string {
	name := fmt.Sprintf("size=%d/mask=%s/backend=%s/workers=%d", c.Size, onOff(c.Masked), c.Backend, c.Workers)
	if c.Tiles != "" {
		name += "/tiles=" + c.Tiles
	}
	return name
}

func onOff(v bool) string {
	if v {
		return "on"
	}
	return "off"
}

// Workload is one timed operation.
type Workload struct {
	// Run performs the operation once over the whole raster. A nil Run
	// skips the case, for combinations that do not apply (such as a
	// single-threaded function with several workers).
	Run func()
	// BytesPerCell is the memory the operation touches per cell, over all
	// operands including dst: 4 bytes per float32 raster plus 1/8 byte per
	// validity mask it reads or writes. It turns throughput into GB/s.
	BytesPerCell float64
}

// Matrix describes one benchmark category. F is its fixture type.
type Matrix[F any] struct {
	Kernels Kernels
	// Fixture builds the operands for one size. It is called once per
	// size, before that size's sub-benchmarks, and should touch every page
	// it allocates so that page faults are not timed.
	Fixture func(size int) F
	// Workload binds an operation to the fixture for one case.
	Workload func(f F, c Case) Workload
	// Sizes lists the raster sizes to run; nil means Sizes. The
	// -strata.sizes flag overrides both.
	Sizes []int
	// Workers lists the worker counts to run; nil means only 1.
	Workers []int
	// Tiles names the tile shapes to run, as a last level of the
	// benchmark name; nil means no tiles level.
	Tiles []string
}

// Run runs m over the whole matrix as sub-benchmarks of b. Sizes above
// MaxShortSize are skipped under -short, and SIMD cases are skipped when
// the build has no SIMD kernels.
func Run[F any](b *testing.B, m Matrix[F]) {
	defer m.Kernels.UseScalar(false)
	for _, size := range sizes(b, m.Sizes) {
		b.Run(fmt.Sprintf("size=%d", size), func(b *testing.B) {
			if testing.Short() && size > MaxShortSize {
				b.Skipf("%d² needs 1 GiB per float32 operand; skipped under -short", size)
			}
			f := m.Fixture(size)
			defer release(size)
			for _, masked := range []bool{false, true} {
				b.Run("mask="+onOff(masked), func(b *testing.B) {
					for _, backend := range Backends {
						b.Run("backend="+backend, func(b *testing.B) {
							m.Kernels.UseScalar(backend == Scalar)
							if backend == SIMD && m.Kernels.Backend() == Scalar {
								b.Skipf("no SIMD kernels in this build of %s: needs GOEXPERIMENT=simd on amd64 with AVX2", m.Kernels.Name)
							}
							for _, workers := range m.workers() {
								b.Run(fmt.Sprintf("workers=%d", workers), func(b *testing.B) {
									c := Case{Size: size, Masked: masked, Backend: backend, Workers: workers}
									if len(m.Tiles) == 0 {
										m.leaf(b, f, c)
										return
									}
									for _, tiles := range m.Tiles {
										c.Tiles = tiles
										b.Run("tiles="+tiles, func(b *testing.B) { m.leaf(b, f, c) })
									}
								})
							}
						})
					}
				})
			}
		})
	}
}

func (m Matrix[F]) workers() []int {
	if m.Workers == nil {
		return []int{1}
	}
	return m.Workers
}

// leaf times one case.
func (m Matrix[F]) leaf(b *testing.B, f F, c Case) {
	// Set again: -count reruns this leaf alone.
	m.Kernels.UseScalar(c.Backend == Scalar)
	w := m.Workload(f, c)
	if w.Run == nil {
		b.Skipf("%s does not apply", c.Name())
	}
	b.ReportAllocs()
	for b.Loop() {
		w.Run()
	}
	Report(b, c.Cells(), w.BytesPerCell)
}

// release returns a large fixture's memory to the OS once its size is
// done, so the next benchmark does not start with gigabytes of garbage.
func release(size int) {
	if size > MaxShortSize {
		debug.FreeOSMemory()
	}
}

func sizes(b *testing.B, def []int) []int {
	if *sizesFlag == "" {
		if def != nil {
			return def
		}
		return Sizes
	}
	var out []int
	for _, s := range strings.Split(*sizesFlag, ",") {
		v, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || v <= 0 {
			b.Fatalf("bad -strata.sizes %q", *sizesFlag)
		}
		out = append(out, v)
	}
	return out
}

// Report adds the suite metrics to a finished benchmark:
//
//   - Mcells/s: million cells processed per second;
//   - ns/cell:  nanoseconds per cell;
//   - GB/s:     bytesPerCell × cells per second, in 10⁹ bytes.
//
// allocs/op comes from b.ReportAllocs.
func Report(b *testing.B, cells int, bytesPerCell float64) {
	total := float64(cells) * float64(b.N)
	sec := b.Elapsed().Seconds()
	b.ReportMetric(total/1e6/sec, "Mcells/s")
	b.ReportMetric(sec*1e9/total, "ns/cell")
	b.ReportMetric(total*bytesPerCell/1e9/sec, "GB/s")
}

// Main is TestMain for a benchmark category. When benchmarks are
// requested it prints the machine configuration first, as "key: value"
// lines that benchstat and stratabench read, then runs the tests.
func Main(m *testing.M, kernels ...Kernels) {
	flag.Parse()
	if f := flag.Lookup("test.bench"); f != nil && f.Value.String() != "" {
		PrintConfig(os.Stdout, kernels...)
	}
	os.Exit(m.Run())
}

// PrintConfig writes the build and machine configuration. go test itself
// prints goos, goarch, pkg and cpu.
func PrintConfig(w io.Writer, kernels ...Kernels) {
	fmt.Fprintf(w, "goversion: %s\n", runtime.Version())
	settings := map[string]string{}
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			settings[s.Key] = s.Value
		}
	}
	exp := settings["GOEXPERIMENT"]
	if exp == "" {
		exp = "none"
	}
	fmt.Fprintf(w, "goexperiment: %s\n", exp)
	if v := settings["GOAMD64"]; v != "" {
		fmt.Fprintf(w, "goamd64: %s\n", v)
	}
	fmt.Fprintf(w, "gomaxprocs: %d\n", runtime.GOMAXPROCS(0))
	// usablecpus follows the process's CPU affinity; the machine's
	// processors do not.
	fmt.Fprintf(w, "usablecpus: %d\n", runtime.NumCPU())
	physical, logical := CPUs()
	fmt.Fprintf(w, "physicalcores: %s\n", countOrUnknown(physical))
	fmt.Fprintf(w, "logicalcpus: %s\n", countOrUnknown(logical))
	for _, k := range kernels {
		fmt.Fprintf(w, "kernels-%s: %s\n", k.Name, k.Backend())
	}
}

func countOrUnknown(n int) string {
	if n <= 0 {
		return "unknown"
	}
	return strconv.Itoa(n)
}

// Fixture data is generated with splitmix64 rather than math/rand: a
// 16384² operand is 268 million cells, and the fixture only needs values
// that do not help any kernel, not statistical quality.
type rng uint64

func (r *rng) next() uint64 {
	*r += 0x9e3779b97f4a7c15
	z := uint64(*r)
	z = (z ^ z>>30) * 0xbf58476d1ce4e5b9
	z = (z ^ z>>27) * 0x94d049bb133111eb
	return z ^ z>>31
}

// FillUniform fills data with values uniform in [lo, hi), deterministic
// for a given seed.
func FillUniform(data []float32, seed uint64, lo, hi float32) {
	r := rng(seed)
	scale := (hi - lo) / (1 << 24)
	for i := range data {
		data[i] = lo + float32(r.next()>>40)*scale
	}
}

// RandomMask returns a validity mask for n cells in which each cell is
// invalid with probability invalid, independently, deterministic for a
// given seed.
func RandomMask(n int, seed uint64, invalid float64) []uint64 {
	m := raster.NewMask(n)
	r := rng(seed)
	threshold := uint64(invalid * (1 << 32))
	for i := range n {
		if r.next()>>32 < threshold {
			m[i>>6] &^= 1 << (i & 63)
		}
	}
	return m
}
