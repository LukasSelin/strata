package vec

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

// A Chain must compute what the same operations compute one at a time.
// applyStaged is that reference: one exported kernel call per step,
// through a temporary buffer, on whichever backend the test is running.
// On the scalar backend it is the same kernels scalarChainFrom calls, so
// it checks the blocking and the wiring; on the AVX2 backend it is the
// staged vector kernels against the fused one, which is the comparison
// the whole file exists for.
func applyStaged(first int, steps []Step, dst []float32, srcs [][]float32) {
	acc := make([]float32, len(dst))
	copy(acc, srcs[first])
	for _, s := range steps {
		switch s.Op {
		case OpAdd:
			Add(acc, acc, srcs[s.Src])
		case OpSub:
			Sub(acc, acc, srcs[s.Src])
		case OpMul:
			Mul(acc, acc, srcs[s.Src])
		case OpDiv:
			Div(acc, acc, srcs[s.Src])
		case OpMin:
			Min(acc, acc, srcs[s.Src])
		case OpMax:
			Max(acc, acc, srcs[s.Src])
		case OpAddScalar:
			AddScalar(acc, acc, s.K[0])
		case OpMulScalar:
			MulScalar(acc, acc, s.K[0])
		case OpAffine:
			Affine(acc, acc, s.K[0], s.K[1])
		case OpClamp:
			Clamp(acc, acc, s.K[0], s.K[1])
		case OpAbs:
			Abs(acc, acc)
		case OpSqrt:
			Sqrt(acc, acc)
		default:
			panic("unknown op")
		}
	}
	copy(dst, acc)
}

// chainValues are the values a chain's operands are drawn from: the
// ordinary ones a raster holds, and every float32 class whose handling
// the kernels are specified for.
var chainValues = []float32{
	0, negZero, 1, -1, 2, 0.5, -0.5, 3.25, -7, 1e-30, -1e-30,
	math.Float32frombits(1), // the smallest denormal
	math.MaxFloat32, -math.MaxFloat32,
	inf, ninf, nan,
}

func randomOperands(rng *rand.Rand, inputs, n int) [][]float32 {
	srcs := make([][]float32, inputs)
	for i := range srcs {
		srcs[i] = make([]float32, n)
		for j := range srcs[i] {
			if rng.IntN(4) == 0 {
				srcs[i][j] = chainValues[rng.IntN(len(chainValues))]
			} else {
				srcs[i][j] = float32(rng.NormFloat64())
			}
		}
	}
	return srcs
}

func randomSteps(rng *rand.Rand, inputs, n int) []Step {
	steps := make([]Step, n)
	for i := range steps {
		op := Op(rng.IntN(int(numOps)))
		s := Step{Op: op, Src: -1}
		if op.Binary() {
			s.Src = rng.IntN(inputs)
		}
		// Immediates near 1 keep a long chain in range, so that a
		// mismatch is arithmetic and not every cell drifting to Inf.
		s.K = [2]float32{float32(rng.NormFloat64()), float32(rng.NormFloat64())}
		if op == OpClamp && s.K[0] > s.K[1] {
			s.K[0], s.K[1] = s.K[1], s.K[0]
		}
		steps[i] = s
	}
	return steps
}

func describe(first int, steps []Step) string {
	var b strings.Builder
	fmt.Fprintf(&b, "src%d", first)
	for _, s := range steps {
		if s.Op.Binary() {
			fmt.Fprintf(&b, " %s src%d", s.Op, s.Src)
		} else {
			fmt.Fprintf(&b, " %s%v", s.Op, s.K)
		}
	}
	return b.String()
}

// TestChainMatchesStaged is the load-bearing test: a Chain must write
// the bits its operations write one at a time. The lengths cross both
// the vector lane (8) and the scalar block (chainCells), where a fused
// evaluator that loses its place would first show it.
func TestChainMatchesStaged(t *testing.T) {
	lengths := []int{0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 33, 63, 64, 100,
		chainCells - 1, chainCells, chainCells + 1, 2*chainCells + 5, 5000}
	rng := rand.New(rand.NewPCG(1, 2))
	for _, n := range lengths {
		for _, nsteps := range []int{1, 2, 3, 6, MaxSteps} {
			for trial := range 4 {
				inputs := 1 + rng.IntN(6)
				first := rng.IntN(inputs)
				steps := randomSteps(rng, inputs, nsteps)
				srcs := randomOperands(rng, inputs, n)

				got := make([]float32, n)
				NewChain(inputs, first, steps).Run(got, srcs)

				want := make([]float32, n)
				applyStaged(first, steps, want, srcs)

				for i := range want {
					if !eqFloat32(got[i], want[i]) {
						t.Fatalf("n=%d steps=%d trial=%d backend=%s: %s\n"+
							"cell %d: got %v (%#08x), want %v (%#08x)",
							n, nsteps, trial, Backend(), describe(first, steps),
							i, got[i], math.Float32bits(got[i]),
							want[i], math.Float32bits(want[i]))
					}
				}
			}
		}
	}
}

// TestChainMatchesStagedBothBackends runs TestChainMatchesStaged's
// comparison on the scalar kernels as well, so a SIMD build checks both
// halves of the dispatch table in one run.
func TestChainMatchesStagedBothBackends(t *testing.T) {
	if Backend() == "scalar" {
		t.Skip("no SIMD backend in this build")
	}
	UseScalar(true)
	defer UseScalar(false)
	TestChainMatchesStaged(t)
}

// TestChainIsNotFused pins, for a chain, what TestAffineIsNotFused pins
// for one kernel: a multiply followed by an add rounds twice. The
// running value crossing the two as a register rather than a buffer is
// exactly the situation where a compiler would contract them, and a
// contracted lane would not match the scalar reference on arm64. See
// docs/adr/0001-simd-backend.md.
func TestChainIsNotFused(t *testing.T) {
	const eps = 1.0 / (1 << 23)
	a := float32(1 + eps)
	b := -float32(1 + 2*eps)
	srcs := [][]float32{{a}}

	for _, tc := range []struct {
		name  string
		steps []Step
	}{
		{"MulScalar then AddScalar", []Step{
			{Op: OpMulScalar, Src: -1, K: [2]float32{a}},
			{Op: OpAddScalar, Src: -1, K: [2]float32{b}},
		}},
		{"Affine", []Step{{Op: OpAffine, Src: -1, K: [2]float32{a, b}}}},
		{"Mul then AddScalar", []Step{
			{Op: OpMul, Src: 0},
			{Op: OpAddScalar, Src: -1, K: [2]float32{b}},
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dst := make([]float32, 1)
			NewChain(1, 0, tc.steps).Run(dst, srcs)
			if got := dst[0]; got != 0 || math.Signbit(float64(got)) {
				t.Errorf("= %v (%#08x), want +0; the multiply and the add were fused",
					got, math.Float32bits(got))
			}
		})
	}
}

// TestChainLongRun checks a chain over a run long enough to cross many
// scalar blocks and vector lanes at once, which is the shape the engine
// actually calls it with: one span, not one row.
func TestChainLongRun(t *testing.T) {
	const n = 1 << 16
	const inputs = 6
	rng := rand.New(rand.NewPCG(7, 11))
	srcs := randomOperands(rng, inputs, n)
	steps := make([]Step, inputs-1)
	for i := range steps {
		steps[i] = Step{Op: OpMul, Src: i + 1}
	}
	got := make([]float32, n)
	NewChain(inputs, 0, steps).Run(got, srcs)

	want := make([]float32, n)
	applyStaged(0, steps, want, srcs)
	assertSlicesEqual(t, "six-input product", got, want)
}

func TestChainChecks(t *testing.T) {
	ok := []Step{{Op: OpMul, Src: 1}}
	for _, tc := range []struct {
		name string
		want string
		run  func()
	}{
		{"no inputs", "at least one", func() { NewChain(0, 0, ok) }},
		{"first out of range", "not one of its", func() { NewChain(2, 2, ok) }},
		{"no steps", "no steps", func() { NewChain(2, 0, nil) }},
		{"too many steps", "at most", func() { NewChain(2, 0, make([]Step, MaxSteps+1)) }},
		{"unknown op", "unknown operation", func() {
			NewChain(2, 0, []Step{{Op: numOps, Src: 0}})
		}},
		{"src out of range", "not one of its", func() {
			NewChain(2, 0, []Step{{Op: OpMul, Src: 2}})
		}},
		{"src on a unary op", "must be -1", func() {
			NewChain(2, 0, []Step{{Op: OpAbs, Src: 0}})
		}},
		{"wrong input count", "takes 2 inputs", func() {
			NewChain(2, 0, ok).Run(make([]float32, 4), [][]float32{make([]float32, 4)})
		}},
		{"short input", "has length 3", func() {
			NewChain(2, 0, ok).Run(make([]float32, 4),
				[][]float32{make([]float32, 4), make([]float32, 3)})
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				v := recover()
				if v == nil {
					t.Fatalf("no panic, want one mentioning %q", tc.want)
				}
				s, isString := v.(string)
				if !isString || !strings.Contains(s, tc.want) {
					t.Fatalf("panic %v, want a string mentioning %q", v, tc.want)
				}
			}()
			tc.run()
		})
	}
}

func TestOpString(t *testing.T) {
	if got := OpMul.String(); got != "Mul" {
		t.Errorf("OpMul = %q, want %q", got, "Mul")
	}
	if got := numOps.String(); got != "Op(12)" {
		t.Errorf("numOps = %q, want %q", got, "Op(12)")
	}
	for o := Op(0); o < numOps; o++ {
		if opNames[o] == "" {
			t.Errorf("op %d has no name", o)
		}
		if o.Binary() != (o <= OpMax) {
			t.Errorf("%s.Binary() = %v", o, o.Binary())
		}
	}
}

// TestChainAllocsFree pins that Run allocates nothing: the scalar
// evaluator's block and the vector one's operand array are stack arrays,
// and a chain runs once per band of every tile of every worker.
func TestChainAllocsFree(t *testing.T) {
	const n = 4096
	const inputs = 6
	rng := rand.New(rand.NewPCG(4, 4))
	srcs := randomOperands(rng, inputs, n)
	dst := make([]float32, n)
	steps := make([]Step, inputs-1)
	for i := range steps {
		steps[i] = Step{Op: OpMul, Src: i + 1}
	}
	c := NewChain(inputs, 0, steps)
	if got := testing.AllocsPerRun(20, func() { c.Run(dst, srcs) }); got != 0 {
		t.Errorf("Run allocated %v objects per call, want 0", got)
	}
}
