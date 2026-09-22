//go:build goexperiment.simd && amd64

package curve

import "simd/archsimd"

// The vector forms BenchmarkTableSize compares, and TestBenchmarkScansAgree
// checks against the scalar kernels. "vec" is the shipped kernel; the
// others are the candidates it was chosen from (DESIGN.md §50):
//
//   - vec4 / vec2: the same walk over four (Reclass) or two (Lookup)
//     vectors per iteration, so each table entry is loaded once per 32 or
//     16 cells and the blend chains are independent. Lookup keeps five
//     live vectors per cell, so four would spill.
//   - bcast: the table broadcast from the caller's slice inside the loop,
//     rather than expanded once per call, which is what the expansion
//     costs its per-call zeroing to avoid.
//   - exit: stops walking once no lane is at or above a break. Every lane
//     must be below it, so on spread cells the walk rarely stops; on
//     clustered or low cells it stops early.
func init() {
	if !archsimd.X86.AVX2() {
		return
	}
	reclassForms = append(reclassForms,
		form{"vec", reclassFloat32AVX2},
		form{"vec4", reclassVec4},
		form{"bcast", reclassBcast},
		form{"exit", reclassExit},
	)
	lookupForms = append(lookupForms,
		form{"vec", lookupFloat32AVX2},
		form{"vec2", lookupVec2},
	)
}

func load8At(p *[4 * lane]float32, i int) archsimd.Float32x8 {
	return archsimd.LoadFloat32x8Array((*[lane]float32)(p[i*lane : (i+1)*lane]))
}

func store8At(v archsimd.Float32x8, p *[4 * lane]float32, i int) {
	v.StoreArray((*[lane]float32)(p[i*lane : (i+1)*lane]))
}

// expandReclass is the shipped wrapper's table expansion.
func expandReclass(steps *[reclassVecMax]reclassStep, breaks, values []float32) vec8 {
	above := values[1:][:len(breaks)]
	for j, b := range breaks {
		steps[j] = reclassStep{splat(b), splat(above[j])}
	}
	return splat(values[0])
}

func reclassVec4(dst, src, breaks, values []float32) {
	if len(src) < lane || len(breaks) > reclassVecMax {
		scalarReclassFloat32(dst, src, breaks, values)
		return
	}
	var steps [reclassVecMax]reclassStep
	first := expandReclass(&steps, breaks, values)
	i := reclassVec4Lanes(dst, src, &first, steps[:len(breaks)])
	scalarReclassFloat32(dst[i:], src[i:], breaks, values)
}

func reclassVec4Lanes(dst, src []float32, first *vec8, steps []reclassStep) int {
	v0 := archsimd.LoadFloat32x8Array(first)
	n := len(dst)
	src = src[:n]
	for len(dst) >= 4*lane && len(src) >= 4*lane {
		ps, pd := (*[4 * lane]float32)(src), (*[4 * lane]float32)(dst)
		a, b, c, d := load8At(ps, 0), load8At(ps, 1), load8At(ps, 2), load8At(ps, 3)
		ra, rb, rc, rd := v0, v0, v0, v0
		for k := range steps {
			s := &steps[k]
			bv, vv := archsimd.LoadFloat32x8Array(&s.b), archsimd.LoadFloat32x8Array(&s.v)
			ra = vv.IfElse(a.GreaterEqual(bv), ra)
			rb = vv.IfElse(b.GreaterEqual(bv), rb)
			rc = vv.IfElse(c.GreaterEqual(bv), rc)
			rd = vv.IfElse(d.GreaterEqual(bv), rd)
		}
		store8At(a.IfElse(a.IsNaN(), ra), pd, 0)
		store8At(b.IfElse(b.IsNaN(), rb), pd, 1)
		store8At(c.IfElse(c.IsNaN(), rc), pd, 2)
		store8At(d.IfElse(d.IsNaN(), rd), pd, 3)
		dst, src = dst[4*lane:], src[4*lane:]
	}
	for len(dst) >= lane && len(src) >= lane {
		v := load8(src)
		r := v0
		for k := range steps {
			s := &steps[k]
			r = archsimd.LoadFloat32x8Array(&s.v).IfElse(v.GreaterEqual(archsimd.LoadFloat32x8Array(&s.b)), r)
		}
		store8(v.IfElse(v.IsNaN(), r), dst)
		dst, src = dst[lane:], src[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

func reclassBcast(dst, src, breaks, values []float32) {
	if len(src) < lane {
		scalarReclassFloat32(dst, src, breaks, values)
		return
	}
	i := reclassBcastLanes(dst, src, breaks, values)
	scalarReclassFloat32(dst[i:], src[i:], breaks, values)
}

func reclassBcastLanes(dst, src, breaks, values []float32) int {
	values = values[:len(breaks)+1]
	v0 := archsimd.BroadcastFloat32x8(values[0])
	n := len(dst)
	src = src[:n]
	for len(dst) >= lane && len(src) >= lane {
		v := load8(src)
		r := v0
		for j, b := range breaks {
			r = archsimd.BroadcastFloat32x8(values[j+1]).IfElse(v.GreaterEqual(archsimd.BroadcastFloat32x8(b)), r)
		}
		store8(v.IfElse(v.IsNaN(), r), dst)
		dst, src = dst[lane:], src[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

func reclassExit(dst, src, breaks, values []float32) {
	if len(src) < lane || len(breaks) > reclassVecMax {
		scalarReclassFloat32(dst, src, breaks, values)
		return
	}
	var steps [reclassVecMax]reclassStep
	first := expandReclass(&steps, breaks, values)
	i := reclassExitLanes(dst, src, &first, steps[:len(breaks)])
	scalarReclassFloat32(dst[i:], src[i:], breaks, values)
}

func reclassExitLanes(dst, src []float32, first *vec8, steps []reclassStep) int {
	v0 := archsimd.LoadFloat32x8Array(first)
	n := len(dst)
	src = src[:n]
	for len(dst) >= lane && len(src) >= lane {
		v := load8(src)
		r := v0
		for k := range steps {
			s := &steps[k]
			m := v.GreaterEqual(archsimd.LoadFloat32x8Array(&s.b))
			if m.ToBits() == 0 {
				break // the breaks increase, so no lane reaches a later one
			}
			r = archsimd.LoadFloat32x8Array(&s.v).IfElse(m, r)
		}
		store8(v.IfElse(v.IsNaN(), r), dst)
		dst, src = dst[lane:], src[lane:]
	}
	archsimd.ClearAVXUpperBits()
	return n - len(dst)
}

func lookupVec2(dst, src, xs, ys []float32) {
	if len(src) < lane || len(xs) > lookupVecMax {
		scalarLookupFloat32(dst, src, xs, ys)
		return
	}
	var table [lookupVecMax]lookupKnot
	i := lookupVec2Lanes(dst, src, expandLookup(&table, xs, ys))
	scalarLookupFloat32(dst[i:], src[i:], xs, ys)
}

// segment8 is one vector's segment value, as lookupLanes evaluates it.
func segment8(v, lo, hi, x0, y0, dx, dy archsimd.Float32x8) archsimd.Float32x8 {
	r := y0.Add(v.Sub(x0).Div(dx).Mul(dy))
	keep := v.Equal(x0).Or(v.Less(lo)).Or(v.GreaterEqual(hi))
	r = y0.IfElse(keep, r)
	return v.IfElse(v.IsNaN(), r)
}

func lookupVec2Lanes(dst, src []float32, knots []lookupKnot) int {
	i := 0
	if len(knots) == 0 {
		return 0
	}
	k0 := &knots[0]
	lo := archsimd.LoadFloat32x8Array(&k0.x)
	hi := archsimd.LoadFloat32x8Array(&knots[len(knots)-1].x)
	rest := knots[1:]
	n := len(dst)
	src = src[:n]
	for len(dst) >= 2*lane && len(src) >= 2*lane {
		a, b := load8(src), load8(src[lane:])
		ax0, ay0 := lo, archsimd.LoadFloat32x8Array(&k0.y)
		adx, ady := archsimd.LoadFloat32x8Array(&k0.dx), archsimd.LoadFloat32x8Array(&k0.dy)
		bx0, by0, bdx, bdy := ax0, ay0, adx, ady
		for k := range rest {
			kn := &rest[k]
			x, y := archsimd.LoadFloat32x8Array(&kn.x), archsimd.LoadFloat32x8Array(&kn.y)
			dx, dy := archsimd.LoadFloat32x8Array(&kn.dx), archsimd.LoadFloat32x8Array(&kn.dy)
			ma, mb := a.GreaterEqual(x), b.GreaterEqual(x)
			ax0, ay0, adx, ady = x.IfElse(ma, ax0), y.IfElse(ma, ay0), dx.IfElse(ma, adx), dy.IfElse(ma, ady)
			bx0, by0, bdx, bdy = x.IfElse(mb, bx0), y.IfElse(mb, by0), dx.IfElse(mb, bdx), dy.IfElse(mb, bdy)
		}
		store8(segment8(a, lo, hi, ax0, ay0, adx, ady), dst)
		store8(segment8(b, lo, hi, bx0, by0, bdx, bdy), dst[lane:])
		dst, src = dst[2*lane:], src[2*lane:]
		i += 2 * lane
	}
	archsimd.ClearAVXUpperBits()
	return i + lookupLanes(dst, src, knots)
}
