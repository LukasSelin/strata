#lang herbie/platform

;; strata's kernel platform: only the operations a rewrite may use and
;; still be mirrored in the scalar, AVX2 and NEON kernels bit for bit
;; (DESIGN.md §15). No fma (§16, docs/adr/0001-simd-backend.md), and no
;; libm function that archsimd has no lane operation for (hypot, expm1,
;; log1p, ...). Adapted from Herbie's built-in c.rkt platform.
;;
;; binary64 carries the per-call coefficient math (HornScales, the
;; hillshade light vector, rescaleCoeffs), which runs once per call in
;; scalar Go and may use sin and cos.
;;
;; Costs are relative float32 lane costs on a Zen 2-class core, where
;; division and square root are several times the price of a multiply.

(require math/flonum)

(define move-cost 0.125)
(define boolean-move-cost 0.100)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;; BOOLEAN ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define-representation <bool> #:cost boolean-move-cost)

(define-operations () <bool>
  [TRUE  #:spec (TRUE)  #:impl (const true)  #:fpcore TRUE  #:cost boolean-move-cost]
  [FALSE #:spec (FALSE) #:impl (const false) #:fpcore FALSE #:cost boolean-move-cost])

(define-operations ([x <bool>] [y <bool>]) <bool>
  [and #:spec (and x y) #:impl (lambda v (andmap values v)) #:cost boolean-move-cost]
  [or  #:spec (or x y)  #:impl (lambda v (ormap values v))  #:cost boolean-move-cost])

(define-operation (not [x <bool>]) <bool>
  #:spec (not x) #:impl not #:cost boolean-move-cost)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;; BINARY 32 ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define-representation <binary32> #:cost move-cost)

;; A lane select (IfElse / masked blend).
(define-operation (if.f32 [c <bool>] [t <binary32>] [f <binary32>]) <binary32>
  #:spec (if c t f) #:impl if-impl
  #:cost (if-cost boolean-move-cost))

(define-operations ([x <binary32>] [y <binary32>]) <bool>
  [==.f32 #:spec (== x y) #:impl =          #:cost move-cost]
  [!=.f32 #:spec (!= x y) #:impl (negate =) #:cost move-cost]
  [<.f32  #:spec (< x y)  #:impl <          #:cost move-cost]
  [>.f32  #:spec (> x y)  #:impl >          #:cost move-cost]
  [<=.f32 #:spec (<= x y) #:impl <=         #:cost move-cost]
  [>=.f32 #:spec (>= x y) #:impl >=         #:cost move-cost])

(define-operations () <binary32> #:fpcore (! :precision binary32 _)
  [PI.f32       #:spec (PI)       #:impl (const (flsingle pi)) #:fpcore PI       #:cost move-cost]
  [INFINITY.f32 #:spec (INFINITY) #:impl (const +inf.0)        #:fpcore INFINITY #:cost move-cost]
  [NAN.f32      #:spec (NAN)      #:impl (const +nan.0)        #:fpcore NAN      #:cost move-cost])

(define-operation (neg.f32 [x <binary32>]) <binary32>
  #:spec (neg x) #:impl (compose flsingle -)
  #:fpcore (! :precision binary32 (- x)) #:cost move-cost)

(define-operations ([x <binary32>] [y <binary32>]) <binary32> #:fpcore (! :precision binary32 _)
  [+.f32 #:spec (+ x y) #:impl (compose flsingle +) #:cost 0.200]
  [-.f32 #:spec (- x y) #:impl (compose flsingle -) #:cost 0.200]
  [*.f32 #:spec (* x y) #:impl (compose flsingle *) #:cost 0.250]
  [/.f32 #:spec (/ x y) #:impl (compose flsingle /) #:cost 0.800])

(define-operations ([x <binary32>]) <binary32> #:fpcore (! :precision binary32 _)
  [fabs.f32 #:spec (fabs x) #:impl (from-libm 'fabsf) #:cost move-cost]
  [sqrt.f32 #:spec (sqrt x) #:impl (from-libm 'sqrtf) #:cost 1.000])

;; Stand-ins for Atan32 and Atan2F32 in the end-to-end inputs only
;; (kernels.fpcore). They are priced high so Herbie does not reach for
;; them; a rewrite that adds one anywhere else is rejected, since the
;; kernels have their own branch-free polynomial.
(define-operation (atan.f32 [x <binary32>]) <binary32>
  #:spec (atan x) #:impl (from-libm 'atanf)
  #:fpcore (! :precision binary32 (atan x)) #:cost 20)

(define-operation (atan2.f32 [y <binary32>] [x <binary32>]) <binary32>
  #:spec (atan2 y x) #:impl (from-libm 'atan2f)
  #:fpcore (! :precision binary32 (atan2 y x)) #:cost 20)

(define-operations ([x <binary32>] [y <binary32>]) <binary32> #:fpcore (! :precision binary32 _)
  [copysign.f32 #:spec (copysign x y) #:impl (from-libm 'copysignf) #:cost 0.200]
  [fmax.f32     #:spec (fmax x y)     #:impl (from-libm 'fmaxf)     #:cost 0.250]
  [fmin.f32     #:spec (fmin x y)     #:impl (from-libm 'fminf)     #:cost 0.250])

;;;;;;;;;;;;;;;;;;;;;;;;;;;;; BINARY 64 ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define-representation <binary64> #:cost move-cost)

(define-operation (if.f64 [c <bool>] [t <binary64>] [f <binary64>]) <binary64>
  #:spec (if c t f) #:impl if-impl
  #:cost (if-cost boolean-move-cost))

(define-operations ([x <binary64>] [y <binary64>]) <bool>
  [==.f64 #:spec (== x y) #:impl =          #:cost move-cost]
  [!=.f64 #:spec (!= x y) #:impl (negate =) #:cost move-cost]
  [<.f64  #:spec (< x y)  #:impl <          #:cost move-cost]
  [>.f64  #:spec (> x y)  #:impl >          #:cost move-cost]
  [<=.f64 #:spec (<= x y) #:impl <=         #:cost move-cost]
  [>=.f64 #:spec (>= x y) #:impl >=         #:cost move-cost])

(define-operations () <binary64> #:fpcore (! :precision binary64 _)
  [PI.f64       #:spec (PI)       #:impl (const pi)     #:fpcore PI       #:cost move-cost]
  [INFINITY.f64 #:spec (INFINITY) #:impl (const +inf.0) #:fpcore INFINITY #:cost move-cost]
  [NAN.f64      #:spec (NAN)      #:impl (const +nan.0) #:fpcore NAN      #:cost move-cost])

(define-operation (neg.f64 [x <binary64>]) <binary64>
  #:spec (neg x) #:impl - #:fpcore (! :precision binary64 (- x)) #:cost move-cost)

(define-operations ([x <binary64>] [y <binary64>]) <binary64> #:fpcore (! :precision binary64 _)
  [+.f64 #:spec (+ x y) #:impl + #:cost 0.200]
  [-.f64 #:spec (- x y) #:impl - #:cost 0.200]
  [*.f64 #:spec (* x y) #:impl * #:cost 0.250]
  [/.f64 #:spec (/ x y) #:impl / #:cost 0.800])

(define-operations ([x <binary64>]) <binary64> #:fpcore (! :precision binary64 _)
  [fabs.f64 #:spec (fabs x) #:impl (from-libm 'fabs) #:cost move-cost]
  [sqrt.f64 #:spec (sqrt x) #:impl (from-libm 'sqrt) #:cost 1.000]
  [sin.f64  #:spec (sin x)  #:impl (from-libm 'sin)  #:cost 4.000]
  [cos.f64  #:spec (cos x)  #:impl (from-libm 'cos)  #:cost 4.000])
