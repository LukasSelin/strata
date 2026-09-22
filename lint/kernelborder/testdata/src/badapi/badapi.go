// want package:"kernel"

//strata:kernel
package badapi

type Number interface{ ~float32 | ~float64 }

type Acc struct{}

func Map(m map[int]float32) {} // want `exported kernel Map takes or returns map\[int\]float32`

func Any(x any) {} // want `exported kernel Any takes or returns any`

func Err() error { return nil } // want `exported kernel Err takes or returns error`

func Chan(c chan float32) {} // want `exported kernel Chan takes or returns chan float32`

func Callback(f func(x any)) {} // want `exported kernel Callback takes or returns func\(x any\)`

func Loose[T any](dst []T) {} // want `exported kernel Loose takes or returns \[\]T`

func Anon(r struct{ Stride int }) {} // want `exported kernel Anon takes or returns struct{Stride int}`

func (*Acc) Merge(other interface{ N() int }) {} // want `exported kernel Merge takes or returns interface{N\(\) int}`

func Fine(dst []float32, k float32) (n int) { return 0 }

func Typed[T Number](dst []T) {}
