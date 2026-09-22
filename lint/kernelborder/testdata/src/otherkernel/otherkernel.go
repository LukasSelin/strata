// want package:"kernel"

//strata:kernel
package otherkernel

type Acc struct{ n int64 }

func (a *Acc) Add(xs []float32) { a.n += int64(len(xs)) }
