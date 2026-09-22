// want package:"kernel"

//strata:kernel
package badconc

func run(c chan int, done chan struct{}) {
	go func() {}() // want `go statement in a kernel package`
	c <- 1         // want `channel send in a kernel package`
	<-c            // want `channel receive in a kernel package`
	select {       // want `select statement in a kernel package`
	case <-done: // want `channel receive in a kernel package`
	}
}
