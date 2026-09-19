package transfer_test

import (
	"fmt"

	"github.com/LukasSelin/strata/algebra"
	"github.com/LukasSelin/strata/raster"
	"github.com/LukasSelin/strata/terrain"
	"github.com/LukasSelin/strata/transfer"
)

// The tables in these examples have the shape a wildfire risk model
// uses, and the numbers are illustrative. Real breakpoints come from
// whoever publishes the index — for Swedish brandriskklass, SMHI's
// current FWI table — and are the caller's to supply, which is the whole
// point of the operations being table-driven.

// Reclass cuts a continuous index into classes. Swedish brandriskklass
// runs from 1, "mycket liten brandrisk", to 5, "extremt stor", so four
// breakpoints separate five classes.
func ExampleReclass() {
	breaks := []float32{1.2, 5.9, 11.2, 21.3}
	classes := []float32{1, 2, 3, 4, 5}

	index := raster.NewFloat32(6, 1, []float32{0.4, 1.2, 8, 11.2, 21.3, 35})
	class := raster.NewFloat32Like(index)
	transfer.Reclass(class, index, breaks, classes)

	fmt.Println(class.Data)
	// A cell exactly on a break takes the class above it: 11.2 is the
	// first cell of class 4, not the last of class 3, which is how a
	// published table reading "klass 4: 11.2 och uppat" means it.

	// Output:
	// [1 2 3 4 5 5]
}

// Lookup applies a bounded curve. A slope factor multiplies fire spread
// by steepness, rising to a plateau where the slope stops mattering;
// the curve is flat beyond its last knot, so there is no need to clamp
// the input first.
func ExampleLookup() {
	degrees := []float32{0, 5, 10, 20, 30, 40}
	factor := []float32{1, 1.1, 1.3, 1.9, 3, 4.5}

	slope := raster.NewFloat32(5, 1, []float32{0, 7.5, 25, 40, 75})
	f := raster.NewFloat32Like(slope)
	transfer.Lookup(f, slope, degrees, factor)

	fmt.Println(f.Data)
	// 7.5 degrees is half way between the 5 and 10 knots, so the factor
	// is half way between 1.1 and 1.3. 75 degrees is past the last knot
	// and takes its value.

	// Output:
	// [1 1.2 2.45 4.5 4.5]
}

// A curve's ys need not be monotone. An aspect factor peaks on the
// south-facing slopes that get the most sun, and the table repeats its
// value at 0 and 360 so the curve is continuous across the wrap.
func ExampleLookup_nonMonotone() {
	bearing := []float32{0, 90, 180, 270, 360}
	factor := []float32{0.8, 1, 1.25, 1, 0.8}

	aspect := raster.NewFloat32(5, 1, []float32{0, 90, 180, 270, 360})
	f := raster.NewFloat32Like(aspect)
	transfer.Lookup(f, aspect, bearing, factor)

	fmt.Println(f.Data)
	// Output:
	// [0.8 1 1.25 1 0.8]
}

// RescaleRange maps one range onto another, without clamping. Here a
// slope in degrees becomes a 0-to-1 weight.
func ExampleRescaleRange() {
	slope := raster.NewFloat32(4, 1, []float32{0, 22.5, 45, 90})
	weight := raster.NewFloat32Like(slope)
	transfer.RescaleRange(weight, slope, 0, 90, 0, 1)

	fmt.Println(weight.Data)
	// Output:
	// [0 0.25 0.5 1]
}

// The operations compose into a model: terrain derivatives become
// factors, the factors combine, and the result becomes a class. Nothing
// in the chain knows it is about fire.
func Example_fireRisk() {
	// A hillside falling away to the east, 10 m cells.
	const w, h = 5, 3
	elev := make([]float32, w*h)
	for y := range h {
		for x := range w {
			elev[y*w+x] = float32(100 - 4*x)
		}
	}
	dem := raster.NewFloat32(w, h, elev)

	slopeDeg := raster.NewFloat32Like(dem)
	aspectDeg := raster.NewFloat32Like(dem)
	terrain.Slope(slopeDeg, dem, terrain.SlopeOptions{CellSize: 10})
	terrain.Aspect(aspectDeg, dem, terrain.AspectOptions{CellSize: 10})

	slopeF := raster.NewFloat32Like(dem)
	aspectF := raster.NewFloat32Like(dem)
	transfer.Lookup(slopeF, slopeDeg, []float32{0, 5, 10, 20, 30, 40}, []float32{1, 1.1, 1.3, 1.9, 3, 4.5})
	transfer.Lookup(aspectF, aspectDeg, []float32{0, 90, 180, 270, 360}, []float32{0.8, 1, 1.25, 1, 0.8})

	risk := raster.NewFloat32Like(dem)
	algebra.Mul(risk, slopeF, aspectF)

	class := raster.NewFloat32Like(dem)
	transfer.Reclass(class, risk, []float32{1.2, 1.6, 2.2, 3}, []float32{1, 2, 3, 4, 5})

	// Slope and Aspect read a 3x3 neighbourhood, so the border is NaN
	// (DESIGN.md §23) and stays NaN through both curves and the class.
	fmt.Printf("slope  %.1f degrees\n", slopeDeg.Data[slopeDeg.Index(2, 1)])
	fmt.Printf("aspect %.1f degrees\n", aspectDeg.Data[aspectDeg.Index(2, 1)])
	fmt.Printf("class  %.0f\n", class.Data[class.Index(2, 1)])
	fmt.Printf("border %v\n", class.Data[class.Index(0, 0)])

	// Output:
	// slope  21.8 degrees
	// aspect 90.0 degrees
	// class  3
	// border NaN
}
