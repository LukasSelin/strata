package cog

import (
	"bytes"
	"context"
	"encoding/binary"
	"testing"

	"github.com/LukasSelin/strata/raster"
)

// FuzzOpen feeds arbitrary bytes to Open and, if they open, reads every
// band of every level. A file's contents must only ever produce errors:
// a panic or a runaway allocation fails the fuzzer.
func FuzzOpen(f *testing.F) {
	for _, order := range []order{binary.LittleEndian, binary.BigEndian} {
		for _, big := range []bool{false, true} {
			for _, comp := range compressions {
				sp := grid16(append(scaleTiepoint(10, 10, 0, 0),
					tagValue{tag: tagGDALNoData, typ: typeASCII, str: "5"},
					geoTags(modelProjected, 1, 32633))...)
				sp.compression, sp.predictor = comp, predictorFloat
				ov := halve(sp)
				ov.subfile = subfileReduced
				f.Add(fileSpec{order: order, big: big, images: []imageSpec{sp, ov}}.write())
			}
		}
	}
	strips := grid16()
	strips.tiled, strips.blockH, strips.planar = false, 5, planarSeparate
	strips.bands, strips.vals = 2, append(strips.vals, strips.vals[0])
	f.Add(fileSpec{order: binary.LittleEndian, images: []imageSpec{strips}}.write())

	f.Fuzz(func(t *testing.T, data []byte) {
		file, err := Open(bytes.NewReader(data))
		if err != nil {
			return
		}
		for lvl := range file.Levels() {
			_ = file.Grid(lvl)
			for band := range min(file.Bands(), 4) {
				src, err := file.Source(SourceOptions{Band: band, Level: lvl})
				if err != nil {
					t.Fatalf("Source(%d, %d): %v", band, lvl, err)
				}
				w, h := src.Size()
				w, h = min(w, 256), min(h, 256)
				dst := raster.NewFloat32(w, h, make([]float32, w*h))
				dst.Valid = raster.NewMask(w * h)
				_ = src.ReadWindow(context.Background(), dst, 0, 0)
			}
		}
	})
}
