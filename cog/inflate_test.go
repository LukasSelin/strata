package cog

import (
	"bytes"
	"compress/flate"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"math/rand/v2"
	"testing"
)

// inflate is judged against compress/flate as readInto uses it: read up
// to limit bytes, a stream that ends early is short but not an error.
func refInflate(src []byte, limit int) ([]byte, error) {
	buf := make([]byte, limit)
	n, err := io.ReadFull(flate.NewReader(bytes.NewReader(src)), buf)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
		return nil, err
	}
	return buf[:n], nil
}

func checkInflate(t testing.TB, name string, src []byte, limit int) {
	t.Helper()
	want, werr := refInflate(src, limit)
	got, gerr := inflate(make([]byte, limit), src)
	switch {
	case (werr != nil) != (gerr != nil):
		t.Fatalf("%s, limit %d: error %v, compress/flate's %v", name, limit, gerr, werr)
	case werr == nil && !bytes.Equal(got, want):
		i := 0
		for i < min(len(got), len(want)) && got[i] == want[i] {
			i++
		}
		t.Fatalf("%s, limit %d: %d bytes, compress/flate %d; first difference at %d", name, limit, len(got), len(want), i)
	}
}

// deflateOf compresses data at level, flushing every flushEvery bytes if
// that is positive, which puts empty stored blocks mid-stream.
func deflateOf(t testing.TB, data []byte, level, flushEvery int) []byte {
	var buf bytes.Buffer
	w, err := flate.NewWriter(&buf, level)
	if err != nil {
		t.Fatal(err)
	}
	for len(data) > 0 {
		n := len(data)
		if flushEvery > 0 {
			n = min(n, flushEvery)
		}
		_, _ = w.Write(data[:n])
		data = data[n:]
		if flushEvery > 0 {
			_ = w.Flush()
		}
	}
	_ = w.Close()
	return buf.Bytes()
}

// testData is data of several kinds a raster block holds or approaches.
func testData(rng *rand.Rand, kind, n int) []byte {
	b := make([]byte, n)
	switch kind {
	case 0: // random: stored and near-literal blocks
		for i := range b {
			b[i] = byte(rng.Uint32())
		}
	case 1: // runs: long matches at distance 1
		for i := 0; i < n; {
			v, r := byte(rng.Uint32()), 1+rng.IntN(300)
			for j := 0; j < r && i < n; j++ {
				b[i] = v
				i++
			}
		}
	case 2: // a small alphabet, many short matches at every distance
		words := [][]byte{[]byte("slope "), []byte("aspect "), []byte("hillshade "), {0, 0, 0, 1}, {0xff}}
		for i := 0; i < n; {
			i += copy(b[i:], words[rng.IntN(len(words))])
		}
	case 3: // float32 samples of a smooth surface, in predictor 3's planes
		m := n / 4
		for i := range m {
			v := math.Float32bits(float32(400 + 50*math.Sin(float64(i)/40) + rng.NormFloat64()*0.01))
			for k := range 4 {
				b[k*m+i] = byte(v >> (8 * (3 - k)))
			}
		}
	default: // uint16 differences, predictor 2's output
		for i := 0; i+1 < n; i += 2 {
			binary.LittleEndian.PutUint16(b[i:], uint16(rng.NormFloat64()*20))
		}
	}
	return b
}

func TestInflateMatchesFlate(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	sizes := []int{0, 1, 7, 100, 4096, 70000, 1 << 18}
	levels := []int{flate.HuffmanOnly, flate.NoCompression, flate.BestSpeed, 5, flate.BestCompression}
	streams := 0
	for kind := range 5 {
		for _, n := range sizes {
			data := testData(rng, kind, n)
			for _, level := range levels {
				for _, flushEvery := range []int{0, 1000} {
					src := deflateOf(t, data, level, flushEvery)
					name := func(what string) string {
						return what + ": kind " + string(rune('0'+kind)) + ", " + itoa(n) + " bytes, level " + itoa(level)
					}
					streams++
					// Whole, and cut at the limit anywhere.
					for _, limit := range []int{n, n + 300, max(n-1, 0), n / 2, rng.IntN(n + 1)} {
						checkInflate(t, name("whole"), src, limit)
					}
					// Cut short anywhere, including inside headers.
					for range 8 {
						cut := rng.IntN(len(src) + 1)
						checkInflate(t, name("cut at "+itoa(cut)), src[:cut], n)
					}
					// Corrupt a byte or two.
					for range 8 {
						bad := bytes.Clone(src)
						if len(bad) == 0 {
							break
						}
						for range 1 + rng.IntN(2) {
							bad[rng.IntN(len(bad))] ^= byte(1 + rng.IntN(255))
						}
						checkInflate(t, name("corrupt"), bad, n)
					}
				}
			}
		}
	}
	t.Logf("%d streams", streams)
}

func itoa(n int) string {
	if n < 0 {
		return "-" + itoa(-n)
	}
	if n < 10 {
		return string(rune('0' + n))
	}
	return itoa(n/10) + string(rune('0'+n%10))
}

// FuzzInflate holds inflate to compress/flate on any input and limit.
func FuzzInflate(f *testing.F) {
	rng := rand.New(rand.NewPCG(13, 14))
	for kind := range 5 {
		for _, level := range []int{flate.HuffmanOnly, flate.NoCompression, flate.BestSpeed, flate.BestCompression} {
			data := testData(rng, kind, 3000)
			f.Add(deflateOf(f, data, level, 700), uint16(3000))
		}
	}
	f.Add([]byte{}, uint16(10))
	f.Add([]byte{0x03, 0x00}, uint16(10)) // an empty fixed block
	f.Fuzz(func(t *testing.T, src []byte, limit uint16) {
		checkInflate(t, "fuzz", src, int(limit))
	})
}
