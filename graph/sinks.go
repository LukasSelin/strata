package graph

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"github.com/LukasSelin/strata/engine"
	"github.com/LukasSelin/strata/internal/exec"
	"github.com/LukasSelin/strata/internal/summary"
	"github.com/LukasSelin/strata/raster"
)

// teeSink writes each window to every one of its sinks, and folds it into
// a summary when one is wanted: one pipeline output serves every
// destination of a value, so asking for statistics of a value that is
// also written costs a fold of the tile already in memory, and asking for
// statistics alone writes nothing (DESIGN.md §55).
type teeSink struct {
	w, h   int
	masked bool
	sinks  []engine.RasterSink
	fold   *folder
}

func (t *teeSink) Size() (int, int) { return t.w, t.h }
func (t *teeSink) Masked() bool     { return t.masked }

func (t *teeSink) WriteWindow(ctx context.Context, src raster.Float32Raster, x, y int) error {
	// Fold first: a sink may overwrite the Data of invalid cells, which
	// the fold does not read, but nothing is gained by folding after.
	if t.fold != nil {
		if err := t.fold.add(ctx, src); err != nil {
			return err
		}
	}
	for _, s := range t.sinks {
		if err := s.WriteWindow(ctx, src, x, y); err != nil {
			return err
		}
	}
	return nil
}

// folder accumulates a summary over the windows written to a teeSink.
// Windows arrive concurrently and in no set order; the summary's partials
// are exact and combine in any order, so the result is the one
// reduce.Stats gives over the whole raster (DESIGN.md §49).
type folder struct {
	mu sync.Mutex
	p  summary.Partial
}

func (f *folder) add(ctx context.Context, src raster.Float32Raster) error {
	p, err := exec.Reduce(ctx, []raster.Float32Raster{src}, summary.Reducer{}, engine.Options{Workers: 1})
	if err != nil {
		return err
	}
	f.mu.Lock()
	summary.Reducer{}.Combine(&f.p, p)
	f.mu.Unlock()
	return nil
}

func (f *folder) summary() summary.Summary {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.p.Summary()
}

// spill is a value a chunked run stores for a later pass: a temporary raw
// float32 file of its cells and, when it has validity, a file of one byte
// per cell. A byte rather than a bit per cell, so that windows written
// concurrently never share a byte, and rather than a fill value, which a
// valid cell could hold (engine.RawOptions): a stored value reads back
// bit for bit, Data and validity.
type spill struct {
	w, h   int
	masked bool
	data   *engine.RawFile
	valid  *engine.RawFile
}

func newSpill(dir, name string, w, h int, masked bool) (*spill, error) {
	s := &spill{w: w, h: h, masked: masked}
	var err error
	s.data, err = engine.CreateRawFile(filepath.Join(dir, name+".f32"), 4*int64(w)*int64(h), 0o600, 0)
	if err != nil {
		return nil, err
	}
	if masked {
		s.valid, err = engine.CreateRawFile(filepath.Join(dir, name+".valid"), int64(w)*int64(h), 0o600, 0)
		if err != nil {
			_ = s.data.Close()
			return nil, err
		}
	}
	return s, nil
}

func (s *spill) close() error {
	err := s.data.Close()
	if s.valid != nil {
		if e := s.valid.Close(); err == nil {
			err = e
		}
	}
	return err
}

func (s *spill) Size() (int, int) { return s.w, s.h }
func (s *spill) Masked() bool     { return s.masked }

func (s *spill) WriteWindow(_ context.Context, src raster.Float32Raster, x, y int) error {
	if src.Valid != nil && !s.masked {
		panic("graph: a window with validity written to an unmasked stored value")
	}
	var bytes []byte
	if s.masked {
		bytes = make([]byte, src.Width)
	}
	for r := range src.Height {
		off := int64(y+r)*int64(s.w) + int64(x)
		if _, err := s.data.WriteAt(cellBytes(src.Row(r)), 4*off); err != nil {
			return err
		}
		if !s.masked {
			continue
		}
		for c := range bytes {
			bytes[c] = 1
			if src.Valid != nil && !src.IsValid(c, r) {
				bytes[c] = 0
			}
		}
		if _, err := s.valid.WriteAt(bytes, off); err != nil {
			return err
		}
	}
	return nil
}

func (s *spill) ReadWindow(_ context.Context, dst raster.Float32Raster, x, y int) error {
	if s.masked && dst.Valid == nil {
		panic("graph: a masked stored value read into a window without validity")
	}
	var bytes []byte
	if dst.Valid != nil {
		bytes = make([]byte, dst.Width)
	}
	for r := range dst.Height {
		off := int64(y+r)*int64(s.w) + int64(x)
		if _, err := s.data.ReadAt(cellBytes(dst.Row(r)), 4*off); err != nil {
			return fmt.Errorf("graph: reading a stored value: %w", err)
		}
		if dst.Valid == nil {
			continue
		}
		if s.masked {
			if _, err := s.valid.ReadAt(bytes, off); err != nil {
				return fmt.Errorf("graph: reading a stored value's validity: %w", err)
			}
		}
		for c := range bytes {
			dst.SetValid(c, r, !s.masked || bytes[c] != 0)
		}
	}
	return nil
}

// cellBytes is cells' memory as bytes. A stored value is private to one
// run on one machine, so it is kept in native byte order.
func cellBytes(cells []float32) []byte {
	if len(cells) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&cells[0])), 4*len(cells))
}

// tempDir makes the directory a chunked run stores values in.
func tempDir(parent string) (string, error) {
	return os.MkdirTemp(parent, "strata-graph-")
}
