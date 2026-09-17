package vec

import (
	"bufio"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestNoBoundsChecksInLoops compiles this package, in the build
// configuration the test runs in (GOEXPERIMENT included), with the
// compiler's optimization log, and fails if a bounds check survives
// inside a loop: in a kernel's own loop, or in code inlined into one,
// such as the array loads of the AVX2 kernels or the scalar tails they
// call. A bounds check per element costs as much as the arithmetic it
// guards. Checks outside loops, such as the reslices that prove the
// operands as long as dst, run once per call and are allowed.
func TestNoBoundsChecksInLoops(t *testing.T) {
	gobin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("no go command in PATH")
	}
	logDir := t.TempDir()
	logURL := (&url.URL{Scheme: "file", Path: "/" + filepath.ToSlash(logDir)}).String()
	if !strings.HasPrefix(filepath.ToSlash(logDir), "/") {
		logURL = "file:///" + filepath.ToSlash(logDir) // a Windows drive path
	}
	// The log directory differs on every run, so the flags do too, and
	// the build cache cannot skip the compile that writes the log.
	cmd := exec.Command(gobin, "build", "-gcflags=strata/internal/vec=-json=0,"+logURL, ".")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build: %v\n%s", err, out)
	}

	loops := map[string][][2]int{} // file base name -> line ranges of loops
	loopRanges := func(file string) [][2]int {
		if r, ok := loops[file]; ok {
			return r
		}
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		var r [][2]int
		ast.Inspect(f, func(n ast.Node) bool {
			switch n.(type) {
			case *ast.ForStmt, *ast.RangeStmt:
				r = append(r, [2]int{fset.Position(n.Pos()).Line, fset.Position(n.End()).Line})
			}
			return true
		})
		loops[file] = r
		return r
	}
	inLoop := func(file string, line int) bool {
		for _, r := range loopRanges(file) {
			if line >= r[0] && line <= r[1] {
				return true
			}
		}
		return false
	}

	type position struct {
		Line      int `json:"line"`
		Character int `json:"character"`
	}
	type entry struct {
		File  string `json:"file"`
		Range struct {
			Start position `json:"start"`
		} `json:"range"`
		Code    string `json:"code"`
		Related []struct {
			Location struct {
				URI   string `json:"uri"`
				Range struct {
					Start position `json:"start"`
				} `json:"range"`
			} `json:"location"`
			Message string `json:"message"`
		} `json:"relatedInformation"`
	}

	checks := 0
	err = filepath.WalkDir(logDir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		source := "" // the file this log describes, from its header line
		for sc.Scan() {
			var e entry
			if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
				return fmt.Errorf("%s: %w", path, err)
			}
			if e.File != "" {
				source = e.File
				continue
			}
			if e.Code != "isInBounds" && e.Code != "isSliceInBounds" {
				continue
			}
			checks++
			// The check's position, then the positions it was inlined
			// from, innermost last.
			where := []string{fmt.Sprintf("%s:%d:%d", filepath.Base(source), e.Range.Start.Line, e.Range.Start.Character)}
			looped := inLoop(source, e.Range.Start.Line)
			for _, r := range e.Related {
				if r.Message != "inlineLoc" {
					continue
				}
				file := strings.TrimPrefix(r.Location.URI, "file://")
				if len(file) > 2 && file[0] == '/' && file[2] == ':' {
					file = file[1:] // /C:/... on Windows
				}
				line := r.Location.Range.Start.Line
				where = append(where, fmt.Sprintf("inlined from %s:%d:%d", filepath.Base(file), line, r.Location.Range.Start.Character))
				looped = looped || inLoop(file, line)
			}
			if looped {
				t.Errorf("%s inside a loop: %s", e.Code, strings.Join(where, ", "))
			}
		}
		return sc.Err()
	})
	if err != nil {
		t.Fatal(err)
	}
	if checks == 0 {
		t.Fatal("the optimization log has no bounds checks at all; it was probably not written")
	}
}
