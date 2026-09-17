// Package bcecheck finds the bounds checks a build leaves inside the
// tightest loops of a package, for the tests that keep the kernels free
// of them (DESIGN.md §39). A check per element costs as much as the
// arithmetic it guards, and one reappears silently when the code or the
// compiler changes.
//
// It compiles the package with the compiler's optimization log
// (-json=0,<dir>), which names every check the compiler could not
// remove, where it is and, for inlined code, the chain of calls it came
// from. The log is per instance, unlike -d=ssa/check_bce, which prints
// one line per source position however many times the code was inlined.
package bcecheck

import (
	"bufio"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Check compiles pkg, in the build configuration it is called from
// (GOEXPERIMENT included), and returns one description per bounds check
// left inside a tightest loop: a loop with no loop of its own, whose
// every iteration pays for the check. A check in a loop that contains
// another loop runs once per row, tile or chunk instead, and is not
// reported; nor is one in a function, or a file, that allow names, where
// the caller's test says why it is allowed. An allow entry ending in
// ".go" is a file, anything else a function name.
//
// A description runs from the position the check was compiled at to the
// function it came from, when the check is in code that was inlined:
// "isInBounds in a loop: ErodeBox (mask.go:78:14) → shrink2Row
// (mask.go:217:10)". A check is allowed when any of those positions is
// in something allow names.
//
// total counts the bounds checks seen anywhere in the package, so a
// caller can tell "none in a loop" from a log that was never written.
func Check(pkg string, allow ...string) (inLoops []string, total int, err error) {
	gobin, err := exec.LookPath("go")
	if err != nil {
		return nil, 0, fmt.Errorf("bcecheck: no go command in PATH: %w", err)
	}
	dir, err := os.MkdirTemp("", "bcecheck")
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = os.RemoveAll(dir) }()

	// The log directory differs on every run, so the flags do too, and
	// the build cache cannot skip the compile that writes the log.
	cmd := exec.Command(gobin, "build", "-gcflags="+pkg+"=-json=0,"+fileURL(dir), pkg) // #nosec G204 -- the package path comes from the test
	if out, cmdErr := cmd.CombinedOutput(); cmdErr != nil {
		return nil, 0, fmt.Errorf("bcecheck: go build: %w\n%s", cmdErr, out)
	}

	allowed := make(map[string]bool, len(allow))
	for _, f := range allow {
		allowed[f] = true
	}
	src := sources{files: map[string]*sourceFile{}}
	// Through an os.Root, so that reading the log stays inside the
	// directory this package made whatever the compiler wrote there.
	root, err := os.OpenRoot(dir)
	if err != nil {
		return nil, 0, err
	}
	defer func() { _ = root.Close() }()
	walkErr := fs.WalkDir(root.FS(), ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		f, err := root.Open(path)
		if err != nil {
			return err
		}
		defer func() { _ = f.Close() }()
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		source := "" // the file this log describes, from its header line
		for sc.Scan() {
			var e entry
			if err := json.Unmarshal(sc.Bytes(), &e); err != nil {
				return fmt.Errorf("bcecheck: %s: %w", path, err)
			}
			if e.File != "" {
				source = e.File
				continue
			}
			if e.Code != "isInBounds" && e.Code != "isSliceInBounds" {
				continue
			}
			total++
			// The position the check was compiled at, then, when it
			// came from inlined code, the positions it was inlined
			// from, innermost last.
			at := []position{e.Range.Start}
			files := []string{source}
			for _, r := range e.Related {
				if r.Message == "inlineLoc" {
					at = append(at, r.Location.Range.Start)
					files = append(files, localPath(r.Location.URI))
				}
			}
			steps, inLoop, ok := make([]string, len(at)), false, false
			for i, p := range at {
				base := filepath.Base(files[i])
				fn, in, err := src.at(files[i], p.Line)
				if err != nil {
					return err
				}
				inLoop = inLoop || in
				ok = ok || allowed[fn] || allowed[base]
				steps[i] = fmt.Sprintf("%s (%s:%d:%d)", fn, base, p.Line, p.Character)
			}
			found := inLoop && !ok
			if found {
				inLoops = append(inLoops, fmt.Sprintf("%s in a loop: %s", e.Code, strings.Join(steps, " → ")))
			}
		}
		return sc.Err()
	})
	if walkErr != nil {
		return nil, total, walkErr
	}
	return inLoops, total, nil
}

// entry is the part of an optimization log record this package reads.
// The first record of a file names it and has nothing else.
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

type position struct {
	Line      int `json:"line"`
	Character int `json:"character"`
}

// sources answers what a line of a file is inside, parsing each file
// once.
type sources struct {
	files map[string]*sourceFile
}

// sourceFile holds the line ranges a position is looked up in: the
// tightest loops, and the functions, by name.
type sourceFile struct {
	tight []lines
	funcs map[string]lines
}

type lines struct{ from, to int }

func (l lines) holds(line int) bool { return line >= l.from && line <= l.to }

// at returns the function line is in, if any, and whether it is inside a
// tightest loop: a loop with no loop of its own.
func (s *sources) at(file string, line int) (fn string, tight bool, err error) {
	f, ok := s.files[file]
	if !ok {
		f, err = parse(file)
		if err != nil {
			return "", false, err
		}
		s.files[file] = f
	}
	for _, r := range f.tight {
		if r.holds(line) {
			tight = true
			break
		}
	}
	for name, r := range f.funcs {
		if r.holds(line) {
			return name, tight, nil
		}
	}
	return "", tight, nil
}

func parse(file string) (*sourceFile, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, file, nil, 0)
	if err != nil {
		return nil, fmt.Errorf("bcecheck: %w", err)
	}
	at := func(n ast.Node) lines {
		return lines{fset.Position(n.Pos()).Line, fset.Position(n.End()).Line}
	}
	out := &sourceFile{funcs: map[string]lines{}}
	ast.Inspect(f, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.FuncDecl:
			out.funcs[s.Name.Name] = at(s)
		case *ast.ForStmt:
			if !hasLoop(s.Body) {
				out.tight = append(out.tight, at(s))
			}
		case *ast.RangeStmt:
			if !hasLoop(s.Body) {
				out.tight = append(out.tight, at(s))
			}
		}
		return true
	})
	return out, nil
}

// hasLoop reports whether n contains a loop.
func hasLoop(n ast.Node) bool {
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			found = true
		}
		return !found
	})
	return found
}

// fileURL is dir as the file URL the compiler's -json flag takes.
func fileURL(dir string) string {
	p := filepath.ToSlash(dir)
	if !strings.HasPrefix(p, "/") {
		return "file:///" + p // a Windows drive path
	}
	return (&url.URL{Scheme: "file", Path: p}).String()
}

// localPath is the file a log's URI names.
func localPath(uri string) string {
	p := strings.TrimPrefix(uri, "file://")
	if len(p) > 2 && p[0] == '/' && p[2] == ':' {
		p = p[1:] // /C:/... on Windows
	}
	return p
}
