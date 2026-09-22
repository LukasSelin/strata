// Package kernelborder checks the border between strata's kernels and
// everything above them (DESIGN.md §12, §14).
//
// Kernels work on spans: slices of cells or mask words, and scalars.
// They know nothing about rasters, grids, sources, workers or IO, so the
// same kernel serves a raster row today and a point-batch column later,
// and the engine stays the only place that schedules work. A kernel
// package says so with a directive line directly above its package
// clause, in any of its files:
//
//	//strata:kernel
//	package vec
//
// Four rules follow. None applies to _test.go files.
//
//   - K0, every package except those under a benchmarks directory: only
//     a kernel package may import simd/archsimd, so SIMD stays behind the
//     kernel layer (§14).
//   - K1, kernel packages: imports come from a short standard-library
//     allowlist, simd/archsimd, or other kernel packages. Not raster, not
//     engine, not internal/exec, not context, sync, os or io.
//   - K2, kernel packages: every parameter and result of an exported
//     function or method is span-level: a basic type, or a slice, array,
//     pointer or func built from basic types, type parameters with a
//     type-set constraint, and named types of this or another kernel
//     package. No interfaces, maps, channels or other packages' types.
//   - K3, kernel packages: no go statements, channel operations or
//     select. Concurrency belongs to the engine.
//
// Whether an imported package is a kernel package travels as a package
// fact, so the analyzer needs no list of paths: marking a package is the
// whole registration.
package kernelborder

import (
	"go/ast"
	"go/token"
	"go/types"
	"slices"
	"strings"

	"golang.org/x/tools/go/analysis"
)

// Directive marks a kernel package.
const Directive = "//strata:kernel"

// Analyzer is the kernelborder check.
var Analyzer = &analysis.Analyzer{
	Name:      "kernelborder",
	Doc:       "check that kernel packages stay span-level and that SIMD stays inside them (DESIGN.md §12, §14)",
	URL:       "https://github.com/LukasSelin/strata/blob/master/DESIGN.md#12-dense-vs-sparse-scheduling",
	Run:       run,
	FactTypes: []analysis.Fact{new(isKernel)},
}

var (
	allowFlag  = "math,math/bits,math/big,fmt,errors"
	simdFlag   = "simd/archsimd"
	exemptFlag = "benchmarks"
)

func init() {
	fs := &Analyzer.Flags
	fs.StringVar(&allowFlag, "allow", allowFlag,
		"comma-separated standard-library packages a kernel package may import")
	fs.StringVar(&simdFlag, "simd", simdFlag,
		"import path of the SIMD package that only kernel packages may import")
	fs.StringVar(&exemptFlag, "exempt", exemptFlag,
		"comma-separated path elements whose packages may import the SIMD package unmarked")
}

// isKernel is the fact exported for every package carrying the directive.
type isKernel struct{}

func (*isKernel) AFact()         {}
func (*isKernel) String() string { return "kernel" }

func run(pass *analysis.Pass) (any, error) {
	var files []*ast.File
	for _, f := range pass.Files {
		if !isTestFile(pass, f) {
			files = append(files, f)
		}
	}
	kernel := slices.ContainsFunc(files, hasDirective)
	if kernel {
		pass.ExportPackageFact(new(isKernel))
	}

	for _, f := range files {
		for _, spec := range f.Imports {
			path := strings.Trim(spec.Path.Value, `"`)
			switch {
			case kernel:
				if !allowedImport(pass, path) {
					pass.Reportf(spec.Pos(),
						"kernel package imports %s: kernels take spans and scalars, not representations, engine types or IO (DESIGN.md §12)", path)
				}
			case path == simdFlag && !exempt(pass.Pkg.Path()):
				pass.Reportf(spec.Pos(),
					"%s imported outside a kernel package: SIMD stays inside packages marked %s (DESIGN.md §14)", path, Directive)
			}
		}
		if kernel {
			checkAPI(pass, f)
			checkConcurrency(pass, f)
		}
	}
	return nil, nil
}

func isTestFile(pass *analysis.Pass, f *ast.File) bool {
	return strings.HasSuffix(pass.Fset.File(f.Pos()).Name(), "_test.go")
}

// hasDirective reports whether a comment line before f's package clause
// is the directive.
func hasDirective(f *ast.File) bool {
	for _, g := range f.Comments {
		if g.Pos() >= f.Package {
			break
		}
		for _, c := range g.List {
			if strings.TrimSpace(c.Text) == Directive {
				return true
			}
		}
	}
	return false
}

func allowedImport(pass *analysis.Pass, path string) bool {
	if path == simdFlag || slices.Contains(strings.Split(allowFlag, ","), path) {
		return true
	}
	for _, p := range pass.Pkg.Imports() {
		if p.Path() == path {
			return pass.ImportPackageFact(p, new(isKernel))
		}
	}
	return false
}

func exempt(pkgPath string) bool {
	elems := strings.Split(pkgPath, "/")
	for e := range strings.SplitSeq(exemptFlag, ",") {
		if e != "" && slices.Contains(elems, e) {
			return true
		}
	}
	return false
}

// checkAPI applies K2 to the exported functions and methods of f.
func checkAPI(pass *analysis.Pass, f *ast.File) {
	for _, decl := range f.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || !fn.Name.IsExported() || !exportedReceiver(fn) {
			continue
		}
		for _, list := range []*ast.FieldList{fn.Type.Params, fn.Type.Results} {
			if list == nil {
				continue
			}
			for _, field := range list.List {
				t := pass.TypesInfo.TypeOf(field.Type)
				if t != nil && !spanLevel(pass, t, nil) {
					pass.Reportf(field.Type.Pos(),
						"exported kernel %s takes or returns %s: kernel APIs are spans and scalars (DESIGN.md §12)",
						fn.Name.Name, types.TypeString(t, types.RelativeTo(pass.Pkg)))
				}
			}
		}
	}
}

// exportedReceiver reports whether fn is a function, or a method whose
// receiver's base type is exported.
func exportedReceiver(fn *ast.FuncDecl) bool {
	if fn.Recv == nil || len(fn.Recv.List) == 0 {
		return true
	}
	t := fn.Recv.List[0].Type
	for {
		switch x := t.(type) {
		case *ast.StarExpr:
			t = x.X
		case *ast.IndexExpr:
			t = x.X
		case *ast.IndexListExpr:
			t = x.X
		case *ast.Ident:
			return x.IsExported()
		default:
			return false
		}
	}
}

// spanLevel reports whether t is a type a kernel API may use (K2). seen
// guards against recursive signatures.
func spanLevel(pass *analysis.Pass, t types.Type, seen map[types.Type]bool) bool {
	if seen[t] {
		return true
	}
	switch x := types.Unalias(t).(type) {
	case *types.Basic:
		return true
	case *types.Slice:
		return spanLevel(pass, x.Elem(), seen)
	case *types.Array:
		return spanLevel(pass, x.Elem(), seen)
	case *types.Pointer:
		return spanLevel(pass, x.Elem(), seen)
	case *types.Signature:
		if seen == nil {
			seen = map[types.Type]bool{}
		}
		seen[t] = true
		for _, tuple := range []*types.Tuple{x.Params(), x.Results()} {
			for v := range tuple.Variables() {
				if !spanLevel(pass, v.Type(), seen) {
					return false
				}
			}
		}
		return true
	case *types.TypeParam:
		// A type-set constraint such as ~float32 | ~float64 limits T to
		// concrete types; a method or any constraint does not.
		iface, ok := x.Constraint().Underlying().(*types.Interface)
		return ok && !iface.IsMethodSet() && iface.NumMethods() == 0
	case *types.Named:
		pkg := x.Obj().Pkg()
		switch pkg {
		case nil: // error
			return false
		case pass.Pkg:
			return true
		default:
			return pass.ImportPackageFact(pkg, new(isKernel))
		}
	default: // interfaces, maps, channels, anonymous structs
		return false
	}
}

// checkConcurrency applies K3 to f.
func checkConcurrency(pass *analysis.Pass, f *ast.File) {
	ast.Inspect(f, func(n ast.Node) bool {
		var what string
		switch x := n.(type) {
		case *ast.GoStmt:
			what = "go statement"
		case *ast.SendStmt:
			what = "channel send"
		case *ast.SelectStmt:
			what = "select statement"
		case *ast.UnaryExpr:
			if x.Op == token.ARROW {
				what = "channel receive"
			}
		}
		if what != "" {
			pass.Reportf(n.Pos(), "%s in a kernel package: concurrency belongs to the engine (DESIGN.md §12)", what)
		}
		return true
	})
}
