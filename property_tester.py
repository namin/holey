#!/usr/bin/env python3
"""
Property-based testing using Holey's symbolic execution.

Instead of finding values that satisfy a predicate (puzzle solving),
we find counterexamples that violate a property.

If prop(x) should hold for all x, we solve for NOT prop(x) to find counterexamples.
"""
from holey import HoleyWrapper, HoleyWrapperITE, SolverStats
from holey.core import type_map
import ast
from func_timeout import func_timeout, FunctionTimedOut
import traceback
from typing import List, Any, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TestResult:
    """Result of testing a property."""
    prop_name: str
    status: str  # "counterexample", "no_counterexample", "timeout", "error", "unsupported"
    counterexample: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class PropertyTester:
    def __init__(self, all_solvers=False):
        self.results: List[TestResult] = []
        self.use_ite = True
        self.use_bounded_lists = True
        self.bounded_list_max_size = 200
        self.solver_stats = SolverStats(run_all_solvers=all_solvers)

    def extract_params(self, prop_func: str) -> List[Tuple[str, str]]:
        """Parse function signature to get (name, type_str) pairs.

        Returns list of (param_name, type_annotation_string) for typed parameters.
        Skips parameters with default values (those are concrete inputs).
        """
        try:
            tree = ast.parse(prop_func)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    params = []
                    args = node.args.args
                    defaults = node.args.defaults
                    # Number of args without defaults
                    num_no_default = len(args) - len(defaults)

                    for i, arg in enumerate(args):
                        # Skip if this arg has a default value
                        if i >= num_no_default:
                            continue
                        if arg.annotation:
                            type_str = ast.unparse(arg.annotation)
                            params.append((arg.arg, type_str))
                    return params
        except Exception as e:
            print(f"Error parsing function signature: {e}")
        return []

    def negate_property(self, prop_func: str) -> str:
        """Transform property function to find counterexamples.

        Wraps the return expression with 'not' so we find inputs where
        the property is False.
        """
        tree = ast.parse(prop_func)

        class ReturnNegator(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Rename function from prop_* to sat
                node.name = 'sat'
                self.generic_visit(node)
                return node

            def visit_Return(self, node):
                if node.value is not None:
                    # return EXPR -> return not (EXPR)
                    node.value = ast.UnaryOp(
                        op=ast.Not(),
                        operand=node.value
                    )
                return node

        negated_tree = ReturnNegator().visit(tree)
        ast.fix_missing_locations(negated_tree)
        return ast.unparse(negated_tree)

    def detect_list_size(self, prop_func: str) -> Optional[int]:
        """Detect required list size from property function.

        Same heuristics as puzzle_solver.py.
        """
        try:
            tree = ast.parse(prop_func)
        except:
            return None

        sizes = []
        # Get all parameter names as potential answer variables
        answer_vars = set()

        # Extract parameter default values and names
        param_defaults = {}
        if tree.body and isinstance(tree.body[0], ast.FunctionDef):
            func = tree.body[0]
            args = func.args.args
            defaults = func.args.defaults

            # All args without defaults are answer variables
            num_no_default = len(args) - len(defaults)
            for i, arg in enumerate(args):
                if i < num_no_default:
                    answer_vars.add(arg.arg)

            # Defaults align to the end of args
            num_defaults = len(defaults)
            for i, default in enumerate(defaults):
                arg_idx = len(args) - num_defaults + i
                arg_name = args[arg_idx].arg
                if isinstance(default, ast.Constant):
                    if isinstance(default.value, int):
                        param_defaults[arg_name] = default.value
                    elif isinstance(default.value, str):
                        param_defaults[arg_name] = len(default.value)
                elif isinstance(default, ast.List):
                    param_defaults[arg_name] = len(default.elts)

        for node in ast.walk(tree):
            # Look for len(x) == N comparisons
            if isinstance(node, ast.Compare):
                if isinstance(node.left, ast.Call):
                    if isinstance(node.left.func, ast.Name) and node.left.func.id == 'len':
                        if node.left.args and isinstance(node.left.args[0], ast.Name):
                            var_name = node.left.args[0].id
                            if var_name in answer_vars:
                                for comparator in node.comparators:
                                    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, int):
                                        sizes.append(comparator.value)
                                    elif isinstance(comparator, ast.Name) and comparator.id in param_defaults:
                                        sizes.append(param_defaults[comparator.id])

            # Look for x[:N] slicing
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id in answer_vars:
                    if isinstance(node.slice, ast.Slice):
                        if node.slice.upper is not None:
                            if isinstance(node.slice.upper, ast.Constant) and isinstance(node.slice.upper.value, int):
                                sizes.append(node.slice.upper.value)

        if sizes:
            return max(sizes)
        return None

    def test_property(self, prop_func: str, prop_name: str, cmds) -> TestResult:
        """Test a single property by trying to find a counterexample."""

        # Extract typed parameters
        params = self.extract_params(prop_func)
        if not params:
            return TestResult(
                prop_name=prop_name,
                status="unsupported",
                message="No typed parameters found"
            )

        # Check all parameter types are supported
        for param_name, type_str in params:
            if type_str not in type_map:
                return TestResult(
                    prop_name=prop_name,
                    status="unsupported",
                    message=f"Unsupported type '{type_str}' for parameter '{param_name}'"
                )

        # Negate the property
        negated = self.negate_property(prop_func)
        print(f"Negated property:\n{negated}\n")

        # For now, handle single parameter case
        # TODO: extend to multi-parameter
        if len(params) == 1:
            param_name, type_str = params[0]
            typ = type_map[type_str]

            # Detect list size for bounded lists
            list_size = None
            if self.use_bounded_lists and type_str in ['List[int]', 'List[str]']:
                detected_size = self.detect_list_size(prop_func)
                if detected_size is not None and detected_size <= self.bounded_list_max_size:
                    list_size = detected_size
                    print(f"Using bounded list with size {list_size}")

            try:
                result = func_timeout(
                    10,
                    self._run_symbolic,
                    args=(negated, typ, prop_name, cmds, list_size)
                )

                if result is not None:
                    # Found counterexample - verify it
                    if self._verify_counterexample(result, prop_func):
                        return TestResult(
                            prop_name=prop_name,
                            status="counterexample",
                            counterexample={param_name: result}
                        )
                    else:
                        return TestResult(
                            prop_name=prop_name,
                            status="error",
                            message=f"Found candidate {result} but verification failed"
                        )
                else:
                    return TestResult(
                        prop_name=prop_name,
                        status="no_counterexample"
                    )

            except FunctionTimedOut:
                return TestResult(
                    prop_name=prop_name,
                    status="timeout",
                    message="Symbolic execution timed out"
                )
            except Exception as e:
                traceback.print_exc()
                return TestResult(
                    prop_name=prop_name,
                    status="error",
                    message=str(e)
                )
        else:
            # Multi-parameter case
            return self._test_multi_param(prop_func, prop_name, params, cmds)

    def _test_multi_param(self, prop_func: str, prop_name: str,
                          params: List[Tuple[str, str]], cmds) -> TestResult:
        """Handle properties with multiple symbolic parameters."""
        from holey.backend import default_backend
        from holey.core import SymbolicTracer, make_symbolic
        from holey.preprocessor import inject, create_namespace, reset, HoleyWrapperITE, HoleyWrapper

        reset()
        backend = default_backend(cmds, puzzle_name=prop_name, solver_stats=self.solver_stats)
        tracer = SymbolicTracer(backend=backend)
        namespace = create_namespace(tracer)

        # Create symbolic variables for each parameter
        sym_vars = {}
        for param_name, type_str in params:
            typ = type_map[type_str]

            # Detect list size
            list_size = None
            if self.use_bounded_lists and type_str in ['List[int]', 'List[str]']:
                detected_size = self.detect_list_size(prop_func)
                if detected_size is not None and detected_size <= self.bounded_list_max_size:
                    list_size = detected_size

            sym_var = make_symbolic(typ, param_name, tracer, size=list_size)
            sym_vars[param_name] = sym_var
            namespace[param_name] = sym_var

        # Negate and execute
        negated = self.negate_property(prop_func)
        wrapper_class = HoleyWrapperITE if self.use_ite else HoleyWrapper

        try:
            exec(inject(negated, wrapper_class), namespace)
            sat_func = namespace['sat']

            # Run with all symbolic variables
            tracer.driver(lambda: sat_func(*[sym_vars[p] for p, _ in params]))

            solution = tracer.solution()
            if solution is not None:
                # Extract counterexample
                counterexample = {}
                for param_name, type_str in params:
                    sym_var = sym_vars[param_name]
                    val = tracer.solution_var(solution, sym_var)
                    if val is not None:
                        typ = type_map[type_str]
                        counterexample[param_name] = val if str(typ).startswith('list') or isinstance(val, typ) else typ(str(val))

                # Verify
                if self._verify_counterexample_multi(counterexample, prop_func):
                    return TestResult(
                        prop_name=prop_name,
                        status="counterexample",
                        counterexample=counterexample
                    )
                else:
                    return TestResult(
                        prop_name=prop_name,
                        status="error",
                        message=f"Found candidate {counterexample} but verification failed"
                    )
            else:
                return TestResult(
                    prop_name=prop_name,
                    status="no_counterexample"
                )

        except FunctionTimedOut:
            return TestResult(
                prop_name=prop_name,
                status="timeout",
                message="Symbolic execution timed out"
            )
        except Exception as e:
            traceback.print_exc()
            return TestResult(
                prop_name=prop_name,
                status="error",
                message=str(e)
            )

    def _run_symbolic(self, negated: str, typ, prop_name: str, cmds, list_size: Optional[int]):
        """Run symbolic execution on negated property."""
        from holey.preprocessor import driver

        wrapper_class = HoleyWrapperITE if self.use_ite else HoleyWrapper
        sym_var = driver(
            negated, typ, cmds,
            list_size=list_size,
            wrapper_class=wrapper_class,
            puzzle_name=prop_name,
            solver_stats=self.solver_stats
        )

        tracer = sym_var.tracer
        solution = tracer.solution()

        if solution is not None:
            result = tracer.solution_var(solution, sym_var)
            if result is not None:
                return result if str(typ).startswith('list') or isinstance(result, typ) else typ(str(result))
        return None

    def _verify_counterexample(self, value, prop_func: str) -> bool:
        """Verify that the counterexample actually violates the property."""
        namespace = {'List': list}

        # Find function name
        tree = ast.parse(prop_func)
        func_name = 'prop'  # default
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break

        exec(prop_func, namespace)
        prop = namespace[func_name]

        try:
            result = prop(value)
            # Counterexample is valid if the property returns False
            return result == False
        except Exception as e:
            print(f"Verification exception: {e}")
            return False

    def _verify_counterexample_multi(self, values: Dict[str, Any], prop_func: str) -> bool:
        """Verify multi-parameter counterexample."""
        namespace = {'List': list}

        tree = ast.parse(prop_func)
        func_name = 'prop'  # default
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                break

        exec(prop_func, namespace)
        prop = namespace[func_name]

        try:
            result = prop(**values)
            return result == False
        except Exception as e:
            print(f"Verification exception: {e}")
            return False

    def test_file(self, filepath: str, cmds, prop_prefix: str = "prop_") -> List[TestResult]:
        """Test all properties in a file."""
        with open(filepath) as f:
            content = f.read()

        # Parse and find all property functions
        tree = ast.parse(content)

        results = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith(prop_prefix):
                # Extract just this function's source
                prop_func = ast.unparse(node)
                print(f"\n{'='*60}")
                print(f"Testing {node.name}")
                print(f"{'='*60}")
                print(f"Property:\n{prop_func}\n")

                result = self.test_property(prop_func, node.name, cmds)
                results.append(result)
                self.results.append(result)

                # Print result
                self._print_result(result)

        return results

    def _print_result(self, result: TestResult):
        """Print a single test result."""
        if result.status == "counterexample" and result.counterexample is not None:
            print(f"  ✗ COUNTEREXAMPLE FOUND:")
            for param, value in result.counterexample.items():
                print(f"    {param} = {repr(value)}")
        elif result.status == "no_counterexample":
            print(f"  ✓ No counterexample found (property may hold)")
        elif result.status == "timeout":
            print(f"  ? Timeout: {result.message}")
        elif result.status == "error":
            print(f"  ! Error: {result.message}")
        elif result.status == "unsupported":
            print(f"  - Unsupported: {result.message}")

    def print_summary(self):
        """Print summary of all test results."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        counterexamples = [r for r in self.results if r.status == "counterexample"]
        no_counterexamples = [r for r in self.results if r.status == "no_counterexample"]
        timeouts = [r for r in self.results if r.status == "timeout"]
        errors = [r for r in self.results if r.status == "error"]
        unsupported = [r for r in self.results if r.status == "unsupported"]

        total = len(self.results)
        print(f"Total properties tested: {total}")
        print(f"  ✗ Counterexamples found: {len(counterexamples)}")
        print(f"  ✓ No counterexample: {len(no_counterexamples)}")
        print(f"  ? Timeouts: {len(timeouts)}")
        print(f"  ! Errors: {len(errors)}")
        print(f"  - Unsupported: {len(unsupported)}")

        if counterexamples:
            print(f"\nProperties with counterexamples:")
            for r in counterexamples:
                print(f"  - {r.prop_name}: {r.counterexample}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Property-based testing with Holey")
    parser.add_argument('--prop-file', required=True,
                        help='Path to Python file containing property functions')
    parser.add_argument('--prop-prefix', default='prop_',
                        help='Prefix for property function names (default: prop_)')
    parser.add_argument('--smtlib-backends', nargs='+', choices=['z3', 'cvc5'],
                        default=['z3'],
                        help='SMT-LIB backends to use')
    parser.add_argument('--no-bounded-lists', action='store_true',
                        help='Disable bounded list optimization')
    parser.add_argument('--bounded-list-max-size', type=int, default=200,
                        help='Maximum size for bounded lists')
    parser.add_argument('--no-ite', action='store_true',
                        help='Disable ITE mode (use explicit branching)')
    parser.add_argument('--all-solvers', action='store_true',
                        help='Run all solvers and collect stats')

    args = parser.parse_args()

    tester = PropertyTester(all_solvers=args.all_solvers)
    tester.use_bounded_lists = not args.no_bounded_lists
    tester.bounded_list_max_size = args.bounded_list_max_size
    tester.use_ite = not args.no_ite

    tester.test_file(args.prop_file, args.smtlib_backends, args.prop_prefix)
    tester.print_summary()


if __name__ == "__main__":
    main()
