from holey import drive_sat, LLMSolver
import copy
import re
import json
import ast
from func_timeout import func_timeout, FunctionTimedOut
import traceback
from typing import List, Any, Dict, Optional, Tuple
import sys
from collections import defaultdict
from contextlib import contextmanager
from io import StringIO
@contextmanager
def capture_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    output = StringIO()
    sys.stdout = output
    sys.stderr = output
    try:
        yield output
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

class PuzzleSolver:
    def __init__(self):
        self.count = 0
        self.counts = defaultdict(int)
        self.success_count = 0
        self.success_counts = defaultdict(int)
        self.success_count_llm = 0
        self.success_counts_llm = defaultdict(int)
        self.timeout_staging_count = 0
        self.error_verify_count = 0
        self.error_staging_count = 0
        self.error_smt_count = 0
        self.error_smt_var_count = 0
        self.error_unsupported_answer_type = 0
        self.extrapolate_small_count = 0
        self.extrapolate_small_success_count = 0
        self.extrapolate_large_success_count = 0
        self.extrapolate_stats = defaultdict(list)
        self.end2end_stats = defaultdict(list)
        self.show_llm_matrix = False
        self.names_of_extrapolated_puzzles = []
        self.names_of_successfully_extrapolated_puzzles = []
        self.use_bounded_lists = False  # Controlled by command-line flag
        self.bounded_list_max_size = 200  # Maximum size for bounded lists

    def detect_list_size(self, sat_func: str) -> Optional[int]:
        """Detect required list size from sat function.

        Looks for patterns that directly constrain list size:
        - len(x) == N or len(li) == N (explicit length constraint)
        - len(x) == param where param has a default value (parameter-based)
        - len(x) == len(param) where param is a string/list with default
        - x[:N] or li[:N] slicing on the answer variable
        - all(...) with range(N) where the iteration uses x[:i] pattern
        - li[i] or li[i+1] access inside range(N) loops

        Does NOT use range(N) from general loops without answer variable access.
        """
        try:
            tree = ast.parse(sat_func)
        except:
            return None

        sizes = []
        answer_vars = ('x', 'li', 'l', 'nums', 'arr', 'lst', 'seq', 'colors', 'ans')

        # Extract parameter default values from function signature
        param_defaults = {}
        if tree.body and isinstance(tree.body[0], ast.FunctionDef):
            func = tree.body[0]
            args = func.args.args
            defaults = func.args.defaults
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
            # Look for len(x) == N or len(x) == param comparisons
            if isinstance(node, ast.Compare):
                if isinstance(node.left, ast.Call):
                    if isinstance(node.left.func, ast.Name) and node.left.func.id == 'len':
                        if node.left.args and isinstance(node.left.args[0], ast.Name):
                            var_name = node.left.args[0].id
                            if var_name in answer_vars:
                                for comparator in node.comparators:
                                    # Direct numeric literal
                                    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, int):
                                        sizes.append(comparator.value)
                                    # Parameter reference: len(li) == n
                                    elif isinstance(comparator, ast.Name) and comparator.id in param_defaults:
                                        sizes.append(param_defaults[comparator.id])
                                    # len(param): len(li) == len(s) where s is a string param
                                    elif isinstance(comparator, ast.Call):
                                        if isinstance(comparator.func, ast.Name) and comparator.func.id == 'len':
                                            if comparator.args and isinstance(comparator.args[0], ast.Name):
                                                param_name = comparator.args[0].id
                                                if param_name in param_defaults:
                                                    sizes.append(param_defaults[param_name])

            # Look for x[:N] or li[:N] slicing on answer variable
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id in answer_vars:
                    if isinstance(node.slice, ast.Slice):
                        if node.slice.upper is not None:
                            if isinstance(node.slice.upper, ast.Constant) and isinstance(node.slice.upper.value, int):
                                sizes.append(node.slice.upper.value)

            # Look for all/any/sum(...for i in range(N)) patterns
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ('all', 'any', 'sum'):
                    if node.args:
                        arg0 = node.args[0]
                        if isinstance(arg0, (ast.GeneratorExp, ast.ListComp)):
                            gen = arg0
                            elt = gen.elt
                            generators = gen.generators
                            for comp in generators:
                                if isinstance(comp.iter, ast.Call):
                                    if isinstance(comp.iter.func, ast.Name) and comp.iter.func.id == 'range':
                                        if comp.iter.args:
                                            # range(stop) -> args[0], range(start, stop[, step]) -> args[1]
                                            if len(comp.iter.args) == 1:
                                                range_arg = comp.iter.args[0]  # range(stop)
                                            else:
                                                range_arg = comp.iter.args[1]  # range(start, stop) or range(start, stop, step)
                                            range_val = None
                                            if isinstance(range_arg, ast.Constant) and isinstance(range_arg.value, int):
                                                range_val = range_arg.value
                                            elif isinstance(range_arg, ast.Name) and range_arg.id in param_defaults:
                                                range_val = param_defaults[range_arg.id]

                                            if range_val is not None:
                                                # Check if body references answer var with indexing
                                                # Collect all index patterns, don't break early
                                                found_access = False
                                                max_offset = 0
                                                for subnode in ast.walk(elt):
                                                    if isinstance(subnode, ast.Subscript):
                                                        if isinstance(subnode.value, ast.Name) and subnode.value.id in answer_vars:
                                                            # Check for slice (li[:i]) or index (li[i], li[i+1])
                                                            if isinstance(subnode.slice, ast.Slice):
                                                                found_access = True
                                                            elif isinstance(subnode.slice, ast.BinOp):
                                                                # li[i+1] pattern - need size range_val + offset
                                                                if isinstance(subnode.slice.op, ast.Add):
                                                                    if isinstance(subnode.slice.right, ast.Constant):
                                                                        max_offset = max(max_offset, subnode.slice.right.value)
                                                                        found_access = True
                                                            elif isinstance(subnode.slice, ast.Name):
                                                                # li[i] pattern - need size range_val
                                                                found_access = True
                                                if found_access:
                                                    sizes.append(range_val + max_offset)

        if sizes:
            return max(sizes)  # Return largest detected size
        return None

    def symbolic_solve1(self, typ, sat_func: str, ans_type: str, name: str, cmds, llm_solver, list_size=None) -> Optional[str]:
        sym_var = drive_sat(sat_func, typ, cmds, llm_solver=llm_solver, list_size=list_size)
        tracer = sym_var.tracer
        with capture_output() as captured:
            solution = tracer.solution()
        log = captured.getvalue()
        print(log)
        return tracer, sym_var, solution, log

    def symbolic_solve(self, sat_func: str, ans_type: str, name: str, cmds, llm_solver, counting=True) -> Optional[str]:
        typ = None
        if ans_type == 'int':
            typ = int
        elif ans_type == 'str':
            typ = str
        elif ans_type == 'float':
            typ = float
        elif ans_type == 'bool':
            typ = bool
        elif ans_type == 'List[int]':
            typ = list[int]
        elif ans_type == 'List[str]':
            typ = list[str]
        elif ans_type == 'List[float]':
            typ = list[float]
        if not typ:
            print("Unsupported answer type", ans_type)
            self.error_unsupported_answer_type += 1
            return None, ""

        # Detect list size for bounded list mode
        list_size = None
        if self.use_bounded_lists and ans_type in ['List[int]', 'List[str]']:
            detected_size = self.detect_list_size(sat_func)
            if detected_size is not None and detected_size <= self.bounded_list_max_size:
                list_size = detected_size
                print(f"Using bounded list with size {list_size}")

        if counting:
            self.count += 1
            self.counts[ans_type] += 1
        tracer, sym_var, solution, log = self.symbolic_solve1(typ, sat_func, ans_type, str, cmds, llm_solver=None, list_size=list_size)
        if False and llm_solver and solution is None:
            tracer_llm, sym_var_llm, solution_llm, log_llm = self.symbolic_solve1(typ, sat_func, ans_type, str, cmds, llm_solver=llm_solver)
            if solution is not None:
                tracer, sym_var, solution, log = tracer_llm, sym_var_llm, solution_llm, log_llm
        if solution is None:
            print("Could not find any solution for puzzle " + name)
            if counting:
                self.error_smt_count += 1
            return None, log
        solution_var = tracer.solution_var(solution, sym_var)
        if solution_var is None:
            if counting:
                self.error_smt_var_count += 1
            print('Solution', solution)
            print("Could not find any solution var")
            return None, log
        result = solution_var if str(typ).startswith('list') or isinstance(solution_var, typ) else typ(str(solution_var))
        print("Found solution", result)
        return result, log

    def solve_puzzle_llm(self, puzzle_data: Any, llm_solver) -> Optional[str]:
        name = puzzle_data.get('name', '')
        sat_func = puzzle_data.get('sat_function', puzzle_data.get('sat', ''))
        if not sat_func:
            print("Missing sat_func")
            return None
        print('sat_func', sat_func)
        ans_type = puzzle_data.get('ans_type', None)
        if not ans_type:
            print("Missing ans_type")
            return None
        result = call_solvers(llm_solver, self.end2end_stats, name, lambda x: x.solve_end2end(sat_func, ans_type, name, check_result))
        if result is not None:
            self.success_count += 1
            self.success_counts[ans_type] += 1
        self.count += 1
        self.counts[ans_type] += 1
        return result

    def solve_puzzle(self, puzzle_data: Any, cmds, llm_solver, llm_end, reason=None) -> Optional[str]:
        name = puzzle_data.get('name', '')
        sat_func = puzzle_data.get('sat_function', puzzle_data.get('sat', ''))
        if not sat_func:
            print("Missing sat_func")
            return None
        print('sat_func', sat_func)
        ans_type = puzzle_data.get('ans_type', None)
        if not ans_type:
            print("Missing ans_type")
            return None
        try:
            result, log = func_timeout(20 if llm_solver else 6, self.symbolic_solve, args=(sat_func, ans_type, name, cmds, llm_solver, not reason))
            if result is not None:
                if not check_result(result, sat_func):
                    self.error_verify_count += 1
                    print("WARNING: Solution verification failed for puzzle "+name)
                    result = None
                else:
                    if not reason:
                        self.success_count += 1
                        self.success_counts[ans_type] += 1
                        print("Yes! Solved for puzzle ", name)
                        if llm_end:
                            result_end2end = call_solvers(llm_solver, self.end2end_stats, name, lambda x: x.solve_end2end(sat_func, ans_type, name, check_result))
                            if result_end2end is not None:
                                print('Solved by LLMs too')
                                self.success_count_llm += 1
                                self.success_counts_llm[ans_type] += 1
            if not reason and result is None and not llm_end:
                varied_puzzle_sat_func, reason = vary(sat_func)
                if varied_puzzle_sat_func is not None:
                    self.extrapolate_small_count += 1
                    print('Solving simpler variation', reason)
                    varied_puzzle = copy.deepcopy(puzzle_data)
                    varied_puzzle['sat_function'] = varied_puzzle_sat_func
                    varied_result = self.solve_puzzle(varied_puzzle, cmds, llm_solver, llm_end, reason=reason)
                    if varied_result is not None:
                        self.extrapolate_small_success_count += 1
                        self.names_of_extrapolated_puzzles.append(name)
                        if llm_solver:
                            result = call_solvers(llm_solver, self.extrapolate_stats, name, lambda x: x.extrapolate(varied_puzzle_sat_func, sat_func, reason, varied_result, ans_type, name, check_result, log))
                            if result is not None:
                                self.names_of_successfully_extrapolated_puzzles.append(name)
                                self.extrapolate_large_success_count += 1
                            result_end2end = call_solvers(llm_solver, self.end2end_stats, name, lambda x: x.solve_end2end(sat_func, ans_type, name, check_result))
                            if result_end2end is not None:
                                self.show_llm_matrix = True
                            if result is not None or result_end2end is not None:
                                self.success_count += 1
                                self.success_counts[ans_type] += 1
                                print("Yes! Solved via extrapolation for puzzle ", name)
                                return result
            if False and llm_solver and result is None:
                print('\nFallback to LLM!')
                result = call_solvers(llm_solver, {}, name, lambda x: x.solve_end2end(sat_func, ans_type, name, check_result) or x.smtlib_solve(sat_func, ans_type, name, log, check_result, cmds))
            return result
        except FunctionTimedOut:
            print("Timed out for puzzle "+name)
            self.timeout_staging_count += 1
        except Exception as e:
            self.error_staging_count += 1
            print("Exception -- for puzzle", name, e)
            traceback.print_exc()
        if False and llm_solver:
            print('\nFallback to LLM after error!')
            return call_solvers(llm_solver, {}, name, lambda x: x.solve_end2end(sat_func, ans_type, name, check_result))
        return None

    def pretty_counts(self):
        return self.pretty_counts_within(self.success_counts, self.counts)

    def pretty_counts_llm(self):
        return self.pretty_counts_within(self.success_counts_llm, self.success_counts)

    def pretty_counts_within(self, success_counts, counts):
        count_stats = sorted([(success_counts[ans_type], total, ans_type) for ans_type,total in counts.items()], reverse=True)
        r = ""
        for success, total, ans_type in count_stats:
            success_percentage = (100.0 * success / total) if total > 0 else 0
            r += (f"- {success_percentage:.0f}% ({success} out of {total}) of `{ans_type}` puzzles,")
            r += '\n'
        total = sum(counts.values())
        success = sum(success_counts.values())
        success_percentage = (100.0 * success / total) if total > 0 else 0
        r += (f"- {success_percentage:.0f}% ({success} out of {total}) overall.")
        r += '\n'
        return r

    def extrapolation_matrix(self):
        r = ""
        for solver_name in self.extrapolate_stats.keys():
            for kind, stat in [('extrapolate', self.extrapolate_stats[solver_name]), ('end-to-end', self.end2end_stats[solver_name])]:
                agg = [0 if x[1] is None else 1 for x in stat]
                r += "- "
                r += solver_name.ljust(10)
                r += ('('+kind+')').rjust(15)
                r += ('_'+str(sum(agg))+'_').rjust(4)
                r += " "
                r += " ".join([str(x) for x in agg])
                r += "\n"
        return r

    def pretty_stats(self):
        extrapolation = f"""
### Extrapolation
- {self.extrapolate_small_count} smaller problems tried
- {self.extrapolate_small_success_count} successes on smaller problem
""" if self.extrapolate_small_count > 0 else ""

        if self.extrapolate_large_success_count > 0:
            extrapolation += f"""- {self.extrapolate_large_success_count} successful extrapolations
"""
            extrapolation += f"""
#### Extrapolated puzzles
{' '.join(self.names_of_extrapolated_puzzles)}
#### Successfully extrapolated puzzles
{' '.join(self.names_of_successfully_extrapolated_puzzles)}
"""

        if self.show_llm_matrix:
            extrapolation += f"""
#### Matrix
{self.extrapolation_matrix()}
"""

        return f"""
## Current status

The symbolic execution{'' if self.llm_solver else ' alone'} currently solves:
{self.pretty_counts()}
with the following errors:
- {self.timeout_staging_count} timeouts after 3 seconds at staging time (while generating the SMTLIB program)
- {self.error_staging_count} errors at staging time
- {self.error_verify_count} SMTLIB programs returning `sat` but the original `sat` function failing on synthesized model input,
- {self.error_smt_count + self.error_smt_var_count} SMTLIB programs returning non-`sat` (e.g. `unsat`, `unknown` or timing out after 2 seconds)
- {self.total_count-self.count} (out of {self.total_count}) puzzles not yet even attempted because their type is not `int` or `str`, such as `float`, `list` (of various specialization), etc.
"""+extrapolation

def check_result(result, sat_func):
    namespace = {'List': list}
    exec(sat_func, namespace)
    sat = namespace['sat']
    try:
        outcome = sat(result)
    except Exception as e:
        print('Exception in checking result:', e)
        return False
    if not outcome:
        return False
    return True

def run_benchmarks(puzzle_file: str, name_prefixes = None, name_suffixes = None, answer_types = None, smtlib_backends = None, llm_solver = None, llm_all = False, llm_end = False, use_bounded_lists = False, bounded_list_max_size = 100, show_shrunk = False):
    with open(puzzle_file) as f:
        puzzles = json.load(f)

    total = len(puzzles)
    print(f"Starting with {total} puzzles...")

    # Filter puzzles
    if name_prefixes:
        puzzles = [p for p in puzzles if any(p.get('name', '').startswith(name_prefix) for name_prefix in name_prefixes)]
    if name_suffixes:
        puzzles = [p for p in puzzles if any(p.get('name', '').endswith(name_suffix) for name_suffix in name_suffixes)]
    if answer_types:
        puzzles = [p for p in puzzles if p['ans_type'] in answer_types]

    solver = PuzzleSolver()
    solver.total_count = total
    solver.llm_solver = llm_solver
    solver.use_bounded_lists = use_bounded_lists
    solver.bounded_list_max_size = bounded_list_max_size
    if use_bounded_lists:
        print(f"Using bounded list encoding (max size: {bounded_list_max_size})")

    print(f"Running benchmarks on {len(puzzles)} puzzles...")
    if name_prefixes:
        print(f"Filtered to puzzles starting with '{name_prefixes}'")
    if answer_types:
        print(f"Filtered to puzzles of answer types: {answer_types}")

    for i, puzzle in enumerate(puzzles):
        name = puzzle.get('name', 'Unknown')
        print(f"\nSolving puzzle {i+1}/{len(puzzles)}: {name}")

        if llm_all:
            result = solver.solve_puzzle_llm(puzzle, llm_solver)
        else:
            result = solver.solve_puzzle(puzzle, smtlib_backends, llm_solver, llm_end)

    if llm_all:
        print(f"""## Current status

LLMs currently solve:
{solver.pretty_counts()}
""")
    else:
        print(solver.pretty_stats())
        if llm_end:
            print(f"Within the symbolic success, the LLMs solves the following:")
            print(solver.pretty_counts_llm())

    if show_shrunk and solver.names_of_extrapolated_puzzles:
        print("\n### Puzzles with successfully solved smaller variations:")
        print(' '.join(solver.names_of_extrapolated_puzzles))

def vary(sat_func):
    # Find all large constants
    constants = re.findall(r'\d\d\d+', sat_func)

    # Identify unique constants
    unique_constants = sorted(list(set(constants)))

    n = len(unique_constants)

    if n == 1:
        print('One large constant for extrapolation')
        constant = unique_constants[0]
        smaller = '3'
        varied_sat_func = re.sub(rf'\b{constant}\b', smaller, sat_func)
        if sat_func != varied_sat_func:
            return varied_sat_func, f'replaced {constant} with {smaller}'
    elif n==2:
        print('Two large constants for extrapolation')
        constant1 = unique_constants[0]
        smaller1 = '3'
        varied_sat_func = re.sub(rf'\b{constant1}\b', smaller1, sat_func)
        constant2 = unique_constants[1]
        smaller2 = '5'
        varied_sat_func = re.sub(rf'\b{constant2}\b', smaller2, varied_sat_func)
        if sat_func != varied_sat_func:
            return varied_sat_func, f'replaced {constant1} with {smaller1} and {constant2} with {smaller2}'
    elif "Hello, world!" in sat_func:
        varied_sat_func = sat_func.replace("Hello, world!", "Hel!")
        return varied_sat_func, f'replaced Hello, world! with Hel!'
    else:
        print('Too many constants for extrapolation')

    return None, None

def call_solvers(llm_solvers, stats, name, callback):
    best = None
    print('Solvers:', llm_solvers.keys())
    for solver_name, solver in llm_solvers.items():
        try:
            result = callback(solver)
        except Exception as e:
            print("Error with solver:", str(e))
            result = None
        if solver_name:
            stats[solver_name].append((name, result))
            if best is None and result is not None:
                best = result
    return best

def infer_ans_type(sat_func: str) -> Optional[str]:
    """Infer answer type from the first parameter's type hint in a sat function."""
    import ast
    try:
        tree = ast.parse(sat_func)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'sat':
                if node.args.args:
                    first_arg = node.args.args[0]
                    if first_arg.annotation:
                        return ast.unparse(first_arg.annotation)
    except:
        pass
    return None

def solve_sat_file(sat_file: str, smtlib_backends: list, llm_solver=None, llm_all=False, llm_end=False, use_bounded_lists=True, bounded_list_max_size=200):
    """Solve a single Python file containing a sat function."""
    with open(sat_file) as f:
        sat_func = f.read()

    ans_type = infer_ans_type(sat_func)
    if ans_type:
        print(f"Inferred ans_type: {ans_type}")
    else:
        print("Could not infer ans_type from type hints.")
        return None

    solver = PuzzleSolver()
    solver.total_count = 1
    solver.llm_solver = llm_solver
    solver.use_bounded_lists = use_bounded_lists
    solver.bounded_list_max_size = bounded_list_max_size
    if use_bounded_lists:
        print(f"Using bounded list encoding (max size: {bounded_list_max_size})")

    puzzle_data = {
        'name': sat_file,
        'sat_function': sat_func,
        'ans_type': ans_type
    }

    if llm_all:
        return solver.solve_puzzle_llm(puzzle_data, llm_solver)
    else:
        return solver.solve_puzzle(puzzle_data, smtlib_backends, llm_solver, llm_end)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sat-file', help='path to a Python file containing a sat function')
    parser.add_argument('--puzzle-file', default="benchmarks/PythonProgrammingPuzzles/puzzles/puzzles.json",
                      help='path to the puzzle JSON file')
    parser.add_argument('--name-prefix',
                        nargs='+',
                        default=[],
                        help='only run puzzles whose names start with this prefix')
    parser.add_argument('--name-suffix',
                        nargs='+',
                        default=[],
                        help='only run puzzles whose names ends with this suffix')
    parser.add_argument('--answer-types',
                        nargs='+',
                        choices=['int', 'str', 'float', 'bool', 'List[int]', 'List[str]', 'List[float]'],
                        default=None,
                        help='only run some answer types (default: all types)')
    parser.add_argument('--smtlib-backends',
                        nargs='+',
                        choices=['z3', 'cvc5'],
                        default=['z3'],
                        help='the SMTLIB backend')
    parser.add_argument('--llm', action='store_true', help='fallback to LLMs')
    parser.add_argument('--llm-all', action='store_true', help='Ask LLMs end-to-end')
    parser.add_argument('--llm-end', action='store_true', help='Ask LLMs end-to-end on success only')
    parser.add_argument('--no-bounded-lists', action='store_true',
                       help='Disable bounded list optimization for lists with known sizes')
    parser.add_argument('--bounded-list-max-size', type=int, default=200,
                       help='Maximum size for bounded lists (default: 200)')
    parser.add_argument('--show-shrunk', action='store_true',
                       help='Show puzzles where the smaller variation was successfully solved')
    args = parser.parse_args()

    llm_solver = None
    if args.llm:
        from holey import llm_generators
        llm_solver = {k: LLMSolver(v) for k,v in llm_generators.items()}

    if args.sat_file:
        solve_sat_file(args.sat_file, args.smtlib_backends, llm_solver, args.llm_all, args.llm_end, not args.no_bounded_lists, args.bounded_list_max_size)
    else:
        run_benchmarks(args.puzzle_file, args.name_prefix, args.name_suffix, args.answer_types, args.smtlib_backends, llm_solver, args.llm_all, args.llm_end, not args.no_bounded_lists, args.bounded_list_max_size, args.show_shrunk)
