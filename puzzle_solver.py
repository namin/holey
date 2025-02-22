from holey import drive_sat, LLMSolver
import copy
import re
import json
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
        self.timeout_staging_count = 0
        self.error_verify_count = 0
        self.error_staging_count = 0
        self.error_smt_count = 0
        self.error_smt_var_count = 0
        self.error_unsupported_answer_type = 0
        self.extrapolate_small_count = 0
        self.extrapolate_small_success_count = 0
        self.extrapolate_large_success_count = 0

    def symbolic_solve1(self, typ, sat_func: str, ans_type: str, name: str, cmds, llm_solver) -> Optional[str]:
        sym_var = drive_sat(sat_func, typ, cmds, llm_solver=llm_solver)
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
        if not typ:
            print("Unsupported answer type", ans_type)
            self.error_unsupported_answer_type += 1
            return None

        if counting:
            self.count += 1
            self.counts[ans_type] += 1
        tracer, sym_var, solution, log = self.symbolic_solve1(typ, sat_func, ans_type, str, cmds, llm_solver=None)
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
        result = typ(str(solution_var))
        print("Found solution", result)
        return result, log

    def solve_puzzle(self, puzzle_data: Any, cmds, llm_solver, reason=None) -> Optional[str]:
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
            result, log = func_timeout(20 if llm_solver else 3, self.symbolic_solve, args=(sat_func, ans_type, name, cmds, llm_solver, not reason))
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
            if not reason and llm_solver and result is None:
                varied_puzzle_sat_func, reason = vary(sat_func)
                if varied_puzzle_sat_func is not None:
                    self.extrapolate_small_count += 1
                    print('Solving simpler variation', reason)
                    varied_puzzle = copy.deepcopy(puzzle_data)
                    varied_puzzle['sat_function'] = varied_puzzle_sat_func
                    varied_result = self.solve_puzzle(varied_puzzle, cmds, llm_solver, reason=reason)
                    if varied_result is not None:
                        self.extrapolate_small_success_count += 1
                        result = llm_solver.extrapolate(varied_puzzle_sat_func, sat_func, reason, varied_result, ans_type, name, check_result)
                        if result is not None:
                            self.extrapolate_large_success_count += 1
                            self.success_count += 1
                            self.success_counts[ans_type] += 1
                            print("Yes! Solved via extrapolation for puzzle ", name)
                            return result
            if False and llm_solver and result is None:
                print('\nFallback to LLM!')
                result = self.llm_solver.solve_end2end(sat_func, ans_type, name, check_result) or self.llm_solver.smtlib_solve(sat_func, ans_type, name, log, check_result, cmds)
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
            return self.llm_solver.solve_end2end(sat_func, ans_type, name, check_result)
        return None

    def pretty_counts(self):
        count_stats = sorted([(self.success_counts[ans_type], total, ans_type) for ans_type,total in self.counts.items()], reverse=True)
        r = ""
        for success, total, ans_type in count_stats:
            success_percentage = 100.0 * success / total
            r += (f"- {success_percentage:.0f}% ({success} out of {total}) of `{ans_type}` puzzles,")
            r += '\n'
        total = self.count
        success = self.success_count
        success_percentage = 100.0 * success / total
        r += (f"- {success_percentage:.0f}% ({success} out of {total}) overall.")
        r += '\n'
        return r

    def pretty_stats(self):
        extrapolation = f"""
### Extrapolation
- {self.extrapolate_small_count} smaller problems tried
- {self.extrapolate_small_success_count} successes on smaller problem
- {self.extrapolate_large_success_count} successful extrapolations
""" if self.extrapolate_small_count > 0 else ""

        return f"""
## Current status

The symbolic execution{'' if self.llm_solver else ' alone'} currently solves:
{self.pretty_counts()}
with the following errors:
- {self.timeout_staging_count} timeouts after 3 seconds at staging time (while generating the SMTLIB program)
- {self.error_staging_count} errors at at staging time
- {self.error_verify_count} SMTLIB programs returning `sat` but the original `sat` function failing on synthesized model input,
- {self.error_smt_count + self.error_smt_var_count} SMTLIB programs returning non-`sat` (e.g. `unsat`, `unknown` or timing out after 2 seconds
timeouts after staging (while building the SMTLIB program), errors during staging time, the SMTLIB
- {self.total_count-self.count} (out of {self.total_count}) puzzles not yet even attempted because their type is not `int` or `str`, such as `float`, `list` (of various specialization), etc.
"""+extrapolation

def check_result(result, sat_func):
    namespace = {}
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

def run_benchmarks(puzzle_file: str, name_prefix = None, answer_types = None, smtlib_backends = None, llm_solver = None):
    with open(puzzle_file) as f:
        puzzles = json.load(f)
    
    total = len(puzzles)
    print(f"Starting with {total} puzzles...")

    # Filter puzzles
    if name_prefix:
        puzzles = [p for p in puzzles if p.get('name', '').startswith(name_prefix)]
    if answer_types:
        puzzles = [p for p in puzzles if p['ans_type'] in answer_types]
        
    solver = PuzzleSolver()
    solver.total_count = total
    solver.llm_solver = llm_solver

    print(f"Running benchmarks on {len(puzzles)} puzzles...")
    if name_prefix:
        print(f"Filtered to puzzles starting with '{name_prefix}'")
    if answer_types:
        print(f"Filtered to puzzles of answer types: {answer_types}")

    for i, puzzle in enumerate(puzzles):
        name = puzzle.get('name', 'Unknown')
        print(f"\nSolving puzzle {i+1}/{len(puzzles)}: {name}")

        result = solver.solve_puzzle(puzzle, smtlib_backends, llm_solver)

    print(solver.pretty_stats())

def vary(sat_func):
    # Find all large constants
    constants = re.findall(r'\d\d\d+', sat_func)

    # Identify unique constants
    unique_constants = set(constants)

    # Only proceed if there is exactly one unique constant
    if len(unique_constants) == 1:
        print('One large constant for extrapolation')
        constant = unique_constants.pop()
        smaller = '3'
        varied_sat_func = re.sub(rf'\b{constant}\b', smaller, sat_func)
        if sat_func != varied_sat_func:
            return varied_sat_func, f'replaced {constant} with {smaller}'
    else:
        print('Too many constants for extrapolation')

    return None, None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--puzzle-file', default="benchmarks/PythonProgrammingPuzzles/puzzles/puzzles.json",
                      help='path to the puzzle JSON file')
    parser.add_argument('--name-prefix', help='only run puzzles whose names start with this prefix')
    parser.add_argument('--answer-types',
                        nargs='+',
                        choices=['int', 'str'],
                        default=['int', 'str'],
                        help='only run some answer types')
    parser.add_argument('--smtlib-backends',
                        nargs='+',
                        choices=['z3', 'cvc5'],
                        default=['z3'],
                        help='the SMTLIB backend')
    parser.add_argument('--llm', action='store_true', help='fallback to LLMs')
    args = parser.parse_args()
    
    llm_solver = None
    if args.llm:
        llm_solver = LLMSolver()
    run_benchmarks(args.puzzle_file, args.name_prefix, args.answer_types, args.smtlib_backends, llm_solver)
