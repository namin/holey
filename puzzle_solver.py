from holey import llm_generate, extract_code_blocks, run_smt, drive_sat
import json
from func_timeout import func_timeout, FunctionTimedOut
import traceback
from typing import List, Any, Dict, Optional, Tuple
import sys
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
        self.success_count = 0
        self.timeout_staging_count = 0
        self.error_verify_count = 0
        self.error_staging_count = 0
        self.error_smt_count = 0
        self.error_smt_var_count = 0
        self.error_unsupported_answer_type = 0

    def symbolic_solve(self, sat_func: str, ans_type: str, name: str, cmds) -> Optional[str]:
        typ = None
        if ans_type == 'int':
            typ = int
        elif ans_type == 'str':
            typ = str
        if not typ:
            print("Unsupported answer type", ans_type)
            self.error_unsupported_answer_type += 1
            return None

        self.count += 1
        sym_var = drive_sat(sat_func, typ, cmds)
        tracer = sym_var.tracer
        with capture_output() as captured:
            solution = tracer.solution()
        log = captured.getvalue()
        print(log)
        if solution is None:
            print("Could not find any solution for puzzle " + name)
            self.error_smt_count += 1
            return None, log
        solution_var = tracer.solution_var(solution, sym_var)
        if solution_var is None:
            self.error_smt_var_count += 1
            print('Solution', solution)
            print("Could not find any solution var")
            return None, log
        result = typ(str(solution_var))
        print("Found solution", result)
        return result, log

    def llm_smtlib_solve(self, sat_func: str, ans_type: str, name: str, log: str, cmds=None) -> Optional[str]:
        print('Asking LLM for SMTLIB')
        prompt = f"""Return a modified SMTLIB z3 program that captures the intent of the `sat` function of puzzle {name}:
{sat_func}

This is the log, you may copy most of any SMTLIB program below.
{log}

Return only the new SMTLIB program without any context.
"""
        blocks = extract_code_blocks(llm_generate(prompt))
        model = None
        result = None
        flag = None
        for smt in blocks:
            flag, model = run_smt(smt, cmds)
            if flag == "sat":
                break
        if model:
            llm_result = model['x']
            if check_result(llm_result, sat_func):
                print("LLM result confirmed for puzzle " + name)
                result = llm_result
        return None

    def llm_solve(self, sat_func: str, ans_type: str, name: str) -> Optional[str]:
        print('Asking LLM for whole answer')
        prompt = f"""Return a constant Python value of type {ans_type} to solve puzzle {name}, where your goal is to synthesize the first argument that makes this `sat` function return `True`:
{sat_func}

Return only the Python constant without any context.
"""
        results = extract_code_blocks(llm_generate(prompt))
        for result in results:
            print('LLM result', result)
            if ans_type == 'int':
                try:
                    result = int(result)
                except ValueError as e:
                    print('LLM returned bad type for int', e)
                    break
            elif ans_type == 'str':
                if result and result[0] in ["'", '"']:
                    result = result[1:-1] # TODO
            if not check_result(result, sat_func):
                print('LLM result fails to verify for puzzle '+name)
            else:
                print('LLM result verifies for puzzle '+name)
                return result
        return None


    def solve_puzzle(self, puzzle_data: Any, cmds, llm) -> Optional[str]:
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
            result, log = func_timeout(3, self.symbolic_solve, args=(sat_func, ans_type, name, cmds))
            if result is not None:
                if not check_result(result, sat_func):
                    self.error_verify_count += 1
                    print("WARNING: Solution verification failed for puzzle "+name)
                    result = None
                else:
                    self.success_count += 1
                    print("Yes! Solved for puzzle ", name)
            if llm and result is None:
                print('\nFallback to LLM!')
                result = self.llm_solve(sat_func, ans_type, name) or self.llm_smtlib_solve(sat_func, ans_type, name, log, cmds)
            return result
        except FunctionTimedOut:
            print("Timed out for puzzle "+name)
            self.timeout_staging_count += 1
        except Exception as e:
            self.error_staging_count += 1
            print("Exception -- for puzzle", name, e)
            traceback.print_exc()
        if llm:
            print('\nFallback to LLM after error!')
            return self.llm_solve(sat_func, ans_type, name)
        return None

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

def run_benchmarks(puzzle_file: str, name_prefix = None, answer_types = None, smtlib_backends = None, llm = False):
    with open(puzzle_file) as f:
        puzzles = json.load(f)
    
    print(f"Starting with {len(puzzles)} puzzles...")

    # Filter puzzles
    if name_prefix:
        puzzles = [p for p in puzzles if p.get('name', '').startswith(name_prefix)]
    if answer_types:
        puzzles = [p for p in puzzles if p['ans_type'] in answer_types]
        
    solver = PuzzleSolver()
    success_count = 0

    print(f"Running benchmarks on {len(puzzles)} puzzles...")
    if name_prefix:
        print(f"Filtered to puzzles starting with '{name_prefix}'")
    if answer_types:
        print(f"Filtered to puzzles of answer types: {answer_types}")

    for i, puzzle in enumerate(puzzles):
        name = puzzle.get('name', 'Unknown')
        print(f"\nSolving puzzle {i+1}/{len(puzzles)}: {name}")

        result = solver.solve_puzzle(puzzle, smtlib_backends, llm)
        if result is not None:
            success_count += 1
    
    print('')
    print('STATS')
    print("Success count:", success_count)
    print("Total considered:", solver.count)
    print("\n")
    print("timeout staging count", solver.timeout_staging_count)
    print("error staging count", solver.error_staging_count)
    print("error verify count", solver.error_verify_count)
    print("error smt count", solver.error_smt_count)
    print("error smt var count", solver.error_smt_var_count)
    print("unsupported answer type", solver.error_unsupported_answer_type)
    
    success_percentage = 100.0 * success_count / solver.count if solver.count > 0 else 0
    print(f"Success percentage: {success_percentage:.0f}%")

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
    
    run_benchmarks(args.puzzle_file, args.name_prefix, args.answer_types, args.smtlib_backends, args.llm)
