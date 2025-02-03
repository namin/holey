from holey import SymbolicTracer, make_symbolic, SymbolicStr
from holey.backends import Z3Backend, MockBackend
import ast
import json
from func_timeout import func_timeout, FunctionTimedOut
import traceback
from typing import List, Any, Dict, Optional, Tuple

class StringWrapper(ast.NodeTransformer):
    def __init__(self):
        self.path = []

    def visit(self, node):
        self.path.append(node)
        result = super().visit(node)
        self.path.pop()
        return result
        
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            # Check if we're in a count() call
            if not any(isinstance(parent, ast.Call) and 
                      isinstance(parent.func, ast.Attribute) and 
                      parent.func.attr == 'count' 
                      for parent in self.path):
                return ast.Call(
                    func=ast.Name(id='wrap_str', ctx=ast.Load()),
                    args=[ast.Constant(value=node.value)],
                    keywords=[]
                )
        return node

def inject(sat_func):
    tree = ast.parse(sat_func)
    modified_tree = StringWrapper().visit(tree)
    modified_func = ast.unparse(modified_tree)
    return modified_func

class PuzzleSolver:
    def __init__(self):
        self.backend = Z3Backend()
        self.tracer = SymbolicTracer(backend=self.backend)

    def symbolic_solve(self, sat_func: str, ans_type: str) -> Optional[str]:
        typ = None
        if ans_type == 'int':
            typ = int
        if not typ:
            print("Unsupported answer type", ans_type)
            return None

        namespace = {
            'tracer': self.tracer,
            'SymbolicStr': SymbolicStr,
            'wrap_str': lambda s: SymbolicStr(s, tracer=self.tracer)
        }
        sym_var = make_symbolic(int, 'x', self.tracer)
        namespace['x'] = sym_var
        exec(inject(sat_func), namespace)
        sat = namespace['sat']
        result = sat(sym_var)
        solution = self.tracer.solution()
        if solution is None:
            print("Could not find any solution")
            return None
        solution_var = self.tracer.solution_var(solution, sym_var)
        if solution_var is None:
            print("Could not find any solution var")
            return None
        result = str(solution_var)
        print("Found solution", result)
        return result

    def solve_puzzle(self, puzzle_data: Any) -> Optional[str]:
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
            return func_timeout(2, self.symbolic_solve, args=(sat_func, ans_type))
        except FunctionTimedOut:
            print("Timed out")
        except Exception as e:
            traceback.print_exc()
        return None

def run_benchmarks(puzzle_file: str):
    with open(puzzle_file) as f:
        puzzles = json.load(f)
    solver = PuzzleSolver()
    success_count = 0

    print(f"Running benchmarks on {len(puzzles)} puzzles...")

    for i, puzzle in enumerate(puzzles):
        name = puzzle.get('name', 'Unknown')
        print(f"\nSolving puzzle {i+1}/{len(puzzles)}: {name}")

        result = solver.solve_puzzle(puzzle)
        if result:
            success_count += 1
    
    print("Success count", success_count)

if __name__ == "__main__":
    puzzle_file = "benchmarks/PythonProgrammingPuzzles/puzzles/puzzles.json"
    run_benchmarks(puzzle_file)
