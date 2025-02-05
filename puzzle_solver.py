from holey import SymbolicTracer, make_symbolic, SymbolicBool, SymbolicInt, SymbolicStr
from holey.backends import CVC5Backend, Z3Backend, MockBackend
import ast
import json
from func_timeout import func_timeout, FunctionTimedOut
import traceback
from typing import List, Any, Dict, Optional, Tuple

def symbolic_not(x):
    if isinstance(x, SymbolicBool):
        return SymbolicBool(x.tracer.backend.Not(x.z3_expr), x.tracer)
    return not x

def symbolic_in(x, container):
    if isinstance(x, SymbolicInt):
        # Create disjunction of equalities
        equalities = [x == val for val in container]
        result = equalities[0]
        for eq in equalities[1:]:
            result = result.__or__(eq)
        return result
    return x in container

def symbolic_any(iterable):
    # For generator expressions, convert to list
    conditions = list(iterable)
    if not conditions:
        return SymbolicBool(False)
    # Create disjunction
    result = conditions[0]
    for cond in conditions[1:]:
        result = result.__or__(cond)
    return result

def symbolic_all(iterable):
    conditions = list(iterable)
    if not conditions:
        return SymbolicBool(True)
    result = conditions[0]
    for cond in conditions[1:]:
        result = result.__and__(cond)
    return result

class HoleyWrapper(ast.NodeTransformer):
    def __init__(self):
        self.path = []

    def visit(self, node):
        self.path.append(node)
        result = super().visit(node)
        self.path.pop()
        return result

    def visit_Assert(self, node):
        # Convert: assert test [, msg]
        # Into: _assert(test [, msg])
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id='_assert', ctx=ast.Load()),
                args=[node.test] + ([node.msg] if node.msg else []),
                keywords=[]
            )
        )
        
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            if not any(isinstance(parent, ast.Call) and 
                       isinstance(parent.func, ast.Attribute) and 
                       (parent.func.attr in ['count', 'startswith'])
                       for parent in self.path):
                return ast.Call(
                    func=ast.Name(id='wrap_str', ctx=ast.Load()),
                    args=[ast.Constant(value=node.value)],
                    keywords=[]
                )
        return node

    def visit_Compare(self, node):
        # Transform: x in y
        # Into: sym_in(x, y)
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.In):
            return ast.Call(
                func=ast.Name(id='sym_in', ctx=ast.Load()),
                args=[node.left, node.comparators[0]],
                keywords=[]
            )
        return node

    def visit_UnaryOp(self, node):
        # Transform: not x
        # Into: sym_not(x)
        if isinstance(node.op, ast.Not):
            return ast.Call(
                func=ast.Name(id='sym_not', ctx=ast.Load()),
                args=[self.visit(node.operand)],
                keywords=[]
            )
        return node

def inject(sat_func):
    tree = ast.parse(sat_func)
    modified_tree = HoleyWrapper().visit(tree)
    modified_func = ast.unparse(modified_tree)
    return modified_func

class PuzzleSolver:
    def __init__(self):
        self.backend = Z3Backend()
        self.count = 0

    def new_tracer(self):
        self.backend.reset()
        return SymbolicTracer(backend=self.backend)

    def symbolic_solve(self, sat_func: str, ans_type: str) -> Optional[str]:
        typ = None
        if ans_type == 'int':
            typ = int
        if not typ:
            print("Unsupported answer type", ans_type)
            return None

        self.tracer = self.new_tracer()
        tracer = self.tracer
        
        self.count += 1
        namespace = {
            'tracer': tracer,
            'SymbolicStr': SymbolicStr,
            'wrap_str': lambda s: SymbolicStr(s, tracer=tracer),
            '_assert': lambda x, msg=None: tracer.add_constraint(x),
            'any': symbolic_any,
            'all': symbolic_all,
            'sym_not': symbolic_not,
            'sym_in': symbolic_in
        }
        sym_var = make_symbolic(typ, 'x', tracer)
        namespace['x'] = sym_var
        exec(inject(sat_func), namespace)
        sat = namespace['sat']
        result = sat(sym_var)
        tracer.add_constraint(result)
        solution = tracer.solution()
        if solution is None:
            print("Could not find any solution")
            return None
        solution_var = tracer.solution_var(solution, sym_var)
        if solution_var is None:
            print("Could not find any solution var")
            return None
        result = str(solution_var)
        print("Found solution", result)
        return result

    def solve_puzzle(self, puzzle_data: Any) -> Optional[str]:
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
            return func_timeout(2, self.symbolic_solve, args=(sat_func, ans_type))
        except FunctionTimedOut:
            print("Timed out")
        except Exception as e:
            print("Exception: ", e)
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
    
    print('')
    print('STATS')
    print("Success count:", success_count)
    print("Total considered:", solver.count)
    success_percentage = 100.0 * success_count / solver.count
    print(f"Success percentage: {success_percentage:.0f}%")

if __name__ == "__main__":
    puzzle_file = "benchmarks/PythonProgrammingPuzzles/puzzles/puzzles.json"
    run_benchmarks(puzzle_file)
