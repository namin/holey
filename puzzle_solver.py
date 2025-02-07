from holey import SymbolicTracer, make_symbolic, SymbolicBool, SymbolicFloat, SymbolicInt, SymbolicRange, SymbolicRangeIterator, SymbolicStr, truthy
from holey.backends import CVC5Backend, Z3Backend, MockBackend
import ast
import json
from func_timeout import func_timeout, FunctionTimedOut
import traceback
from typing import List, Any, Dict, Optional, Tuple
import types

def sym_ord(x):
    if isinstance(x, SymbolicStr):
        return SymbolicInt(x.tracer.backend.StrToCode(x.z3_expr), tracer=x.tracer)
    return ord(x)

def sym_bin(x):
    if isinstance(x, SymbolicInt):
        return SymbolicStr(x.tracer.backend.Bin(x.z3_expr), tracer=x.tracer)
    return bin(x)
                           
def sym_int(x):
    if isinstance(x, SymbolicStr):
        return SymbolicInt(x.tracer.backend.StrToInt(x.z3_expr), tracer=x.tracer)
    if isinstance(x, SymbolicFloat):
        return SymbolicInt(x.tracer.backend.ToInt(x.z3_expr), tracer=x.tracer)
    if isinstance(x, SymbolicInt):
        return x
    return int(x)

def sym_len(x):
    if isinstance(x, SymbolicStr):
        return x.__len__()
    return len(x)

def sym_str(x):
    if isinstance(x, SymbolicInt):
        return SymbolicStr(x.tracer.backend.IntToStr(x.z3_expr), tracer=x.tracer)
    return str(x)

def first_tracer(xs):
    if xs==[]:
        return None
    elif hasattr(xs[0], 'tracer'):
        return xs[0].tracer
    else:
        return first_tracer(xs[1:])

def sym_range(a, b=None, c=None):
    if b is None:
        start = 0
        end = a
        step = 1
    else:
        start = a
        end = b
        step = c if c is not None else 1
        
    if all(isinstance(x, (int, type(None))) for x in (start, end, step)):
        return range(start, end, step) if step is not None else range(start, end)
    
    return SymbolicRange(start, end, step, tracer=first_tracer([start, end, step]))

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
    if isinstance(iterable, types.GeneratorType):
        iterator = iter(iterable.gi_frame.f_locals['.0'])
        if isinstance(iterator, SymbolicRangeIterator):
            predicate = next(iterable)
            bounds = iterator.get_bounds()
            return SymbolicBool(iterator.tracer.backend.Not(
                iterator.tracer.backend.Implies(
                    bounds.z3_expr,
                    iterator.tracer.backend.Not(truthy(predicate).z3_expr)
                )
            ), tracer=iterator.tracer)

    # Default case for concrete lists/other iterables
    conditions = list(iterable)
    if not conditions:
        return False
    result = conditions[0]
    for cond in conditions[1:]:
        result = result.__or__(cond)
    return result

def symbolic_all(iterable):
    if isinstance(iterable, types.GeneratorType):
        iterator = iter(iterable.gi_frame.f_locals['.0'])
        if isinstance(iterator, SymbolicRangeIterator):
            predicate = next(iterable)
            bounds = iterator.get_bounds()
            return SymbolicBool(iterator.tracer.backend.Implies(
                bounds.z3_expr,
                truthy(predicate).z3_expr
            ), tracer=iterator.tracer)

    # Default case for concrete lists/other iterables
    conditions = list(iterable)
    if not conditions:
        return True
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
        node = self.generic_visit(node)
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
        node = self.generic_visit(node)
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
        elif isinstance(node.value, int) and not isinstance(node.value, bool):
            if not any((isinstance(parent, ast.Call) and 
                       ((isinstance(parent.func, ast.Name) and parent.func.id == 'range') or
                        (isinstance(parent.func, ast.Attribute) and parent.func.attr == 'range'))) or
                        isinstance(parent, (ast.Slice, ast.Index))
                       for parent in self.path):
                return ast.Call(
                        func=ast.Name(id='wrap_int', ctx=ast.Load()),
                        args=[ast.Constant(value=node.value)],
                        keywords=[]
                )
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        # Transform: x in y
        # Into: sym_in(x, y)
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.In):
            return ast.Call(
                func=ast.Name(id='sym_in', ctx=ast.Load()),
                args=[node.left, node.comparators[0]],
                keywords=[]
            )
        # Handle chained comparisons like a < b <= c
        elif len(node.ops) > 1:
            values = [node.left] + node.comparators
            result = None
            for i in range(len(node.ops)):
                comp = ast.Compare(
                    left=values[i],
                    ops=[node.ops[i]],
                    comparators=[values[i+1]]
                )
                if result is None:
                    result = comp
                else:
                    result = ast.Call(
                        func=ast.Attribute(
                            value=result,
                            attr='__and__',
                            ctx=ast.Load()
                        ),
                        args=[comp],
                        keywords=[]
                    )
            return result
        return node

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        # Transform: not x
        # Into: sym_not(x)
        if isinstance(node.op, ast.Not):
            return ast.Call(
                func=ast.Name(id='sym_not', ctx=ast.Load()),
                args=[self.visit(node.operand)],
                keywords=[]
            )
        return node

    def visit_BoolOp(self, node):
        """Transform: 
        a and b and c  ->  a.__and__(b).__and__(c)
        a or b or c    ->  a.__or__(b).__or__(c)
        """
        node = self.generic_visit(node)
        values = node.values
        # Build up the chain
        result = values[0]
        for val in values[1:]:
            attr = '__and__' if isinstance(node.op, ast.And) else '__or__'
            result = ast.Call(
                func=ast.Attribute(
                    value=result,
                    attr=attr,
                    ctx=ast.Load()
                ),
                args=[val],
                keywords=[]
            )
        return result

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id in ['int', 'str', 'len', 'range', 'bin', 'ord']:
                return ast.Call(
                    func=ast.Name(id='sym_'+node.func.id, ctx=ast.Load()),
                    args=node.args,
                    keywords=[]
                )
        return node

    def visit_JoinedStr(self, node):
        """Transform f-string into concatenation"""
        node = self.generic_visit(node)
        
        # Convert each part into a string
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                # String literal part
                parts.append(ast.Constant(value=value.value))
            elif isinstance(value, ast.FormattedValue):
                # Expression part {expr}
                parts.append(
                    ast.Call(
                        func=ast.Name(id='str', ctx=ast.Load()),
                        args=[value.value],
                        keywords=[]
                    )
                )
        
        # Join parts with '+'
        result = parts[0]
        for part in parts[1:]:
            result = ast.BinOp(left=result, op=ast.Add(), right=part)
            
        return result

def inject(sat_func):
    tree = ast.parse(sat_func)
    modified_tree = HoleyWrapper().visit(tree)
    modified_func = ast.unparse(modified_tree)
    print('modified_func', modified_func)
    return modified_func

class PuzzleSolver:
    def __init__(self):
        self.backend = MockBackend()
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
            'SymbolicInt': SymbolicInt,
            'wrap_str': lambda s: SymbolicStr(s, tracer=tracer),
            'wrap_int': lambda n: SymbolicInt(n, tracer=tracer),
            '_assert': lambda x, msg=None: tracer.add_constraint(x),
            'any': symbolic_any,
            'all': symbolic_all,
            'sym_not': symbolic_not,
            'sym_in': symbolic_in,
            'sym_str': sym_str,
            'sym_int': sym_int,
            'sym_len': sym_len,
            'sym_range': sym_range,
            'sym_bin': sym_bin,
            'sym_ord': sym_ord
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
            print('Solution', solution)
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
            result = func_timeout(3, self.symbolic_solve, args=(sat_func, ans_type))
            if result is not None:
                namespace = {'x': result}
                exec(sat_func, namespace)
                sat = namespace['sat']
                if not sat(int(result)):
                    print("WARNING: Solution verification failed!")
                    return None
                else:
                    print("Yes! Solved", name)
            return result
        except FunctionTimedOut:
            print("Timed out")
        except Exception as e:
            print("Exception: ", e)
            traceback.print_exc()
        return None

def run_benchmarks(puzzle_file: str, name_prefix: str = None):
    with open(puzzle_file) as f:
        puzzles = json.load(f)
    
    # Filter puzzles if name_prefix is provided
    if name_prefix:
        puzzles = [p for p in puzzles if p.get('name', '').startswith(name_prefix)]
        
    solver = PuzzleSolver()
    success_count = 0

    print(f"Running benchmarks on {len(puzzles)} puzzles...")
    if name_prefix:
        print(f"Filtered to puzzles starting with '{name_prefix}'")

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
    success_percentage = 100.0 * success_count / solver.count if solver.count > 0 else 0
    print(f"Success percentage: {success_percentage:.0f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--puzzle-file', default="benchmarks/PythonProgrammingPuzzles/puzzles/puzzles.json",
                      help='Path to the puzzle JSON file')
    parser.add_argument('--name-prefix', help='Only run puzzles whose names start with this prefix')
    args = parser.parse_args()
    
    run_benchmarks(args.puzzle_file, args.name_prefix)
