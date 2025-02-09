from holey import SymbolicTracer, make_symbolic, SymbolicBool, SymbolicFloat, SymbolicInt, SymbolicList,SymbolicRange, SymbolicRangeIterator, SymbolicStr, truthy, llm_generate, extract_code_blocks, run_smt
from holey.backends import CVC5Backend, Z3Backend, MockBackend
import ast
import json
from func_timeout import func_timeout, FunctionTimedOut
import traceback
from typing import List, Any, Dict, Optional, Tuple
import types
import itertools

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

# Counter for generating unique variable names
counter = itertools.count()

def sym_ord(x):
    if isinstance(x, SymbolicStr):
        return SymbolicInt(x.tracer.backend.StrToCode(x.z3_expr), tracer=x.tracer)
    return ord(x)

def sym_bin(x):
    if isinstance(x, SymbolicInt):
        return SymbolicStr(x.tracer.backend.Bin(x.z3_expr), tracer=x.tracer)
    return bin(x)
                           
def sym_int(x, base=None):
    if isinstance(x, SymbolicStr):
        if base:
            base = x.ensure_symbolic(base).z3_expr
        return SymbolicInt(x.tracer.backend.StrToInt(x.z3_expr, base), tracer=x.tracer)
    if isinstance(x, SymbolicFloat):
        assert base is None
        return SymbolicInt(x.tracer.backend.ToInt(x.z3_expr), tracer=x.tracer)
    if isinstance(x, SymbolicInt):
        assert base is None
        return x
    if base:
        return int(x, base)
    else:
        return int(x)

def sym_float(x):
    if isinstance(x, SymbolicStr):
        return SymbolicFloat(x.tracer.backend.StrToFloat(x.z3_expr), tracer=x.tracer)
    if isinstance(x, SymbolicFloat):
        return x
    if isinstance(x, SymbolicInt):
        return SymbolicFloat(x.tracer.backend.IntToFloat(x.z3_expr), tracer=x.tracer)
    return float(x)

def sym_len(x):
    if isinstance(x, SymbolicStr):
        return x.__len__()
    return len(x)

def sym_str(x):
    if isinstance(x, SymbolicInt):
        return SymbolicStr(str(x.concrete) if x.concrete else x.tracer.backend.IntToStr(x.z3_expr), tracer=x.tracer)
    if isinstance(x, SymbolicStr):
        return x
    return x.__str__()

def first_tracer(xs):
    if xs==[]:
        return None
    elif hasattr(xs[0], 'tracer'):
        return xs[0].tracer
    else:
        return first_tracer(xs[1:])

def sym_range(*args):
    """Symbolic version of range that adds proper bounds"""
    if all(isinstance(arg, int) for arg in args):
        return range(*args)

    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    elif len(args) == 2:
        start, stop = args
        step = 1
    else:
        start, stop, step = args
    
    tracer = first_tracer(args)
    [start, stop, step] = [tracer.ensure_symbolic(arg) for arg in [start, stop, step]]
    i = make_symbolic(int, f"i_{next(counter)}", tracer=tracer)
    # Add range constraints
    i.tracer.add_constraint(i >= start)
    i.tracer.add_constraint(i < stop)
    if step != 1:
        k = make_symbolic(int, f"k_{next(counter)}", tracer=tracer)
        i.tracer.add_constraint(k >= 0)
        i.tracer.add_constraint(i == start + k * step)
        i.tracer.add_constraint(k < (stop - start) // step)
    return [i]

def sym_not(x):
    return x.__not__()

def sym_in(x, container):
    if isinstance(x, SymbolicInt):
        # Create disjunction of equalities
        equalities = [x == val for val in container]
        result = equalities[0]
        for eq in equalities[1:]:
            result = result.__or__(eq)
        return result
    if isinstance(container, SymbolicStr):
        return container.__contains__(x)
    return x in container

def sym_any(iterable):
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

def sym_all(iterable):
    """Handle all() for symbolic values"""
    if isinstance(iterable, types.GeneratorType):
        # Convert generator to list and ensure boolean conditions
        iterable = [x for x in list(iterable)]
    result = None
    for item in iterable:
        item = truthy(item)
        if result is None:
            result = item
        else:
            result = result.__and__(item)
    return result if result is not None else True

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

    def visit_List(self, node):
        node = self.generic_visit(node)
        return ast.Call(
            func=ast.Name(id='wrap_list', ctx=ast.Load()),
            args=[node],
            keywords=[]
        )

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        # Transform: not x
        # Into: sym_not(x)
        if isinstance(node.op, ast.Not):
            return ast.Call(
                func=ast.Name(id='sym_not', ctx=ast.Load()),
                args=[node.operand],
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
            if node.func.id in ['int', 'float', 'str', 'len', 'range', 'bin', 'ord']:
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
            if isinstance(value, ast.FormattedValue):
                # Expression part {expr}
                parts.append(
                    ast.Call(
                        func=ast.Name(id='sym_str', ctx=ast.Load()),
                        args=[value.value],
                        keywords=[]
                    )
                )
            else:
                parts.append(value)
        
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
        self.success_count = 0
        self.timeout_staging_count = 0
        self.error_verify_count = 0
        self.error_staging_count = 0
        self.error_smt_count = 0
        self.error_smt_var_count = 0
        self.error_unsupported_answer_type = 0

    def new_tracer(self):
        self.backend.reset()
        return SymbolicTracer(backend=self.backend)

    def symbolic_solve(self, sat_func: str, ans_type: str, name: str) -> Optional[str]:
        typ = None
        if ans_type == 'int':
            typ = int
        elif ans_type == 'str':
            typ = str
        if not typ:
            print("Unsupported answer type", ans_type)
            self.error_unsupported_answer_type += 1
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
            'wrap_list': lambda l: SymbolicList(l, tracer=tracer),
            '_assert': lambda x, msg=None: tracer.add_constraint(x),
            'any': sym_any,
            'all': sym_all,
            'sym_not': sym_not,
            'sym_in': sym_in,
            'sym_str': sym_str,
            'sym_int': sym_int,
            'sym_float': sym_float,
            'sym_len': sym_len,
            'sym_range': sym_range,
            'sym_bin': sym_bin,
            'sym_ord': sym_ord
        }
        sym_var = make_symbolic(typ, 'x', tracer)
        namespace['x'] = sym_var
        exec(inject(sat_func), namespace)
        sat = namespace['sat']
        tracer.driver(lambda: sat(sym_var))
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

    def solve_puzzle(self, puzzle_data: Any, llm) -> Optional[str]:
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
            result, log = func_timeout(3, self.symbolic_solve, args=(sat_func, ans_type, name))
            if result is not None:
                if not check_result(result, sat_func):
                    self.error_verify_count += 1
                    print("WARNING: Solution verification failed for puzzle "+name)
                    return None
                else:
                    self.success_count += 1
                    print("Yes! Solved for puzzle ", name)
            elif llm:
                print('\nFallback to LLM!')
                prompt = f"""Return a modified SMTLIB z3 program that captures the intent of the `sat` function of puzzle {name}:
{sat_func}

This is the log, you may copy most of any SMTLIB program below.
{log}

Return only the new SMTLIB program without any context.
"""
                blocks = extract_code_blocks(llm_generate(prompt))
                model = None
                for smt in blocks:
                    flag, model = run_smt(smt)
                    if flag == "sat":
                        break
                if model:
                    llm_result = model['x']
                    if check_result(llm_result, sat_func):
                        print("LLM result confirmed for puzzle " + name)
                        result = llm_result
            return result
        except FunctionTimedOut:
            print("Timed out for puzzle "+name)
            self.timeout_staging_count += 1
            print("Timed out")
        except Exception as e:
            self.error_staging_count += 1
            print("Exception -- for puzzle", name, e)
            traceback.print_exc()
        return None

def check_result(result, sat_func):
    namespace = {}
    exec(sat_func, namespace)
    sat = namespace['sat']
    outcome = sat(result)
    if not outcome:
        return False
    return True

def run_benchmarks(puzzle_file: str, name_prefix = None, answer_types = None, llm = False):
    with open(puzzle_file) as f:
        puzzles = json.load(f)
    
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

        result = solver.solve_puzzle(puzzle, llm)
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
                      help='Path to the puzzle JSON file')
    parser.add_argument('--name-prefix', help='Only run puzzles whose names start with this prefix')
    parser.add_argument('--answer-types',
                        nargs='+',
                        choices=['int', 'str'],
                        default=['int', 'str'],
                        help='Only run some answer types')
    parser.add_argument('--llm', action='store_true')
    args = parser.parse_args()
    
    run_benchmarks(args.puzzle_file, args.name_prefix, args.answer_types, args.llm)
