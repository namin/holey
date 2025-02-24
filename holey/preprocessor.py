from .core import SymbolicTracer, make_symbolic, SymbolicBool, SymbolicFloat, SymbolicInt, SymbolicList, SymbolicRange, SymbolicRangeIterator, SymbolicStr, SymbolicSlice, truthy
from .backend import default_backend
import ast
from typing import List, Any, Dict, Optional, Tuple
import types
import itertools

# Counter for generating unique variable names
counter = itertools.count()
def reset():
    global counter
    counter = itertools.count()

class SymbolicGenerator:
    """Wrapper for generator expressions to maintain symbolic evaluation"""
    def __init__(self, iterator, comparison):
        self.iterator = iterator
        self.comparison = comparison  # Store the comparison expression
    
    def __iter__(self):
        return self

    def __next__(self):
        if not isinstance(self.iterator, SymbolicZipIterator):
            return next(self.iterator)
        if self.iterator.used:
            raise StopIteration
        return self.comparison

def sym_generator(gen):
    """Convert a generator expression to a symbolic generator"""
    if isinstance(gen, types.GeneratorType):
        frame = gen.gi_frame
        iterator = iter(frame.f_locals['.0'])  # Get underlying iterator
        if isinstance(iterator, SymbolicZipIterator):
            comparison = next(gen)  # Get comparison expression
            return SymbolicGenerator(iterator, comparison)
    return gen

def sym_sum(iterable):
    """Symbolic summation that maintains symbolic operations"""
    if isinstance(iterable, SymbolicGenerator):
        iterator = iterable.iterator
        if isinstance(iterator, SymbolicZipIterator):
            # Get the comparison from the generator
            comparison = iterable.comparison
            tracer = iterator.tracer
            term = tracer.backend.If(
                truthy(comparison).z3_expr,
                tracer.backend.IntVal(1),
                tracer.backend.IntVal(0)
            )
            return SymbolicInt(term, tracer=tracer)
    
    # For non-symbolic case
    return sum(iterable)

class SymbolicZipIterator:
    def __init__(self, iterables, tracer):
        self.iterables = iterables
        self.tracer = tracer
        self.used = False
        self.pos = None
        self.bounds = None
            
    def __iter__(self):
        return self
            
    def __next__(self):
        if self.used:
            raise StopIteration
            
        # Create symbolic index
        name = f"zip_pos_{next(counter)}"
        self.tracer.backend.quantified_vars.add(name)
        self.pos = make_symbolic(int, name, tracer=self.tracer)
        
        # Get min length
        min_length = self.iterables[0].__len__()
        for it in self.iterables[1:]:
            length = it.__len__()
            min_length = SymbolicInt(
                self.tracer.backend.If(
                    min_length.z3_expr < length.z3_expr,
                    min_length.z3_expr,
                    length.z3_expr
                ),
                tracer=self.tracer
            )
        
        # Create bounds condition
        self.bounds = (self.pos >= 0).__and__(self.pos < min_length)
        
        # Add to forall conditions
        self.tracer.forall_conditions.append((self.pos.z3_expr, self.bounds.z3_expr))
        
        # Let each iterable's __getitem__ handle the indexing
        elements = [it[self.pos] if hasattr(it, '__getitem__') else it 
                   for it in self.iterables]
        
        self.used = True
        return tuple(elements)

def sym_zip(*iterables):
    """Symbolic version of zip that works with symbolic sequences"""
    # Handle empty case
    if not iterables:
        return []

    # Get first symbolic tracer we find
    tracer = first_tracer(iterables)
    if tracer is None:
        return zip(*iterables)  # If no symbolic values, use regular zip

    # Convert all items to symbolic form
    symbolic_iterables = [tracer.ensure_symbolic(it) for it in iterables]
    
    # If all concrete, use regular zip
    if all(hasattr(it, 'concrete') and it.concrete is not None for it in symbolic_iterables):
        return zip(*[it.concrete for it in symbolic_iterables])

    return SymbolicZipIterator(symbolic_iterables, tracer)

def sym_ord(x):
    if isinstance(x, SymbolicStr):
        return SymbolicInt(x.tracer.backend.StrToCode(x.z3_expr), tracer=x.tracer)
    if isinstance(x, SymbolicSlice):
        return sym_ord(x.get_slice())

    return ord(x)

def sym_bin(x):
    if isinstance(x, SymbolicInt):
        return SymbolicStr(x.tracer.backend.Bin(x.z3_expr), tracer=x.tracer)
    return bin(x)

def valid_int_inputs(x, b, tracer):
    tracer.add_constraint(tracer.backend.GT(tracer.backend.StrLen(x), 0))
                   
def sym_int(x, base=None):
    if isinstance(x, SymbolicStr):
        if x.concrete is not None and not base:
            return SymbolicInt(int(x.concrete), tracer=x.tracer)
        if base:
            base = x.tracer.ensure_symbolic(base).z3_expr
        args = (x.z3_expr, base)
        valid_int_inputs(*args, tracer=x.tracer)
        return SymbolicInt(x.tracer.backend.StrToInt(*args), tracer=x.tracer)
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
        return SymbolicStr(str(x.concrete) if x.concrete is not None else x.tracer.backend.IntToStr(x.z3_expr), tracer=x.tracer)
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

    if start.concrete is not None and stop.concrete is not None and step.concrete is not None:
        return range(start.concrete, stop.concrete, step.concrete)

    name = f"i_{next(counter)}"
    tracer.backend.quantified_vars.add(name)
    i = make_symbolic(int, name, tracer=tracer)
    
    # Create bounds condition
    bounds = (i >= start).__and__(i < stop)
    if step != 1:
        k = make_symbolic(int, f"k_{next(counter)}", tracer=tracer)
        bounds = bounds.__and__(k >= 0).__and__(
            i == start + k * step).__and__(
            k < (stop - start) // step)
    
    # Add to forall conditions instead of direct constraints
    tracer.forall_conditions.append((i.z3_expr, bounds.z3_expr))
    return [i]

def sym_not(x):
    return x.__not__()

def sym_in(x, container):
    return container.contains(x)

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
            if node.func.id in ['int', 'float', 'str', 'len', 'range', 'bin', 'ord', 'sum', 'zip']:
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

    def visit_GeneratorExp(self, node):
        """Transform: (expr for x in iter)
        Into: sym_generator(expr for x in iter)
        """
        node = self.generic_visit(node)
        return ast.Call(
            func=ast.Name(id='sym_generator', ctx=ast.Load()),
            args=[node],
            keywords=[]
        )

def inject(sat_func):
    tree = ast.parse(sat_func)
    modified_tree = HoleyWrapper().visit(tree)
    modified_func = ast.unparse(modified_tree)
    print('modified_func', modified_func)
    return modified_func

def create_namespace(tracer):
    return {
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
        'sym_ord': sym_ord,
        "sym_sum": sym_sum,
        'sym_zip': sym_zip,
        'sym_generator': sym_generator
    }

def driver(sat_func, typ, cmds=None, llm_solver=None):
    reset()
    backend = default_backend(cmds)
    tracer = SymbolicTracer(backend=backend, llm_solver=llm_solver)
    namespace = create_namespace(tracer)
    sym_var = make_symbolic(typ, 'x', tracer)
    namespace['x'] = sym_var
    exec(inject(sat_func), namespace)
    sat = namespace['sat']
    tracer.driver(lambda: sat(sym_var))
    return sym_var
