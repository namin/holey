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

def sym_sorted(s):
    if isinstance(s, SymbolicStr):
        if s.concrete is not None:
            return SymbolicStr("".join(sorted(s.concrete)), tracer=s.tracer)
        return SymbolicStr(s.tracer.backend.StrSorted(s.z3_expr), tracer=s.tracer)
    return sorted(s)

class SymbolicZipIterator:
    def __init__(self, iterables, tracer):
        self.iterables = iterables
        self.tracer = tracer
        self.used = False
        self.pos = None
        
        # Calculate length once since strings are equal length
        self.length = sym_len(iterables[0])
            
    def __iter__(self):
        return self
            
    def __next__(self):
        if self.used:
            raise StopIteration
            
        # Create symbolic index
        name = f"zip_pos_{next(counter)}"
        self.tracer.backend.quantified_vars.add(name)
        self.pos = make_symbolic(int, name, tracer=self.tracer)
        
        # Create character extraction for strings
        elements = []
        for it in self.iterables:
            if isinstance(it, SymbolicStr):
                elements.append(
                    SymbolicStr(
                        self.tracer.backend.StrIndex(it.z3_expr, self.pos.z3_expr),
                        tracer=self.tracer
                    )
                )
            elif isinstance(it, str):
                elements.append(
                    SymbolicStr(
                        self.tracer.backend.StrIndex(
                            self.tracer.backend.StringVal(it),
                            self.pos.z3_expr
                        ),
                        tracer=self.tracer
                    )
                )
            else:
                elements.append(it[self.pos] if hasattr(it, '__getitem__') else it)
        
        self.used = True
        return tuple(elements)

def sym_sum(iterable):
    """Symbolic summation that maintains symbolic operations"""
    if isinstance(iterable, SymbolicGenerator):
        iterator = iterable.iterator
        if isinstance(iterator, SymbolicZipIterator):
            comparison = iterable.comparison
            tracer = iterator.tracer
            
            # Create sum variable
            sum_var = make_symbolic(int, f"sum_{next(counter)}", tracer=tracer)
            
            if len(iterator.iterables) == 2:
                s1, s2 = iterator.iterables
                length = iterator.length
                
                # Extract the concrete length if one of the strings has it
                concrete_length = None
                for s in (s1, s2):
                    if isinstance(s, str):
                        concrete_length = len(s)
                        break
                    if hasattr(s, 'concrete') and isinstance(s.concrete, str):
                        concrete_length = len(s.concrete)
                        break
                
                if concrete_length is not None:
                    # If we have a concrete length, use it for enumeration
                    diffs = []
                    for i in range(concrete_length):
                        pos = tracer.backend.IntVal(i)
                        diff = tracer.backend.If(
                            tracer.backend.Not(tracer.backend.Eq(
                                tracer.backend.StrIndex(s1.z3_expr, pos),
                                tracer.backend.StrIndex(s2.z3_expr, pos)
                            )),
                            tracer.backend.IntVal(1),
                            tracer.backend.IntVal(0)
                        )
                        diffs.append(diff)
                    
                    # Sum must equal total differences
                    tracer.add_constraint(
                        sum_var.z3_expr == tracer.backend.Add(*diffs)
                    )
                else:
                    # For fully symbolic case, use a counter and forall
                    cnt_name = f"cnt_{next(counter)}"
                    tracer.backend.quantified_vars.add(cnt_name)
                    cnt = make_symbolic(int, cnt_name, tracer=tracer)
                    
                    # Bounds and difference at current position
                    bounds = tracer.backend.And(
                        cnt.z3_expr >= 0,
                        cnt.z3_expr < length.z3_expr
                    )
                    
                    diff = tracer.backend.If(
                        tracer.backend.Not(tracer.backend.Eq(
                            tracer.backend.StrIndex(s1.z3_expr, cnt.z3_expr),
                            tracer.backend.StrIndex(s2.z3_expr, cnt.z3_expr)
                        )),
                        tracer.backend.IntVal(1),
                        tracer.backend.IntVal(0)
                    )
                    
                    # Use different constraint form for symbolic case
                    tracer.add_constraint(
                        tracer.backend.ForAll(
                            [cnt.z3_expr],
                            tracer.backend.Implies(
                                bounds,
                                tracer.backend.And(
                                    sum_var.z3_expr >= diff,
                                    sum_var.z3_expr <= length.z3_expr
                                )
                            )
                        )
                    )
            
            return sum_var
        
        # Handle boolean conditions
        conditions = list(iterable)
        if conditions:
            # Find first condition with a tracer
            tracer = None
            for cond in conditions:
                if hasattr(cond, 'tracer'):
                    tracer = cond.tracer
                    break
            
            if tracer:
                terms = []
                for cond in conditions:
                    # Ensure condition is symbolic boolean
                    cond = tracer.ensure_symbolic(cond)
                    if not isinstance(cond, SymbolicBool):
                        cond = truthy(cond)
                    
                    # Convert to 0/1 integer
                    term = tracer.backend.If(
                        cond.z3_expr,
                        tracer.backend.IntVal(1),
                        tracer.backend.IntVal(0)
                    )
                    terms.append(term)
                
                # Create final sum as one operation
                return SymbolicInt(tracer.backend.Add(*terms), tracer=tracer)
    
    # For non-symbolic case
    return sum(iterable)

def sym_generator(gen):
    """Convert a generator expression to a symbolic generator"""
    if isinstance(gen, types.GeneratorType):
        frame = gen.gi_frame
        iterator = iter(frame.f_locals['.0'])  # Get underlying iterator
        if isinstance(iterator, SymbolicZipIterator):
            comparison = next(gen)  # Get comparison expression
            return SymbolicGenerator(iterator, comparison)
    return gen

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
        if x.concrete is not None:
            return SymbolicInt(ord(x.concrete), tracer=x.tracer)
        return SymbolicInt(x.tracer.backend.StrToCode(x.z3_expr), tracer=x.tracer)
    if isinstance(x, SymbolicSlice):
        return sym_ord(x.get_slice())

    return ord(x)

def sym_chr(x):
    if isinstance(x, SymbolicInt):
        if x.concrete is not None:
            return SymbolicStr(chr(x.concrete), tracer=x.tracer)
        return SymbolicStr(x.tracer.backend.CodeToStr(x.z3_expr), tracer=x.tracer)
    return chr(x)

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
    if isinstance(x, SymbolicList):
        return x.__len__()
    return len(x)

def sym_str(x):
    if isinstance(x, SymbolicInt):
        return SymbolicStr(str(x.concrete) if x.concrete is not None else x.tracer.backend.IntToStr(x.z3_expr), tracer=x.tracer)
    if isinstance(x, SymbolicFloat):
        return SymbolicStr(str(x.concrete) if x.concrete is not None else x.tracer.backend.RealToStr(x.z3_expr), tracer=x.tracer)
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

def infer_element_type(elements):
    for elem in elements:
        if isinstance(elem, (SymbolicStr, str)):
            return str
        elif isinstance(elem, (SymbolicFloat, float)):
            return float
        elif isinstance(elem, (SymbolicInt, int)):
            return int
        elif isinstance(elem, (SymbolicBool, bool)):
            return bool
    return int # Default

def wrap_list(elements, tracer=None):
    element_type = infer_element_type(elements)
    return SymbolicList(elements, element_type, tracer=tracer)

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

def sym_index(container, start, stop=None, step=None):
    tracer = first_tracer(container, start, stop, step)
    
    if tracer is None:
        if stop is None:
            assert step is None
            return container[start]
        if step is None:
            return container[start:stop]
        return container[start:stop:step]
    
    slice_obj = slice(start, stop, step)
    return container[slice_obj]

class HoleyWrapper(ast.NodeTransformer):
    def __init__(self):
        self.path = []

    def visit(self, node):
        self.path.append(node)
        result = super().visit(node)
        self.path.pop()
        return result

    def visit_Subscript(self, node):
        node = self.generic_visit(node)
        if isinstance(node.slice, ast.Slice):
            lower = node.slice.lower or ast.Constant(value=None)
            upper = node.slice.upper or ast.Constant(value=None)
            step = node.slice.step or ast.Constant(value=None)

            return ast.Call(
                func=ast.Name(id='sym_index', ctx=ast.Load()),
                args=[node.value, lower, upper, step],
                keywords=[]
            )
        else:
            return ast.Call(
                func=ast.Name(id='sym_index', ctx=ast.Load()),
                args=[node.value, node.slice],
                keywords=[]
            )

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
            if node.func.id in ['int', 'float', 'str', 'len', 'range', 'bin', 'ord', 'chr', 'sum', 'zip', 'sorted']:
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
        'List': list,
        'SymbolicStr': SymbolicStr,
        'SymbolicInt': SymbolicInt,
        'wrap_str': lambda s: SymbolicStr(s, tracer=tracer),
        'wrap_int': lambda n: SymbolicInt(n, tracer=tracer),
        'wrap_list': wrap_list,
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
        'sym_chr': sym_chr,
        "sym_sum": sym_sum,
        'sym_zip': sym_zip,
        'sym_generator': sym_generator,
        'sym_sorted': sym_sorted,
        'sym_index': sym_index
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
