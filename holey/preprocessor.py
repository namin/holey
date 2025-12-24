from .core import SymbolicTracer, make_symbolic, SymbolicBool, SymbolicFloat, SymbolicInt, SymbolicList, SymbolicRange, SymbolicRangeIterator, SymbolicStr, SymbolicSlice, truthy, BoundedSymbolicList, BoundedSymbolicSlice
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
    from holey.core import SymbolicList, SymbolicInt, SymbolicSlice

    # Handle BoundedSymbolicSlice (from BoundedSymbolicList slicing)
    if isinstance(iterable, BoundedSymbolicSlice):
        return iterable.sum()

    # Handle BoundedSymbolicList directly
    if isinstance(iterable, BoundedSymbolicList):
        if iterable.elementTyp == int:
            return SymbolicInt(iterable.bounded_vars.sum(), tracer=iterable.tracer)
        # For non-int lists, fall back to iteration
        return sum(iterable)

    # Handle SymbolicList directly without iteration
    if isinstance(iterable, SymbolicList):
        if iterable.elementTyp == int:
            return SymbolicInt(iterable.tracer.backend.ListSum(iterable.z3_expr), tracer=iterable.tracer)
        # For non-int lists, fall back to iteration
        return sum(iterable)

    # Handle SymbolicSlice - need to compute sum of slice
    if isinstance(iterable, SymbolicSlice):
        # Use the sum method we just added
        if hasattr(iterable, 'sum'):
            return iterable.sum()
        # Fallback to get_slice
        sliced = iterable.get_slice()
        if isinstance(sliced, SymbolicList):
            return sym_sum(sliced)
        # For other types, try regular sum
        return sum(sliced)
    
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

    # Check if all iterables are bounded (BoundedSymbolicList or BoundedSymbolicSlice)
    # These can be iterated concretely since they have known sizes
    all_bounded = all(
        isinstance(it, (BoundedSymbolicList, BoundedSymbolicSlice))
        for it in iterables
    )
    if all_bounded:
        # Iterate concretely over bounded lists/slices
        return zip(*[list(it) for it in iterables])

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
            return SymbolicStr(ord(x.concrete), tracer=x.tracer)
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
    if isinstance(x, BoundedSymbolicList):
        return x.__len__()
    # Handle set() of bounded list elements - check for distinctness
    if isinstance(x, set):
        elems = list(x)
        if elems and all(isinstance(e, SymbolicInt) for e in elems):
            # This is likely set(bounded_list) - return a SymbolicInt that
            # equals len(elems) only when all elements are distinct
            tracer = elems[0].tracer
            n = len(elems)
            # Build pairwise distinctness constraints
            distinct_constraints = []
            for i in range(n):
                for j in range(i + 1, n):
                    ei = elems[i].z3_expr if elems[i].z3_expr is not None else tracer.backend.IntVal(elems[i].concrete)
                    ej = elems[j].z3_expr if elems[j].z3_expr is not None else tracer.backend.IntVal(elems[j].concrete)
                    distinct_constraints.append(tracer.backend.Not(tracer.backend.Eq(ei, ej)))
            if distinct_constraints:
                # Return a SymbolicInt that is n when all distinct, undefined otherwise
                # We record the constraint - when this is compared to n, the constraint is added
                all_distinct = tracer.backend.And(*distinct_constraints) if len(distinct_constraints) > 1 else distinct_constraints[0]
                # Return a special object that, when compared to n, adds the distinctness constraint
                return SymbolicSetLen(n, all_distinct, tracer)
            return SymbolicInt(n, tracer=tracer)
    return len(x)


class SymbolicSetLen:
    """Represents len(set(x)) for a bounded list - equals n only if all elements are distinct"""
    def __init__(self, n, distinct_constraint, tracer):
        self.n = n
        self.distinct_constraint = distinct_constraint
        self.tracer = tracer

    def __eq__(self, other):
        if isinstance(other, int):
            if other == self.n:
                # len(set(x)) == n means all elements must be distinct
                return SymbolicBool(self.distinct_constraint, tracer=self.tracer)
            else:
                # len(set(x)) == m where m != n is false (we have n elements)
                return SymbolicBool(False, tracer=self.tracer)
        if isinstance(other, SymbolicInt):
            if other.concrete is not None:
                return self.__eq__(other.concrete)
            # Symbolic comparison - n if distinct, else less
            return SymbolicBool(
                self.tracer.backend.And(
                    self.distinct_constraint,
                    self.tracer.backend.Eq(self.tracer.backend.IntVal(self.n), other.z3_expr)
                ),
                tracer=self.tracer
            )
        return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        if isinstance(eq, SymbolicBool):
            if eq.concrete is not None:
                return SymbolicBool(not eq.concrete, tracer=self.tracer)
            return SymbolicBool(self.tracer.backend.Not(eq.z3_expr), tracer=self.tracer)
        return NotImplemented

def sym_str(x):
    if isinstance(x, SymbolicInt):
        return SymbolicStr(str(x.concrete) if x.concrete is not None else x.tracer.backend.IntToStr(x.z3_expr), tracer=x.tracer)
    if isinstance(x, SymbolicFloat):
        return SymbolicStr(str(x.concrete) if x.concrete is not None else x.tracer.backend.RealToStr(x.z3_expr), tracer=x.tracer)
    if isinstance(x, SymbolicBool):
        if x.concrete is not None:
            return SymbolicStr(str(x.concrete), tracer=x.tracer)
        return SymbolicStr(
            x.tracer.backend.If(
                x.z3_expr,
                x.tracer.backend.StringVal("True"),
                x.tracer.backend.StringVal("False")
            ),
            tracer=x.tracer
        )
    if isinstance(x, SymbolicStr):
        return x
    return str(x)

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
    
    # If we have concrete values, return a regular range
    if all(isinstance(arg, int) for arg in [start, stop, step]):
        return range(start, stop, step)
    
    # For symbolic ranges, check if all values are concrete
    start_val = start if isinstance(start, int) else (start.concrete if hasattr(start, 'concrete') else None)
    stop_val = stop if isinstance(stop, int) else (stop.concrete if hasattr(stop, 'concrete') else None) 
    step_val = step if isinstance(step, int) else (step.concrete if hasattr(step, 'concrete') else None)
    
    if start_val is not None and stop_val is not None and step_val is not None:
        return range(start_val, stop_val, step_val)
    
    # For use with all()/any(), return a SymbolicRange that can be properly quantified
    from holey.core import SymbolicRange
    return SymbolicRange(start, stop, step if step != 1 else None, tracer=tracer)

def sym_not(x):
    if hasattr(x, 'tracer'):
        return x.__not__()
    return not x

def sym_in(x, container):
    # Handle native Python set/dict/list which don't have .contains()
    if isinstance(container, (set, frozenset, dict, list)):
        # For symbolic x in concrete container, check membership
        if hasattr(x, 'tracer'):
            tracer = x.tracer
            # Build OR of all equality checks
            if isinstance(container, dict):
                container = container.keys()
            checks = [x == tracer.ensure_symbolic(item) for item in container]
            if not checks:
                return SymbolicBool(False, tracer=tracer)
            result = checks[0]
            for check in checks[1:]:
                result = result.__or__(check)
            return result
        else:
            return x in container
    # Handle Python range objects
    if isinstance(container, range):
        if hasattr(x, 'tracer'):
            tracer = x.tracer
            # x in range(start, stop, step) means:
            # x >= start and x < stop and (x - start) % step == 0
            start = tracer.ensure_symbolic(container.start)
            stop = tracer.ensure_symbolic(container.stop)
            step = tracer.ensure_symbolic(container.step)
            result = (x >= start).__and__(x < stop)
            if container.step != 1:
                result = result.__and__((x - start) % step == 0)
            return result
        else:
            return x in container
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
    # Handle SymbolicGenerator from sym_generator()
    if isinstance(iterable, SymbolicGenerator):
        iterator = iterable.iterator
        if isinstance(iterator, SymbolicZipIterator):
            predicate = iterable.comparison
            length = iterator.tracer.ensure_symbolic(iterator.length)
            bounds = iterator.tracer.backend.And(
                iterator.tracer.backend.GE(iterator.pos.z3_expr, iterator.tracer.backend.IntVal(0)),
                iterator.tracer.backend.LT(iterator.pos.z3_expr, length.z3_expr)
            )
            return SymbolicBool(iterator.tracer.backend.ForAll(
                [iterator.tracer.backend.Int(iterator.pos.z3_expr.decl().name())],
                iterator.tracer.backend.Implies(
                    bounds,
                    truthy(predicate).z3_expr
                )
            ), tracer=iterator.tracer)

    if isinstance(iterable, types.GeneratorType):
        iterator = iter(iterable.gi_frame.f_locals['.0'])
        if isinstance(iterator, SymbolicRangeIterator):
            predicate = next(iterable)
            bounds = iterator.get_bounds()
            # For all(), we want: forall i. (bounds(i) => predicate(i))
            # This is handled by returning a special marker that add_constraint will recognize
            return SymbolicBool(iterator.tracer.backend.ForAll(
                [iterator.tracer.backend.Int(iterator.var.z3_expr.decl().name())],
                iterator.tracer.backend.Implies(
                    bounds.z3_expr,
                    truthy(predicate).z3_expr
                )
            ), tracer=iterator.tracer)
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

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        # Handle tuple unpacking: a, b = expr -> _tmp = expr; a = _tmp[0]; b = _tmp[1]
        # Also adds length constraint: len(expr) == n
        # Using a temp variable ensures the RHS is evaluated before any LHS vars are updated
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            targets = node.targets[0].elts
            value = node.value
            n = len(targets)

            # Create individual assignments
            assignments = []

            # Generate a unique temp variable name
            temp_name = f'_unpack_tmp_{next(counter)}'
            temp_var = ast.Name(id=temp_name, ctx=ast.Load())

            # First, assign RHS to temp: _tmp = expr
            temp_assign = ast.Assign(
                targets=[ast.Name(id=temp_name, ctx=ast.Store())],
                value=value
            )
            ast.copy_location(temp_assign, node)
            ast.fix_missing_locations(temp_assign)
            assignments.append(temp_assign)

            # Add a length constraint: _assert(sym_len(_tmp) == n)
            length_check = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='_assert', ctx=ast.Load()),
                    args=[
                        ast.Compare(
                            left=ast.Call(
                                func=ast.Name(id='sym_len', ctx=ast.Load()),
                                args=[temp_var],
                                keywords=[]
                            ),
                            ops=[ast.Eq()],
                            comparators=[ast.Constant(value=n)]
                        )
                    ],
                    keywords=[]
                )
            )
            ast.copy_location(length_check, node)
            ast.fix_missing_locations(length_check)
            assignments.append(length_check)

            # Now extract from temp: a = _tmp[0]; b = _tmp[1]; etc
            for i, target in enumerate(targets):
                assign = ast.Assign(
                    targets=[target],
                    value=ast.Subscript(
                        value=temp_var,
                        slice=ast.Constant(value=i),
                        ctx=ast.Load()
                    )
                )
                # Copy location info from original node
                ast.copy_location(assign, node)
                ast.fix_missing_locations(assign)
                assignments.append(assign)
            return assignments
        return node

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
        # Transform: x not in y
        # Into: sym_not(sym_in(x, y))
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.NotIn):
            return ast.Call(
                func=ast.Name(id='sym_not', ctx=ast.Load()),
                args=[ast.Call(
                    func=ast.Name(id='sym_in', ctx=ast.Load()),
                    args=[node.left, node.comparators[0]],
                    keywords=[]
                )],
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
        # Don't wrap lists that are assignment targets (Store context)
        if isinstance(node.ctx, ast.Store):
            return node
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
    
    def visit_SetComp(self, node):
        """Transform set comprehension to set(list comprehension)
        {expr for x in iter} -> set([expr for x in iter])
        This allows symbolic execution to track the values.
        """
        # Convert SetComp to ListComp
        list_comp = ast.ListComp(
            elt=node.elt,
            generators=node.generators
        )
        # Visit the list comprehension to apply transformations
        list_comp = self.generic_visit(list_comp)
        # Wrap in set() call
        return ast.Call(
            func=ast.Name(id='set', ctx=ast.Load()),
            args=[list_comp],
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
        'wrap_list': lambda xs: wrap_list(xs, tracer=tracer),
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
        'sym_sorted': sym_sorted
    }

def driver(sat_func, typ, cmds=None, llm_solver=None, list_size=None):
    """Run symbolic execution on a sat function.

    Args:
        sat_func: The Python sat function source code
        typ: The type of the symbolic variable
        cmds: SMT solver commands to use
        llm_solver: Optional LLM solver for guidance
        list_size: For list types, optional size bound. If provided, uses BoundedSymbolicList
                   which is much faster for SMT solving. When None with list type, uses
                   the slower but more general recursive list encoding.
    """
    reset()
    backend = default_backend(cmds)
    tracer = SymbolicTracer(backend=backend, llm_solver=llm_solver)
    namespace = create_namespace(tracer)
    sym_var = make_symbolic(typ, 'x', tracer, size=list_size)
    namespace['x'] = sym_var
    exec(inject(sat_func), namespace)
    sat = namespace['sat']
    tracer.driver(lambda: sat(sym_var))
    return sym_var
