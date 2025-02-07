from typing import Any, Optional, List, Type
from dataclasses import dataclass
from contextlib import contextmanager
import z3
from .backends import default_backend, Backend

class SymbolicTracer:
    """Tracer for symbolic execution"""

    def __init__(self, backend: Optional[Backend] = None):
        """Initialize tracer with optional backend"""
        self.backend = backend or default_backend()
        self.solver = self.backend.Solver()
        self.path_conditions = []
        self._stack = []
        
    def __enter__(self):
        self._stack.append((self.path_conditions.copy(), self.solver.assertions()))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stack:
            old_conditions, old_assertions = self._stack.pop()
            self.path_conditions = old_conditions
            self.solver = self.backend.Solver()
            self.solver.add(old_assertions)
    
    def add_constraint(self, constraint):
        if isinstance(constraint, SymbolicBool):
            constraint = constraint.z3_expr
        elif isinstance(constraint, SymbolicInt):
            constraint = (constraint != 0).z3_expr
        self.path_conditions.append(constraint)
        self.solver.add(constraint)
    
    def check(self):
        return self.solver.check()
    
    def model(self):
        return self.solver.model()

    def solution(self):
        result = self.check()
        if self.backend.is_sat(result):
            return self.model()
        return None

    def solution_var(self, model, var):
        return  model[var.name]
    
    @contextmanager
    def branch(self):
        """Context manager for handling branches in symbolic execution"""
        old_conditions = self.path_conditions.copy()
        old_solver = self.backend.Solver()
        old_solver.add(self.solver.assertions())
        try:
            yield
        finally:
            self.path_conditions = old_conditions
            self.solver = old_solver

    def ensure_symbolic(self, other):
        if isinstance(other, bool):
            return SymbolicBool(other, tracer=self)
        if isinstance(other, int):
            return SymbolicInt(other, tracer=self)
        if isinstance(other, float):
            return SymbolicFloat(other, tracer=self)
        if isinstance(other, str):
            return SymbolicStr(other, tracer=self)
        return other

class SymbolicBool:
    """Wrapper class for symbolic boolean expressions"""
    def __init__(self, value, tracer: Optional[SymbolicTracer] = None):
        """Initialize boolean expression with optional tracer"""
        self.tracer = tracer or SymbolicTracer()
        self.z3_expr = value

    def __bool__(self):
        if isinstance(self.z3_expr, bool):
            return self.z3_expr
        self.tracer.add_constraint(self.z3_expr)
        return True

    def __and__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.And(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __or__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.Or(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __not__(self):
        if isinstance(self.z3_expr, bool):
            return SymbolicBool(not self.z3_expr, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.Not(self.z3_expr), tracer=self.tracer)
    
class SymbolicInt:
    """Wrapper class for symbolic integer expressions"""
    def __init__(self, value: Optional[Any] = None, name: Optional[str] = None, tracer: Optional[SymbolicTracer] = None):
        self.tracer = tracer or SymbolicTracer()
        self.concrete = None
        if name is not None:
            self.z3_expr = self.tracer.backend.Int(name)
        elif isinstance(value, int):
            self.concrete = value
            self.z3_expr = self.tracer.backend.IntVal(value)
        else:
            self.z3_expr = value
        self.name = name

    def __int__(self):
        if self.concrete is not None:
            return self.concrete
        raise ValueError("SymbolincInt cannot be concretized.")

    def __str__(self):
        return f"SymbolicInt({self.name})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Add(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Add(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __sub__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Sub(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Sub(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __mul__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Mul(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Mul(self.z3_expr, other.z3_expr), tracer=self.tracer)
        
    def __eq__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.Eq(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __lt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.LT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __gt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.GT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.z3_expr / other.z3_expr, tracer=self.tracer)
    
    def __rtruediv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(other.z3_expr / self.z3_expr, tracer=self.tracer)
    
    def __floordiv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.UDiv(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __rfloordiv__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.UDiv(other.z3_expr, self.z3_expr), tracer=self.tracer)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = SymbolicInt(self.tracer.backend.IntVal(other), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Sub(other.z3_expr, self.z3_expr), tracer=self.tracer)
    
    def __hash__(self):
        return hash(str(self.z3_expr))

    def __abs__(self):
        return SymbolicInt(self.tracer.backend.If(self.z3_expr >= 0, 
                                                  self.z3_expr, 
                                                  -self.z3_expr), 
                           tracer=self.tracer)

    def __mod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.Mod(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rmod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.Mod(other.z3_expr, self.z3_expr), tracer=self.tracer)

    def __pow__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rpow__(self, other):
        return self.__pow__(other)

    def __le__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.LE(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __ge__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.GE(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __ne__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.z3_expr != other.z3_expr, tracer=self.tracer)

    def __divmod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        q = SymbolicInt(self.z3_expr / other.z3_expr, tracer=self.tracer)
        r = SymbolicInt(self.z3_expr % other.z3_expr, tracer=self.tracer)
        return (q, r)

    def __rdivmod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return divmod(other, self)

    def __lshift__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.z3_expr * (2 ** other.z3_expr), tracer=self.tracer)

    def __rshift__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicInt(self.z3_expr / (2 ** other.z3_expr), tracer=self.tracer)

    def __neg__(self):
        return SymbolicInt(-self.z3_expr, tracer=self.tracer)

    def is_integer(self):
        """Always returns True since SymbolicInt is always an integer"""
        return SymbolicBool(True, tracer=self.tracer)

    def __and__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.And(truthy(self).z3_expr, truthy(other).z3_expr), tracer=self.tracer)
    
    def __or__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.Or(truthy(self).z3_expr, truthy(other).z3_expr), tracer=self.tracer)

    def __index__(self):
        if self.concrete is not None:
            return self.concrete
        if hasattr(self.z3_expr, 'as_long'):
            return self.z3_expr.as_long()
        raise ValueError("Cannot convert symbolic integer to index")

class SymbolicFloat:
    def __init__(self, value, tracer=None):
        self.tracer = tracer
        self.z3_expr = value

    def __sub__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicFloat(self.tracer.backend.Sub(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __abs__(self):
        result = SymbolicFloat(
            self.tracer.backend.If(self.z3_expr >= 0,
                                   self.z3_expr,
                                   -self.z3_expr),
            tracer=self.tracer)
        return result

    def __le__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.z3_expr <= other.z3_expr, tracer=self.tracer)

    def is_integer(self):
        """Returns True if the number equals its floor"""
        return SymbolicBool(
            self.tracer.backend.Eq(
                self.z3_expr,
                self.tracer.backend.ToInt(self.z3_expr)
            ),
            tracer=self.tracer
        )

class SymbolicStr:
    def __init__(self, value: str, name: Optional[str] = None, tracer: Optional[SymbolicTracer] = None):
        self.tracer = tracer
        self.concrete = None
        if name is not None:
            self.z3_expr = self.tracer.backend.String(name)
        elif isinstance(value, str):
            self.concrete = value
            self.z3_expr = self.tracer.backend.StringVal(value)
        else:
            self.z3_expr = value

    def split(self):
        if self.concrete is not None:
            return self.concrete.split()
        raise ValueError("Split not implemented for symbolic strings.")

    def __getitem__(self, key):
        if self.concrete is not None:
            if isinstance(key, slice):
                if (not isinstance(key.start, SymbolicInt) and 
                    not isinstance(key.stop, SymbolicInt) and 
                    not isinstance(key.step, SymbolicInt)):
                    return SymbolicStr(self.concrete[key], tracer=self.tracer)
                return SymbolicSlice(self.concrete, key.start, key.stop, key.step, tracer=self.tracer)
            elif isinstance(key, SymbolicInt):
                return SymbolicSlice(self.concrete, key, key+1, None, tracer=self.tracer)
            return SymbolicStr(self.concrete[key], tracer=self.tracer)
        else:
            if isinstance(key, slice):
                start = key.start if key.start is not None else 0
                stop = key.stop if key.stop is not None else self.tracer.backend.StrLen(self.z3_expr)
                return SymbolicStr(
                    self.tracer.backend.StrSubstr(
                        self.z3_expr,
                        start if isinstance(start, SymbolicInt) else SymbolicInt(start, tracer=self.tracer),
                        stop if isinstance(stop, SymbolicInt) else SymbolicInt(stop, tracer=self.tracer)
                    ),
                    tracer=self.tracer
                )
            else:
                # Single index access
                return SymbolicStr(
                    self.tracer.backend.StrSubstr(
                        self.z3_expr,
                        key if isinstance(key, SymbolicInt) else SymbolicInt(key, tracer=self.tracer),
                        key + 1 if isinstance(key, SymbolicInt) else SymbolicInt(key + 1, tracer=self.tracer)
                    ),
                    tracer=self.tracer
                )

    def __len__(self):
        if self.concrete is not None:
            return SymbolicInt(len(self.concrete), tracer=self.tracer)
        else:
            return SymbolicInt(self.tracer.backend.StrLen(self.z3_expr), tracer=self.tracer)

    def count(self, sub: str) -> 'SymbolicInt':
        """Count occurrences of substring in string"""
        if not isinstance(sub, (str, SymbolicStr)):
            raise TypeError(f"Can't count occurrences of {type(sub)}")
            
        if isinstance(sub, str):
            sub = SymbolicStr(sub, tracer=self.tracer)
            
        # Create a symbolic integer for the count
        result = SymbolicInt(self.tracer.backend.StrCount(self.z3_expr, sub.z3_expr), tracer=self.tracer)
        
        # Add constraint that count is non-negative
        self.tracer.add_constraint(result.z3_expr >= 0)
        
        return result

    def __str__(self):
        return self.concrete

    def __repr__(self):
        return f"SymbolicStr({self.concrete!r}, tracer={self.tracer!r})"

    def __add__(self, other):
        if isinstance(other, SymbolicStr):
            return SymbolicStr(self.concrete + other.concrete, tracer=self.tracer)
        elif isinstance(other, str):
            return SymbolicStr(self.concrete + other, tracer=self.tracer)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            return SymbolicStr(other + self.concrete, tracer=self.tracer)
        return NotImplemented

    # For comparison operations
    def __eq__(self, other):
        if isinstance(other, (str, SymbolicStr)):
            other_str = other.concrete if isinstance(other, SymbolicStr) else other
            result = self.concrete == other_str
            if self.tracer:
                return SymbolicBool(self.tracer.backend.BoolVal(result), tracer=self.tracer)
            return result
        return NotImplemented

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        if isinstance(eq, SymbolicBool):
            return SymbolicBool(self.tracer.backend.Not(eq.z3_expr), tracer=self.tracer)
        return not eq

    def startswith(self, prefix):
        prefix = self.tracer.ensure_symbolic(prefix)
        return SymbolicBool(self.tracer.backend.StrPrefixOf(prefix.z3_expr, self.z3_expr), tracer=self.tracer)

    def isupper(self):
        if self.concrete is not None:
            return SymbolicBool(self.concrete.isupper(), tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.IsUpper(self.z3_expr), tracer=self.tracer)

class SymbolicSlice:
    def __init__(self, concrete_str: str, start, end, step=None, tracer: Optional[SymbolicTracer] = None):
        self.concrete = concrete_str
        self.start = start
        self.end = end
        self.step = step
        self.tracer = tracer

    def count(self, sub: str) -> 'SymbolicInt':
        """Count occurrences of substring in sliced string"""
        if not isinstance(sub, (str, SymbolicStr)):
            raise TypeError(f"Can't count occurrences of {type(sub)}")
            
        if isinstance(sub, str):
            sub = SymbolicStr(sub, tracer=self.tracer)
            
        # Get the sliced string as a SymbolicStr
        sliced = self.get_slice()
        
        # Use the SymbolicStr count method
        return sliced.count(sub)

    def get_slice(self) -> SymbolicStr:
        """Convert slice to SymbolicStr with appropriate constraints"""
        # If all indices are concrete, just return the concrete slice
        if (isinstance(self.start, (int, type(None))) and 
            isinstance(self.end, (int, type(None))) and 
            isinstance(self.step, (int, type(None)))):
            start = self.start if self.start is not None else 0
            end = self.end if self.end is not None else len(self.concrete)
            step = self.step if self.step is not None else 1
            return SymbolicStr(self.concrete[start:end:step], tracer=self.tracer)
        
        start = (self.start.z3_expr if isinstance(self.start, SymbolicInt) 
                else self.tracer.backend.IntVal(self.start if self.start is not None else 0))
        end = (self.end.z3_expr if isinstance(self.end, SymbolicInt)
               else self.tracer.backend.IntVal(self.end if self.end is not None else len(self.concrete)))
        
        result = SymbolicStr(
            self.tracer.backend.StrSubstr(
                self.tracer.backend.StringVal(self.concrete),
                start,
                end
            ), 
            tracer=self.tracer
        )
        
        # Add constraints for valid indices
        str_len = len(self.concrete)
        if isinstance(self.start, SymbolicInt):
            self.tracer.add_constraint(self.start.z3_expr >= 0)
            self.tracer.add_constraint(self.start.z3_expr <= str_len)
        if isinstance(self.end, SymbolicInt):
            self.tracer.add_constraint(self.end.z3_expr >= 0)
            self.tracer.add_constraint(self.end.z3_expr <= str_len)
            
        return result

class SymbolicRange:
    _counter = 0  # Class variable to generate unique variable names
    
    def __init__(self, start, end, step=None, tracer=None):
        self.start = start if isinstance(start, SymbolicInt) else SymbolicInt(start, tracer=tracer)
        self.end = end if isinstance(end, SymbolicInt) else SymbolicInt(end, tracer=tracer)
        self.step = step if step is None else (step if isinstance(step, SymbolicInt) else SymbolicInt(step, tracer=tracer))
        self.tracer = tracer or self.start.tracer

    def __iter__(self):
        # Generate a unique variable name for this range iteration
        SymbolicRange._counter += 1
        var_name = f'i_{SymbolicRange._counter}'
        
        # Create a symbolic integer for the loop variable
        i = SymbolicInt(name=var_name, tracer=self.tracer)
        
        # Add range constraints
        self.tracer.add_constraint((i >= self.start).z3_expr)
        self.tracer.add_constraint((i < self.end).z3_expr)
        
        if self.step is not None:
            # i = start + k * step for some k
            k = SymbolicInt(name=f'k_{SymbolicRange._counter}', tracer=self.tracer)
            self.tracer.add_constraint((i == self.start + k * self.step).z3_expr)
            
            # Add constraint that k is within valid range
            self.tracer.add_constraint((k >= 0).z3_expr)
            self.tracer.add_constraint((k < (self.end - self.start) / self.step).z3_expr)
            
        yield i

    def __contains__(self, item):
        item = self.tracer.ensure_symbolic(item)
            
        result = (item >= self.start) and (item < self.end)
        if self.step is not None:
            # item = start + k * step for some k
            k = SymbolicInt('k', tracer=self.tracer)
            result = result and (item == self.start + k * self.step)
        return result

    def __len__(self):
        if self.step is None or self.step == 1:
            return self.end - self.start
        return (self.end - self.start) / self.step

def fresh_symbolic(var):
    typ = type(var).__name__.lower().replace('symbolic', '')
    return make_symbolic(typ, var.name, var.tracer)

def make_symbolic(typ: Type, name: str, tracer: Optional[SymbolicTracer] = None) -> Any:
    """Create a new symbolic variable of given type"""
    if typ == int or typ == 'int':
        sym = SymbolicInt(name=name, tracer=tracer)
    elif typ == bool or typ == 'bool':
        sym = SymbolicBool(name=name, tracer=tracer)
    elif typ == float or typ == 'float':
        sym = SymbolicFloat(name=name, tracer=tracer)
    elif typ == str or typ == 'str':
        sym = SymbolicStr(name=name, tracer=tracer)
    else:
        raise ValueError(f"Unsupported symbolic type: {typ}")
    return sym

def truthy(x):
    if isinstance(x, SymbolicBool):
        return x
    elif isinstance(x, SymbolicInt):
        return x != 0
    else:
        return x

