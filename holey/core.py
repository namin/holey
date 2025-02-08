from typing import Any, Optional, List, Type
from dataclasses import dataclass
from contextlib import contextmanager
import z3
from .backends import default_backend, Backend

@dataclass 
class SymbolicTracer:
    def __init__(self, backend=None):
        self.backend = backend or default_backend()
        self.path_conditions = []
        self.branch_counter = 0
        
    def branch(self, condition):
        """Handle a branching point in execution"""
        self.backend.push()
        self.backend.add(condition.z3_expr)  # Try true path
        
        # If true path is possible, take it and record in trace
        if self.backend.check() == "sat":
            self.path_conditions.append(condition.z3_expr)
            self.backend.pop()
            return True
            
        # If true path impossible, must take false path
        self.backend.pop()
        self.backend.push()
        not_cond = self.backend.Not(condition.z3_expr)
        self.backend.add(not_cond)
        if self.backend.check() == "sat":
            self.path_conditions.append(not_cond)
            self.backend.pop()
            return False
            
        self.backend.pop()
        raise ValueError("No feasible branches found")

    def __enter__(self):
        self._stack.append((self.path_conditions.copy(), self.backend.solver.assertions()))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stack:
            old_conditions, old_assertions = self._stack.pop()
            self.path_conditions = old_conditions
            self.backend.solver = self.backend.Solver()
            self.backend.solver.add(old_assertions)
    
    def add_constraint(self, constraint):
        if isinstance(constraint, SymbolicBool):
            constraint = constraint.z3_expr
        elif isinstance(constraint, SymbolicInt):
            constraint = (constraint != 0).z3_expr
        self.path_conditions.append(constraint)
        self.backend.solver.add(constraint)
    
    def check(self):
        """Check satisfiability including path conditions"""
        self.backend.push()
        for cond in self.path_conditions:
            self.backend.add(cond)
        result = self.backend.check()
        self.backend.pop()
        return result
    
    def model(self):
        return self.backend.solver.model()

    def solution(self):
        result = self.check()
        if self.backend.is_sat(result):
            return self.model()
        return None

    def solution_var(self, model, var):
        return  model[var.name]
    
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
        if isinstance(value, bool):
            self.concrete = value
        else:
            self.concrete = None

    def __bool__(self):
        if isinstance(self.z3_expr, bool):
            return self.z3_expr
        return self.tracer.branch(self)

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
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete == other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.Eq(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __lt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete < other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.LT(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __gt__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete > other.concrete, tracer=self.tracer)
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
        if self.concrete is not None:
            return abs(self.concrete)
        return SymbolicInt(self.tracer.backend.If(self.z3_expr >= 0, 
                                                  self.z3_expr, 
                                                  -self.z3_expr), 
                           tracer=self.tracer)

    def __mod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicInt(self.concrete % other.concrete, tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Mod(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rmod__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicInt(other.concrete % self.concrete, tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Mod(other.z3_expr, self.z3_expr), tracer=self.tracer)


    def __pow__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rpow__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if isinstance(other, SymbolicFloat):
            return SymbolicFloat(self.tracer.backend.Pow(other.z3_expr, self.z3_expr), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Pow(other.z3_expr, self.z3_expr), tracer=self.tracer)


    def __le__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.LE(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __ge__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.GE(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __ne__(self, other):
        other = self.tracer.ensure_symbolic(other)
        if self.concrete is not None and other.concrete is not None:
            return SymbolicBool(self.concrete != other.concrete, tracer=self.tracer)
        return SymbolicBool(self.tracer.backend.Not(self.tracer.backend.Eq(self.z3_expr, other.z3_expr)), tracer=self.tracer)

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
        if self.concrete is not None:
            return SymbolicInt(-self.concrete, tracer=self.tracer)
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

class SymbolicList:
    def __init__(self, value, name: Optional[str] = None, tracer: Optional[SymbolicTracer] = None):
        self.tracer = tracer        
        assert name is None       
        assert isinstance(value, list)
        self.concrete = value

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Handle slice
            start = key.start or 0
            stop = key.stop or len(self.concrete)
            step = key.step or 1
            return SymbolicList([self.concrete[i] for i in range(start, stop, step)], tracer=self.tracer)
        # Handle integer index
        if isinstance(key, SymbolicInt):
            # For symbolic index, we need to build an If expression
            result = None
            for i, item in enumerate(self.concrete):
                if result is None:
                    result = SymbolicInt(
                        self.tracer.backend.If(
                            key.z3_expr == i,
                            item.z3_expr,
                            self.tracer.backend.IntVal(0)  # default value
                        ),
                        tracer=self.tracer
                    )
                else:
                    result = SymbolicInt(
                        self.tracer.backend.If(
                            key.z3_expr == i,
                            item.z3_expr,
                            result.z3_expr
                        ),
                        tracer=self.tracer
                    )
            return result
        return self.concrete[key]

    def __iter__(self):
        return iter(self.concrete)

    def __len__(self):
        return len(self.concrete)

    def __add__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicList(self.concrete + other.concrete, tracer=self.tracer)

    def __radd__(self, other):
        other = self.tracer.ensure_symbolic(other)
        return SymbolicList(other.concrete + self.concrete, tracer=self.tracer)

    def index(self, item):
        # Return first index where item appears as SymbolicInt
        result = None
        for i, x in enumerate(self.concrete):
            eq = (x == item)  # This gives us a SymbolicBool
            if result is None:
                result = SymbolicInt(
                    self.tracer.backend.If(
                        eq.z3_expr,  # condition
                        self.tracer.backend.IntVal(i),  # then
                        self.tracer.backend.IntVal(-1)  # else
                    ),
                    tracer=self.tracer
                )
            else:
                result = SymbolicInt(
                    self.tracer.backend.If(
                        eq.z3_expr,  # condition
                        self.tracer.backend.If(
                            result.z3_expr == -1,  # if not found yet
                            self.tracer.backend.IntVal(i),  # use this index
                            result.z3_expr  # keep previous index
                        ),
                        result.z3_expr  # else keep previous result
                    ),
                    tracer=self.tracer
                )
        return result

    def count(self, item):
        """Count occurrences of item in list"""
        count = 0
        for x in self.concrete:
            eq = (x == item)  # This gives us a SymbolicBool
            if eq.concrete is not None:  # If we can evaluate it concretely
                count += int(eq.concrete)
            else:
                # For now, just return a symbolic int for the whole count
                # This could be made more precise later
                result = None
                for x in self.concrete:
                    eq = (x == item)
                    if result is None:
                        result = SymbolicInt(
                            self.tracer.backend.If(
                                eq.z3_expr,
                                self.tracer.backend.IntVal(1),
                                self.tracer.backend.IntVal(0)
                            ),
                            tracer=self.tracer
                        )
                    else:
                        result = result + SymbolicInt(
                            self.tracer.backend.If(
                                eq.z3_expr,
                                self.tracer.backend.IntVal(1),
                                self.tracer.backend.IntVal(0)
                            ),
                            tracer=self.tracer
                        )
                return result
        return SymbolicInt(count, tracer=self.tracer)

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

class SymbolicRangeIterator:
    def __init__(self, sym_range):
        self.tracer = sym_range.tracer
        self.sym_range = sym_range
        # Create fresh variable for the iterator
        self.var = SymbolicInt(name=f'i_{SymbolicRange._counter}', tracer=sym_range.tracer)
        SymbolicRange._counter += 1
        self.used = False
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.used:
            raise StopIteration
        self.used = True
        return self.var
    
    def get_bounds(self):
        """Get bounds constraints including step if present"""
        bounds = (self.var >= self.sym_range.start).__and__(
                 self.var < self.sym_range.end)
        
        if self.sym_range.step is not None:
            # i = start + k * step for some k >= 0
            k = SymbolicInt(name=f'k_{SymbolicRange._counter}', tracer=self.sym_range.tracer)
            SymbolicRange._counter += 1
            step_constraint = (self.var == self.sym_range.start + k * self.sym_range.step).__and__(
                             k >= 0).__and__(
                             k < (self.sym_range.end - self.sym_range.start) / self.sym_range.step)
            bounds = bounds.__and__(step_constraint)
            
        return bounds

class SymbolicRange:
    _counter = 0

    def __init__(self, start, end, step=None, tracer=None):
        self.start = start if isinstance(start, SymbolicInt) else SymbolicInt(start, tracer=tracer)
        self.end = end if isinstance(end, SymbolicInt) else SymbolicInt(end, tracer=tracer)
        self.step = step if step is None else (step if isinstance(step, SymbolicInt) else SymbolicInt(step, tracer=tracer))
        self.tracer = tracer or self.start.tracer

    def __iter__(self):
        return SymbolicRangeIterator(self)

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

