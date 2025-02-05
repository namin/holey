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
        constraint = constraint.z3_expr if isinstance(constraint, SymbolicBool) else constraint
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
        return  model[var.z3_expr]
    
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

class SymbolicBool:
    """Wrapper class for symbolic boolean expressions"""
    def __init__(self, value, tracer: Optional[SymbolicTracer] = None):
        """Initialize boolean expression with optional tracer"""
        self.tracer = tracer or SymbolicTracer()
        self.z3_expr = value

    def __bool__(self):
        self.tracer.add_constraint(self.z3_expr)
        return True

    def __and__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.And(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __or__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicBool(self.tracer.backend.Or(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __not__(self):
        return SymbolicBool(self.tracer.backend.Not(self.z3_expr), tracer=self.tracer)
    
    def _ensure_symbolic(self, other):
        if isinstance(other, bool):
            return SymbolicBool(self.tracer.backend.BoolVal(other), tracer=self.tracer)
        return other

class SymbolicInt:
    """Wrapper class for symbolic integer expressions"""
    def __init__(self, value: Optional[Any] = None, name: Optional[str] = None, tracer: Optional[SymbolicTracer] = None):
        self.tracer = tracer or SymbolicTracer()
        if name is not None:
            self.z3_expr = self.tracer.backend.Int(name)
        elif isinstance(value, int):
            self.z3_expr = self.tracer.backend.IntVal(value)
        else:
            self.z3_expr = value
        self.name = name

    def __str__(self):
        return f"SymbolicInt({self.name})"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = SymbolicInt(self.tracer.backend.IntVal(other), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Add(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = SymbolicInt(self.tracer.backend.IntVal(other), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Sub(self.z3_expr, other.z3_expr), tracer=self.tracer)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = SymbolicInt(self.tracer.backend.IntVal(other), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Mul(self.z3_expr, other.z3_expr), tracer=self.tracer)
        
    def __eq__(self, other):
        if isinstance(other, (int, float)):
            other = SymbolicInt(self.tracer.backend.IntVal(other), tracer=self.tracer)
        return self.tracer.backend.Eq(self.z3_expr, other.z3_expr)
    
    def __lt__(self, other):
        if isinstance(other, (int, float)):
            other = SymbolicInt(self.tracer.backend.IntVal(other), tracer=self.tracer)
        return self.tracer.backend.LT(self.z3_expr, other.z3_expr)
    
    def __gt__(self, other):
        if isinstance(other, (int, float)):
            other = SymbolicInt(self.tracer.backend.IntVal(other), tracer=self.tracer)
        return self.tracer.backend.GT(self.z3_expr, other.z3_expr)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicInt(self.z3_expr / other.z3_expr, tracer=self.tracer)
    
    def __rtruediv__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicInt(other.z3_expr / self.z3_expr, tracer=self.tracer)
    
    def __floordiv__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicInt(self.z3_expr / other.z3_expr, tracer=self.tracer)
    
    def __rfloordiv__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicInt(other.z3_expr / self.z3_expr, tracer=self.tracer)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = SymbolicInt(self.tracer.backend.IntVal(other), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Sub(other.z3_expr, self.z3_expr), tracer=self.tracer)
    
    def __hash__(self):
        return hash(str(self.z3_expr))

    def _ensure_symbolic(self, other):
        if isinstance(other, int):
            return SymbolicInt(value=int(other), tracer=self.tracer)
        return other

    def __abs__(self):
        return SymbolicInt(self.tracer.backend.If(self.z3_expr >= 0, 
                                                  self.z3_expr, 
                                                  -self.z3_expr), 
                           tracer=self.tracer)

    def __mod__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.Mod(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rmod__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.Mod(other.z3_expr, self.z3_expr), tracer=self.tracer)

    def __pow__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.Pow(self.z3_expr, other.z3_expr), tracer=self.tracer)

    def __rpow__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicInt(self.tracer.backend.Pow(other.z3_expr, self.z3_expr), tracer=self.tracer)

    def __le__(self, other):
        if isinstance(other, (int, float)):
            other = SymbolicInt(self.tracer.backend.IntVal(other), tracer=self.tracer)
        return self.tracer.backend.LE(self.z3_expr, other.z3_expr)
    
    def __ge__(self, other):
        if isinstance(other, (int, float)):
            other = SymbolicInt(self.tracer.backend.IntVal(other), tracer=self.tracer)
        return self.tracer.backend.GE(self.z3_expr, other.z3_expr)

    def __ne__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicBool(self.z3_expr != other.z3_expr, tracer=self.tracer)

    def __divmod__(self, other):
        other = self._ensure_symbolic(other)
        q = SymbolicInt(self.z3_expr / other.z3_expr, tracer=self.tracer)
        r = SymbolicInt(self.z3_expr % other.z3_expr, tracer=self.tracer)
        return (q, r)

    def __rdivmod__(self, other):
        other = self._ensure_symbolic(other)
        return divmod(other, self)

    def __lshift__(self, other):
        if isinstance(other, (int, SymbolicInt)):
            other = self._ensure_symbolic(other)
            return SymbolicInt(self.z3_expr * (2 ** other.z3_expr), tracer=self.tracer)
        return NotImplemented

    def __rshift__(self, other):
        """Support right shift (>>)"""
        if isinstance(other, (int, SymbolicInt)):
            other = self._ensure_symbolic(other)
            return SymbolicInt(self.z3_expr / (2 ** other.z3_expr), tracer=self.tracer)
        return NotImplemented

    def __neg__(self):
        # Use backend's Sub method with 0 for negation
        zero = SymbolicInt(self.tracer.backend.IntVal(0), tracer=self.tracer)
        return SymbolicInt(self.tracer.backend.Sub(zero.z3_expr, self.z3_expr), tracer=self.tracer)

class SymbolicStr:
    def __init__(self, concrete_str: str, tracer: Optional[SymbolicTracer] = None):
        self.concrete = concrete_str
        self.tracer = tracer
        self._count_cache = {}

    def __getitem__(self, key):
        if isinstance(key, slice):
            if (not isinstance(key.start, SymbolicInt) and 
                not isinstance(key.stop, SymbolicInt) and 
                not isinstance(key.step, SymbolicInt)):
                return SymbolicStr(self.concrete[key], tracer=self.tracer)
            return SymbolicSlice(self.concrete, key.start, key.stop, key.step, self.tracer)
        elif isinstance(key, SymbolicInt):
            return SymbolicSlice(self.concrete, key, key+1, key.step, self.tracer)
        return self.concrete[key]
        
    def __len__(self):
        return len(self.concrete)
        
    def count(self, char: str) -> SymbolicInt:
        if char not in self._count_cache:
            if isinstance(self, SymbolicStr):
                # Create new symbolic variable for the count
                result = self.tracer.backend.Int(f'str_count_{char}_{id(self)}')

                # The real count in the concrete string
                concrete_count = self.concrete.count(char)

                # Add constraint that our symbolic count equals the concrete count
                self.tracer.add_constraint(result == concrete_count)

                result = SymbolicInt(result, tracer=self.tracer)
            else:
                result = self.concrete.count(char)
            self._count_cache[char] = result
        return self._count_cache[char]

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

    # Optional but useful string methods
    def startswith(self, prefix):
        result = self.concrete.startswith(prefix)
        if self.tracer:
            return SymbolicBool(self.tracer.backend.BoolVal(result), tracer=self.tracer)
        return result

    def endswith(self, suffix):
        result = self.concrete.endswith(suffix)
        if self.tracer:
            return SymbolicBool(self.tracer.backend.BoolVal(result), tracer=self.tracer)
        return result

class SymbolicSlice:
    def __init__(self, concrete_str: str, start, end, step=None, tracer: Optional[SymbolicTracer] = None):
        self.concrete = concrete_str
        self.start = start
        self.end = end
        self.step = step
        self.tracer = tracer

    def count(self, substr: str) -> SymbolicInt:
        """For a slice s[start:end], return count of substr in that slice"""
        if isinstance(self.start, SymbolicInt):
            result = self.tracer.backend.Int(f'count_{substr}_{id(self)}')
            str_len = len(self.concrete)
            
            # Check if end is start + constant
            if (isinstance(self.end, SymbolicInt) and 
                hasattr(self.end, 'z3_expr') and 
                str(self.end.z3_expr).startswith(str(self.start.z3_expr) + " + ")):
                # Extract the constant
                constant = int(str(self.end.z3_expr).split(" + ")[1])
                
                # Add bounds constraints
                self.tracer.add_constraint(self.start >= 0)
                self.tracer.add_constraint(self.start + constant <= str_len)
                
                # Only iterate through valid start positions
                for start in range(str_len - constant + 1):
                    count_here = self.concrete[start:start + constant].count(substr)
                    self.tracer.add_constraint(
                        self.tracer.backend.Implies(
                            self.tracer.backend.And(
                                self.start == start,
                                self.end == start + constant
                            ),
                            result == count_here
                        )
                    )
            else:
                # Fall back to general case for arbitrary start/end
                for start in range(str_len):
                    for end in range(start, str_len + 1):
                        count_here = self.concrete[start:end].count(substr)
                        self.tracer.add_constraint(
                            self.tracer.backend.Implies(
                                self.tracer.backend.And(
                                    self.start == start,
                                    self.end == end
                                ),
                                result == count_here
                            )
                        )
            return SymbolicInt(result, tracer=self.tracer)
        return self.concrete[self.start:self.end].count(substr)

def make_symbolic(typ: Type, name: str, tracer: Optional[SymbolicTracer] = None) -> Any:
    """Create a new symbolic variable of given type"""
    if typ == int:
        sym = SymbolicInt(name=name, tracer=tracer)
    elif typ == bool:
        sym = SymbolicBool(name=name, tracer=tracer)
    else:
        raise ValueError(f"Unsupported symbolic type: {typ}")
    return sym

