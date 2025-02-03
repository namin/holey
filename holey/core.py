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
        self.path_conditions.append(constraint)
        self.solver.add(constraint)
    
    def check(self):
        return self.solver.check()
    
    def model(self):
        return self.solver.model()

    def solution(self):
        if self.backend.is_sat(self.check()):
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
        other = self._ensure_symbolic(other)
        return SymbolicInt(self.z3_expr + other.z3_expr, tracer=self.tracer)
    
    def __sub__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicInt(self.z3_expr - other.z3_expr, tracer=self.tracer)
    
    def __mul__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicInt(self.z3_expr * other.z3_expr, tracer=self.tracer)
        
    def __eq__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicBool(self.z3_expr == other.z3_expr, tracer=self.tracer)
    
    def __lt__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicBool(self.z3_expr < other.z3_expr, tracer=self.tracer)
    
    def __gt__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicBool(self.z3_expr > other.z3_expr, tracer=self.tracer)
    
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
        other = self._ensure_symbolic(other)
        return SymbolicInt(other.z3_expr - self.z3_expr, tracer=self.tracer)
    
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
        other = self._ensure_symbolic(other)
        return SymbolicBool(self.z3_expr <= other.z3_expr, tracer=self.tracer)
    
    def __ge__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicBool(self.z3_expr >= other.z3_expr, tracer=self.tracer)

    def __eq__(self, other):
        other = self._ensure_symbolic(other)
        return SymbolicBool(self.z3_expr == other.z3_expr, tracer=self.tracer)
    
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
        return SymbolicInt(-self.z3_expr, tracer=self.tracer)

def make_symbolic(typ: Type, name: str, tracer: Optional[SymbolicTracer] = None) -> Any:
    """Create a new symbolic variable of given type"""
    if typ == int:
        sym = SymbolicInt(name=name, tracer=tracer)
        if tracer:
            tracer.add_constraint(sym.z3_expr >= -1000)
            tracer.add_constraint(sym.z3_expr < 1000)
    elif typ == bool:
        sym = SymbolicBool(name=name, tracer=tracer)
    else:
        raise ValueError(f"Unsupported symbolic type: {typ}")
    return sym

