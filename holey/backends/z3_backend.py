"""
Z3 backend implementation.
"""
from typing import Any
from .base import Backend

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

class Z3Backend(Backend):
    HAS_Z3 = HAS_Z3
    
    def __init__(self):
        if not HAS_Z3:
            raise ImportError("Z3 is required for this backend")
        super().__init__()
    
    @staticmethod
    def is_available() -> bool:
        """Check if Z3 is available"""
        return HAS_Z3
    
    def Int(self, name: str) -> Any:
        return z3.Int(name)
    
    def IntVal(self, val: int) -> Any:
        return z3.IntVal(val)
    
    def Bool(self, name: str) -> Any:
        return z3.Bool(name)
        
    def BoolVal(self, val: bool) -> Any:
        return z3.BoolVal(val)
    
    def And(self, *args) -> Any:
        return z3.And(*args)
    
    def Or(self, *args) -> Any:
        return z3.Or(*args)
    
    def Not(self, arg) -> Any:
        return z3.Not(arg)
    
    def Implies(self, a, b) -> Any:
        return z3.Implies(a, b)
    
    def If(self, cond, t, f) -> Any:
        return z3.If(cond, t, f)
    
    def ForAll(self, vars, body) -> Any:
        return z3.ForAll(vars, body)
    
    def Distinct(self, args) -> Any:
        return z3.Distinct(args)
    
    def Solver(self) -> Any:
        return z3.Solver()
    
    def is_sat(self, result) -> bool:
        return result == z3.sat
