"""
Base backend implementation.
"""
from typing import Any, Optional

class Backend:
    def __init__(self):
        self.base_backend = self

    def get_base_backend(self):
        """Get the underlying base backend for low-level operations"""
        return self.base_backend

    def Int(self, name: str) -> Any:
        raise NotImplementedError
    
    def Bool(self, name: str) -> Any:
        raise NotImplementedError
    
    def IntVal(self, val: int) -> Any:
        raise NotImplementedError
    
    def BoolVal(self, val: bool) -> Any:
        raise NotImplementedError
    
    def And(self, *args) -> Any:
        raise NotImplementedError
    
    def Or(self, *args) -> Any:
        raise NotImplementedError
    
    def Not(self, arg) -> Any:
        raise NotImplementedError
    
    def Implies(self, a, b) -> Any:
        raise NotImplementedError
    
    def If(self, cond, t, f) -> Any:
        raise NotImplementedError
    
    def ForAll(self, vars, body) -> Any:
        raise NotImplementedError
    
    def Distinct(self, args) -> Any:
        raise NotImplementedError

    def Mod(self, a, b) -> Any:
        raise NotImplementedError

    def Pow(self, a, b) -> Any:
        raise NotImplementedError
    
    def Solver(self) -> Any:
        raise NotImplementedError
    
    def is_sat(self, result) -> bool:
        raise NotImplementedError

    def Real(self, val: float) -> Any:
        """Create a real number constant"""
        raise NotImplementedError
