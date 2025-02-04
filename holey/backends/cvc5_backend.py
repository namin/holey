"""
CVC5 backend implementation.
"""
from typing import Any
from .base import Backend

try:
    import cvc5
    from cvc5 import Kind
    HAS_CVC5 = True
except ImportError:
    HAS_CVC5 = False

class CVC5Backend(Backend):
    HAS_CVC5 = HAS_CVC5
    
    def __init__(self):
        if not HAS_CVC5:
            raise ImportError("CVC5 is required for this backend")
        super().__init__()
        self.solver = cvc5.Solver()
        self.intSort = self.solver.getIntegerSort()
        self.boolSort = self.solver.getBooleanSort()

    @staticmethod
    def is_available() -> bool:
        """Check if CVC5 is available"""
        return HAS_CVC5
    
    def Int(self, name: str) -> Any:
        return self.solver.mkConst(self.intSort, name)
    
    def IntVal(self, val: int) -> Any:
        return self.solver.mkInteger(val)
    
    def Bool(self, name: str) -> Any:
        return self.solver.mkConst(self.boolSort, name)
        
    def BoolVal(self, val: bool) -> Any:
        return self.solver.mkBoolean(val)
    
    def And(self, *args) -> Any:
        if len(args) == 0:
            return self.solver.mkTrue()
        result = args[0]
        for arg in args[1:]:
            result = self.solver.mkTerm(Kind.AND, result, arg)
        return result
    
    def Or(self, *args) -> Any:
        if len(args) == 0:
            return self.solver.mkFalse()
        result = args[0]
        for arg in args[1:]:
            result = self.solver.mkTerm(Kind.OR, result, arg)
        return result
    
    def Not(self, arg) -> Any:
        return self.solver.mkTerm(Kind.NOT, arg)
    
    def Implies(self, a, b) -> Any:
        return self.solver.mkTerm(Kind.IMPLIES, a, b)
    
    def If(self, cond, t, f) -> Any:
        return self.solver.mkTerm(Kind.ITE, cond, t, f)
    
    def ForAll(self, vars, body) -> Any:
        return self.solver.mkTerm(Kind.FORALL, vars, body)
    
    def Distinct(self, args) -> Any:
        return self.solver.mkTerm(Kind.DISTINCT, *args)

    def Mod(self, a, b) -> Any:
        return self.solver.mkTerm(Kind.MODULO, a, b)

    def Pow(self, a, b) -> Any:
        return self.solver.mkTerm(Kind.POW, a, b)

    def Solver(self) -> Any:
        return self.solver
    
    def is_sat(self, result) -> bool:
        return result.isSat()
