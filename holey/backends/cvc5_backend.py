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

class CVC5SolverWrapper:
    def __init__(self, solver):
        self.solver = solver
        
    def add(self, constraint):
        if isinstance(constraint, bool):
            constraint = self.solver.mkBoolean(constraint)
        self.solver.assertFormula(constraint)
        
    def check(self):
        return self.solver.checkSat()
        
    def model(self):
        return self.solver.getValue

    def get_value(self, expr):
        try:
            return self.solver.getValue(expr)
        except:
            raise ValueError(f"Failed to get value for expression: {type(expr)}")

class CVC5Backend(Backend):
    HAS_CVC5 = HAS_CVC5
    MAX_TERMS = 1000  # Limit number of terms to prevent memory issues
    
    def __init__(self):
        if not HAS_CVC5:
            raise ImportError("CVC5 is required for this backend")
        self.solver = None
        self.intSort = None
        self.boolSort = None
        self._term_count = 0
        self.reset()

    def reset(self):
        """Reset the solver state and term counter"""
        try:
            # Clean up old solver if it exists
            if self.solver is not None:
                self.cleanup()
            
            # Create new solver
            self.solver = cvc5.Solver()
            self.solver.setOption("produce-models", "true")
            self.solver.setOption("incremental", "true")
            self.solver.setOption("tlimit", "5000")
            self.solver.setOption("rlimit", "10000000")
            
            # Initialize sorts
            self.intSort = self.solver.getIntegerSort()
            self.boolSort = self.solver.getBooleanSort()
            self._term_count = 0
        except Exception as e:
            print(f"Error during CVC5 reset: {e}")
            self.solver = None
            self.intSort = None
            self.boolSort = None
            raise

    def cleanup(self):
        """Cleanup solver resources"""
        if self.solver is not None:
            # Clear references to solver and its resources
            self.solver = None
            self.intSort = None
            self.boolSort = None
            # Force garbage collection
            import gc
            gc.collect()

    def __del__(self):
        self.cleanup()

    def _check_terms(self):
        self._term_count += 1
        if self._term_count > self.MAX_TERMS:
            raise RuntimeError("Too many CVC5 terms created. The problem might be too complex.")

    @staticmethod
    def is_available() -> bool:
        """Check if CVC5 is available"""
        return HAS_CVC5
    
    def Int(self, name: str) -> Any:
        return self.solver.mkConst(self.intSort, name)
    
    def IntVal(self, val: int) -> Any:
        self._check_terms()
        try:
            if abs(val) > 1000000:  # Handle very large numbers differently
                return self.solver.mkInteger(str(val))
            return self.solver.mkInteger(val)
        except:
            if val < 0:
                return self.solver.mkTerm(Kind.NEG, self.solver.mkInteger(str(-val)))
            return self.solver.mkInteger(str(val))
    
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

    def Mul(self, a, b) -> Any:
        self._check_terms()
        return self.solver.mkTerm(Kind.MULT, a, b)
    
    def Add(self, a, b) -> Any:
        self._check_terms()
        return self.solver.mkTerm(Kind.ADD, a, b)
    
    def Sub(self, a, b) -> Any:
        self._check_terms()
        return self.solver.mkTerm(Kind.SUB, a, b)
    
    def Div(self, a, b) -> Any:
        return self.solver.mkTerm(Kind.DIVISION, a, b)
    
    def UDiv(self, a, b) -> Any:
        # Integer division
        return self.solver.mkTerm(Kind.INTS_DIVISION, a, b)

    def LT(self, a, b) -> Any:
        return self.solver.mkTerm(Kind.LT, a, b)
    
    def LE(self, a, b) -> Any:
        return self.solver.mkTerm(Kind.LEQ, a, b)
    
    def GT(self, a, b) -> Any:
        return self.solver.mkTerm(Kind.GT, a, b)
    
    def GE(self, a, b) -> Any:
        return self.solver.mkTerm(Kind.GEQ, a, b)
    
    def Eq(self, a, b) -> Any:
        return self.solver.mkTerm(Kind.EQUAL, a, b)

    def Solver(self) -> Any:
        return CVC5SolverWrapper(self.solver)
    
    def is_sat(self, result) -> bool:
        return result.isSat()
