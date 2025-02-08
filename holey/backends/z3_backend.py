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
        self.solver = z3.Solver()
        self.reset()
    
    def reset(self):
        """Reset the solver state"""
        try:
            # Clean up old solver if it exists
            if self.solver is not None:
                self.cleanup()
            
            # Create new solver
            self.solver = z3.Solver()
            # Set timeout in milliseconds (same as CVC5)
            self.solver.set(timeout=5000)
        except Exception as e:
            print(f"Error during Z3 reset: {e}")
            self.solver = None
            raise

    def cleanup(self):
        """Cleanup solver resources"""
        if self.solver is not None:
            self.solver = None
            # Force garbage collection
            import gc
            gc.collect()

    def __del__(self):
        self.cleanup()

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

    def Mod(self, a, b) -> Any:
        return a % b

    def Pow(self, a, b) -> Any:
        return a ** b

    def Solver(self) -> Any:
        return self.solver
    
    def push(self):
        """Push a new scope for backtracking"""
        self.solver.push()
        
    def pop(self):
        """Pop the most recent scope"""
        self.solver.pop()
        
    def add(self, constraint):
        """Add constraint to current scope"""
        self.solver.add(constraint)

    def check(self):
        """Check satisfiability, returning string result for consistency with mock backend"""
        return str(self.solver.check())

    def is_sat(self, result: str) -> bool:
        """Check if string result indicates satisfiability"""
        return result == 'sat'

    def Mul(self, a, b) -> Any:
        return a * b
    
    def Add(self, a, b) -> Any:
        return a + b
    
    def Sub(self, a, b) -> Any:
        return a - b
    
    def Div(self, a, b) -> Any:
        return a / b
    
    def UDiv(self, a, b) -> Any:
        # Integer division
        return a / b
    
    def LT(self, a, b) -> Any:
        return a < b
    
    def LE(self, a, b) -> Any:
        return a <= b
    
    def GT(self, a, b) -> Any:
        return a > b
    
    def GE(self, a, b) -> Any:
        return a >= b
    
    def Eq(self, a, b) -> Any:
        return a == b

    def Real(self, val: float) -> Any:
        """Create a real number constant"""
        return z3.RealVal(val)  # Pass float directly to Z3
