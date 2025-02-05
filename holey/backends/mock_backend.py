"""
Mock backend for collecting and displaying constraints.
"""
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
from .base import Backend

@dataclass
class MockExpr:
    op: str
    args: List[Any]
    _name: Optional[str] = None
    
    def __str__(self):
        if self._name is not None:
            return self._name
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"{self.op}({args_str})"
    
    def __gt__(self, other):
        return MockExpr('>', [self, other])
    
    def __ge__(self, other):
        return MockExpr('>=', [self, other])
        
    def __lt__(self, other):
        return MockExpr('<', [self, other])
        
    def __le__(self, other):
        return MockExpr('<=', [self, other])
        
    def __eq__(self, other):
        return MockExpr('==', [self, other])
    
    def __add__(self, other):
        return MockExpr('+', [self, other])
    
    def __sub__(self, other):
        return MockExpr('-', [self, other])
        
    def __mul__(self, other):
        return MockExpr('*', [self, other])
        
    def __neg__(self):
        return MockExpr('neg', [self])

    def __truediv__(self, other):
        return MockExpr('/', [self, other])

    def decl(self):
        return self
        
    def name(self) -> str:
        return self._name if self._name else str(self)

@dataclass
class MockSolver:
    constraints: List[MockExpr] = field(default_factory=list)
    
    def add(self, constraint):
        self.constraints.append(constraint)
    
    def assertions(self):
        return self.constraints
    
    def check(self):
        print("\nCollected constraints:")
        for i, c in enumerate(self.constraints):
            print(f"{i+1}. {c}")
        return 'sat'
    
    def model(self):
        return MockModel()

@dataclass
class MockModel:
    def __getitem__(self, key):
        return MockValue(0)

@dataclass
class MockValue:
    value: int
    
    def as_long(self):
        return self.value

class MockBackend(Backend):
    def Int(self, name: str) -> MockExpr:
        return MockExpr('Int', [], _name=name)

    def IntVal(self, val: int) -> MockExpr:
        return MockExpr('IntVal', [val], _name=str(val))

    def BoolVal(self, val: bool) -> MockExpr:
        return MockExpr('BoolVal', [val], _name=str(val))

    def And(self, *args) -> MockExpr:
        return MockExpr('And', list(args))

    def Or(self, *args) -> MockExpr:
        return MockExpr('Or', list(args))

    def Not(self, arg) -> MockExpr:
        return MockExpr('Not', [arg])

    def Implies(self, a, b) -> MockExpr:
        return MockExpr('Implies', [a, b])

    def If(self, cond, t, f) -> MockExpr:
        return MockExpr('If', [cond, t, f])

    def ForAll(self, vars, body) -> MockExpr:
        return MockExpr('ForAll', [vars, body])

    def Distinct(self, args) -> MockExpr:
        return MockExpr('Distinct', args)

    def Mod(self, a, b) -> Any:
        return a + ' % ' + b

    def Pow(self, a, b) -> Any:
        return a + ' ** ' + b

    def Solver(self) -> MockSolver:
        return MockSolver()
    
    def is_sat(self, result) -> bool:
        return result == 'sat'

    def Mul(self, a, b) -> Any:
        return MockExpr('*', [a, b])
    
    def Add(self, a, b) -> Any:
        return MockExpr('+', [a, b])
    
    def Sub(self, a, b) -> Any:
        return MockExpr('-', [a, b])
    
    def Div(self, a, b) -> Any:
        return MockExpr('/', [a, b])
    
    def UDiv(self, a, b) -> Any:
        # Integer division
        return MockExpr('//', [a, b])
    
    def LT(self, a, b) -> Any:
        return MockExpr('<', [a, b])
    
    def LE(self, a, b) -> Any:
        return MockExpr('<=', [a, b])
    
    def GT(self, a, b) -> Any:
        return MockExpr('>', [a, b])
    
    def GE(self, a, b) -> Any:
        return MockExpr('>=', [a, b])
    
    def Eq(self, a, b) -> Any:
        return MockExpr('==', [a, b])
