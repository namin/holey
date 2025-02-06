"""
Mock backend for collecting and displaying constraints.
"""
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
from .base import Backend
import tempfile
import subprocess
import os
import sexpdata

def to_smtlib_string(s):
    return '"' + ''.join(
        ch if ord(ch) < 128 else f"\\u{{{ord(ch):x}}}"
        for ch in s
    ) + '"'

def from_stmlib_int(v):
    if isinstance(v, list):
        if len(v)==2 and isinstance(v[0], sexpdata.Symbol) and v[0].value()=='-':
            return -int(v[1])
        print("Unparseable as int", v)
        return None
    return int(v)

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
        return MockExpr('-', [self])

    def __truediv__(self, other):
        return MockExpr('/', [self, other])

    def __and__(self, other):
        return MockExpr('and', [self, other])

    def __or__(self, other):
        return MockExpr('or', [self, other])

    def decl(self):
        return self
        
    def name(self) -> str:
        return self._name if self._name else str(self)

    def to_smt2(self) -> str:
        """Convert expression to SMT-LIB2 format"""
        if self._name is not None:
            return self._name
        if not self.args:
            return self.op
        if self.op == 'str.val':
            return to_smtlib_string(self.args[0])
            
        op = self.op
        
        args_str = " ".join(arg.to_smt2() if isinstance(arg, MockExpr) else str(arg).lower() if isinstance(arg, bool) else str(arg)
                          for arg in self.args)
        return f"({op} {args_str})"

library = {
'python.mod':
"""
(define-fun python.mod ((a Int) (b Int)) Int
  (let ((m (mod a b)))
    (ite (and (< m 0) (> b 0)) (+ m b)
         (ite (and (> m 0) (< b 0)) (+ m b) m))))
"""
,
'str.count':
"""
(define-fun-rec str.count.rec ((s String) (sub String) (start Int)) Int
  (let ((idx (str.indexof s sub start)))
    (ite (or (= idx (- 1)) (> start (str.len s)))
         0
         (+ 1 (str.count.rec s sub (+ idx (str.len sub)))))))

(define-fun str.count ((s String) (sub String)) Int
  (ite (= (str.len sub) 0)
       (+ 1 (str.len s))
       (str.count.rec s sub 0)))
"""
}

@dataclass
class MockSolver:
    def __init__(self):
        self.constraints = []
        self.declarations = set()
        self._model = {}
    
    def add(self, constraint):
        self.constraints.append(constraint)
    
    def model(self):
        return self._model

    def check(self):
        # Generate SMT-LIB2 file
        smt2 = ""

        # Declare variables
        for decl in self.declarations:
            smt2 += f"(declare-const {decl} Int)\n"
            
        # Assert constraints
        for c in self.constraints:
            if isinstance(c, bool):
                if not c:
                    smt2 += f"(assert (= 1 0))\n"
            else:
                assert isinstance(c, MockExpr)
                smt2 += f"(assert {c.to_smt2()})\n"
            
        smt2 += "(check-sat)\n(get-model)\n"

        smt2_preambule = "(set-logic ALL)\n" 
        for fun,defn in library.items():
            if fun in smt2:
                smt2_preambule += defn + "\n"
        smt2 = smt2_preambule + smt2

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
            f.write(smt2)
            smt2_file = f.name

        print('### smt2')
        print(smt2)
        
        try:
            # Run Z3 (or CVC5) on the file
            result = subprocess.run(['z3', '-T:2', smt2_file], capture_output=True, text=True)
            output = result.stdout.strip()
            
            print('### output')
            print(output)

            # Parse output
            if output.startswith('sat'):
                self._model = self._parse_model(output)
                return 'sat'
            elif output.startswith('unsat'):
                return 'unsat'
            else:
                return 'unknown'
        finally:
            os.unlink(smt2_file)
    
    def _parse_model(self, output):
        # Parse the entire output as S-expressions
        lines = output.strip().split('\n')
        if not lines[0] == 'sat':
            return {}
            
        # Join all lines after 'sat' into a single string
        model_str = ''.join(lines[1:])
        
        # Parse the model
        model_sexp = sexpdata.loads(model_str)
        _model = {}
        
        # Each definition in the model is a list like:
        # (define-fun x () Int 42)
        for defn in model_sexp:
            if defn[0].value() == 'define-fun':
                var_name = defn[1].value()
                value = defn[-1]
                typ = defn[3].value()
                _model[var_name] = from_stmlib_int(value) if typ == 'Int' else value
                
        return _model

class MockBackend(Backend):
    def __init__(self):
        self.reset()

    def reset(self):
        self.solver = MockSolver()

    def Int(self, name: str) -> MockExpr:
        self.solver.declarations.add(name)
        return MockExpr('Int', [], _name=name)

    def IntVal(self, val: int) -> MockExpr:
        return MockExpr('int_val', [val], _name=str(val))

    def BoolVal(self, val: bool) -> MockExpr:
        return MockExpr('bool_val', [val], _name=str(val))

    def And(self, *args) -> MockExpr:
        return MockExpr('and', list(args))

    def Or(self, *args) -> MockExpr:
        return MockExpr('or', list(args))

    def Not(self, arg) -> MockExpr:
        return MockExpr('not', [arg])

    def Implies(self, a, b) -> MockExpr:
        return MockExpr('implies', [a, b])

    def If(self, cond, t, f) -> MockExpr:
        return MockExpr('if', [cond, t, f])

    def ForAll(self, vars, body) -> MockExpr:
        return MockExpr('forall', [vars, body])

    def Distinct(self, args) -> MockExpr:
        return MockExpr('distinct', args)

    def Mod(self, a, b) -> MockExpr:
        return MockExpr('python.mod', [a, b])

    def Pow(self, a, b) -> MockExpr:
        return MockExpr('^', [a, b])

    def Solver(self) -> MockSolver:
        return self.solver
    
    def is_sat(self, result) -> bool:
        return result == 'sat'

    def Mul(self, a, b) -> MockExpr:
        return MockExpr('*', [a, b])
    
    def Add(self, a, b) -> MockExpr:
        return MockExpr('+', [a, b])
    
    def Sub(self, a, b) -> MockExpr:
        return MockExpr('-', [a, b])
    
    def Div(self, a, b) -> Any:
        return MockExpr('/', [a, b])
    
    def UDiv(self, a, b) -> Any:
        return MockExpr('div', [a, b])
    
    def LT(self, a, b) -> Any:
        return MockExpr('<', [a, b])
    
    def LE(self, a, b) -> Any:
        return MockExpr('<=', [a, b])
    
    def GT(self, a, b) -> Any:
        return MockExpr('>', [a, b])
    
    def GE(self, a, b) -> Any:
        return MockExpr('>=', [a, b])
    
    def Eq(self, a, b) -> Any:
        return MockExpr('=', [a, b])

    def IntToStr(self, x: Any) -> Any:
        return MockExpr('int.to.str', [x])

    def StrToInt(self, x: Any) -> Any:
        return MockExpr('str.to.int', [x])

    def StrLen(self, x: Any) -> Any:
       return MockExpr('str.len', [x])

    def StrPrefixOf(self, x: Any, y: Any) -> Any:
       return MockExpr('str.prefixof', [x, y])

    def StringVal(self, val: str) -> Any:
        return MockExpr('str.val', [val])

    def StrCount(self, s: Any, sub: Any) -> Any:
        return MockExpr('str.count', [s, sub])

    def StrSubstr(self, s: Any, start: Any, end: Any) -> Any:
        return MockExpr('str.substr', [s, start, end])
