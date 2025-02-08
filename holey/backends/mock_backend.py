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
        #print("Unparseable as int", v)
        return None
    return int(v)

@dataclass
class MockExpr:
    op: str
    args: List[Any]
    _name: Optional[str] = None

    def __init__(self, op: str, args):
        self.op = op
        self.args = args
        self._name = None

    def to_smt2(self) -> str:
        """Convert expression to SMT-LIB2 format"""
        if self._name is not None:
            return self._name
        if self.op == "IntVal":
            return str(self.args[0])
        elif self.op == "BoolVal":
            return str(self.args[0]).lower()
        elif self.op == 'str.val':
            return to_smtlib_string(self.args[0])
        elif self.op in ["Int", "String"]:
            # For variable references, just return the name
            return str(self.args[0])
        elif not self.args:
            return self.op
            
        args_str = " ".join(arg.to_smt2() if isinstance(arg, MockExpr) else str(arg).lower() if isinstance(arg, bool) else str(arg)
                          for arg in self.args)
        return f"({self.op} {args_str})"

    def __str__(self):
        return self.to_smt2()
    
    def __gt__(self, other):
        return MockExpr('>', [self, other])
    
    def __ge__(self, other):
        return MockExpr('>=', [self, other])
        
    def __lt__(self, other):
        return MockExpr('<', [self, other])
        
    def __le__(self, other):
        return MockExpr('<=', [self, other])
        
    def __eq__(self, other):
        return MockExpr('=', [self, other])
    
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

library = {
'str.reverse':
"""
(define-fun-rec str.reverse ((s String)) String
  (ite (= s "")
       ""
       (str.++ (str.substr s (- (str.len s) 1) 1)
               (str.reverse (str.substr s 0 (- (str.len s) 1))))))
"""
,
'isupper':
"""
(define-fun is-upper-char ((c String)) Bool
  (and (>= (str.to_code c) 65) (<= (str.to_code c) 90))
)

(define-fun-rec isupper ((s String)) Bool
  (ite (= s "")
       true
       (and (is-upper-char (str.at s 0))
            (isupper (str.substr s 1 (- (str.len s) 1))))
  )
)
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
,
'bin':
"""
(define-fun-rec bin.rec ((x Int) (n Int)) String
  (ite (<= n 0)
       ""
       (let ((bit (mod x 2))
             (rest (div x 2)))
         (str.++ (bin.rec rest (- n 1))
                 (ite (= bit 0) "0" "1")))))

(define-fun bin ((x Int)) String
  (str.++ "0b" (bin.rec x 32)))
"""
}

@dataclass
class MockSolver:
    def __init__(self):
        self.constraints = []
        self.declarations = set()
        self._model = {}
    
    def add(self, constraint):
        assert isinstance(constraint, MockExpr), "found bad constraint " + str(constraint) + " of type " + str(type(constraint))
        self.constraints.append(constraint)
    
    def model(self):
        return self._model

    def check(self):
        # Generate SMT-LIB2 file
        smt2 = ""

        # Declare variables
        for decl,sort in self.declarations:
            smt2 += f"(declare-const {decl} {sort})\n"
        
        # Assert constraints
        for c in self.constraints:
            if isinstance(c, bool):
                if not c:
                    smt2 += f"(assert (= 1 0))\n"
            else:
                assert isinstance(c, MockExpr), "found " + str(c) + " of type " + str(type(c))
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

        import sys
        sys.stdout.flush()

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
        self.operations = []
        self.solver = MockSolver()
        self.stack = []

    def _record(self, op: str, *args) -> Any:
        """Record operation and return a MockExpr"""
        self.operations.append((op, args))
        assert all(arg is not None for arg in args), "some arg is none: " + str(args)
        return MockExpr(op, args)

    def push(self):
        self.stack.append(self.solver.constraints.copy())
        return self._record("push")

    def pop(self):
        if self.stack:
            self.solver.constraints = self.stack.pop()
        return self._record("pop")

    def add(self, constraint):
        self.solver.add(constraint)
        return self._record("assert", constraint)

    def check(self):
        result = self.solver.check()
        self._record("check")
        return result

    def Not(self, x):
        return self._record("not", x)

    def reset(self):
        self.solver = MockSolver()
        self._record("reset")

    def Int(self, name: str) -> MockExpr:
        self.solver.declarations.add((name, 'Int'))
        return self._record("Int", name)

    def IntVal(self, val: int) -> MockExpr:
        return self._record("IntVal", val)

    def BoolVal(self, val: bool) -> MockExpr:
        return self._record("BoolVal", val)

    def And(self, *args) -> MockExpr:
        if not args:
            return self.BoolVal(True)
        if len(args) == 1:
            return args[0]
        return self._record("and", *args)

    def Or(self, *args) -> MockExpr:
        if not args:
            return self.BoolVal(False)
        if len(args) == 1:
            return args[0]
        return self._record("or", *args)

    def Implies(self, a, b) -> MockExpr:
        return self._record("=>", a, b)

    def If(self, cond, t, f) -> MockExpr:
        return self._record("ite", cond, t, f)

    def ForAll(self, vars, body) -> MockExpr:
        return self._record("forall", vars, body)

    def Distinct(self, *args) -> MockExpr:
        return self._record("distinct", *args)

    def Mod(self, a, b) -> MockExpr:
        """Create modulo expression with non-zero divisor constraint"""
        self.add(self.Not(self.Eq(b, self.IntVal(0))))  # Add constraint: b != 0
        return self._record("mod", a, b)

    def Pow(self, a, b) -> MockExpr:
        return self._record("^", a, b)

    def Solver(self) -> MockSolver:
        return self.solver

    def is_sat(self, result) -> bool:
        return result == 'sat'

    def Mul(self, a, b) -> MockExpr:
        return self._record("*", a, b)
    
    def Add(self, a, b) -> MockExpr:
        return self._record("+", a, b)
    
    def Sub(self, a, b) -> MockExpr:
        return self._record("-", a, b)
    
    def Div(self, a, b) -> MockExpr:
        return self._record("/", a, b)
    
    def UDiv(self, a, b) -> MockExpr:
        return self._record("div", a, b)
    
    def LT(self, a, b) -> MockExpr:
        return self._record("<", a, b)
    
    def LE(self, a, b) -> MockExpr:
        return self._record("<=", a, b)
    
    def GT(self, a, b) -> MockExpr:
        return self._record(">", a, b)
    
    def GE(self, a, b) -> MockExpr:
        return self._record(">=", a, b)
    
    def Eq(self, a, b) -> MockExpr:
        return self._record("=", a, b)

    def ToInt(self, x) -> MockExpr:
        return self._record("to_int", x)

    def IntToStr(self, x) -> MockExpr:
        return self._record("int.to.str", x)

    def StrToCode(self, x) -> MockExpr:
        return self._record("str.to_code", x)

    def StrToInt(self, x) -> MockExpr:
        return self._record("str.to.int", x)

    def StrIndex(self, x, y) -> MockExpr:
        return self._record("str.index", x, y)

    def StrLen(self, x) -> MockExpr:
        return self._record("str.len", x)

    def StrPrefixOf(self, x, y) -> MockExpr:
        return self._record("str.prefixof", x, y)

    def String(self, name: str) -> MockExpr:
        self.solver.declarations.add((name, 'String'))
        return self._record("String", name)

    def StringVal(self, val: str) -> MockExpr:
        return self._record("str.val", val)

    def StrReverse(self, s) -> MockExpr:
        return self._record("str.reverse", s)

    def StrCount(self, s, sub) -> MockExpr:
        return self._record("str.count", s, sub)

    def StrContains(self, x, y) -> MockExpr:
        return self._record("str.contains", x, y)

    def StrSubstr(self, s, start, length) -> MockExpr:
        return self._record("str.substr", s, start, length)

    def StrConcat(self, *args) -> MockExpr:
        return self._record("str.++", *args)

    def StrSplit(self, s, sep) -> MockExpr:
        """Split string by separator"""
        return self._record("str.to.re", s, sep)

    def Bin(self, x) -> MockExpr:
        return self._record("bin", x)

    def IsUpper(self, x) -> MockExpr:
        return self._record("str.is.upper", x)
