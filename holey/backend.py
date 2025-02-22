"""
SMTLIB backend for collecting and displaying constraints.
"""
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import tempfile
import subprocess
import os
import sexpdata

def to_smtlib_string(s):
    return '"' + ''.join(
        ch if ord(ch) < 128 else f"\\u{{{ord(ch):x}}}"
        for ch in s
    ) + '"'

def from_smtlib_string(s):
    """Convert SMT-LIB string with unicode escapes back to Python string"""
    import re
    
    def replace_unicode(match):
        hex_val = match.group(1)
        return chr(int(hex_val, 16))
    
    # Replace \u{XX} with actual unicode characters
    return re.sub(r'\\u\{([0-9a-fA-F]+)\}', replace_unicode, s)

def from_stmlib_int(v):
    if isinstance(v, list):
        if len(v)==2 and isinstance(v[0], sexpdata.Symbol) and v[0].value()=='-':
            return -int(v[1])
        #print("Unparseable as int", v)
        return None
    return int(v)

cmd_prefixes = {
    'z3': ['z3', '-T:2'],
    'cvc5': ['cvc5', '--tlimit=2000', '--produce-model']
}
def smtlib_cmd(smt2_file, cmd=None):
    cmd = cmd or next(iter(cmd_prefixes.keys()))
    print('running backend', cmd)
    return cmd_prefixes[cmd] + [smt2_file]

def run_smt(smt2, cmds=None):
    print('### smt2')
    print(smt2)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smt2', delete=False) as f:
        f.write(smt2)
        smt2_file = f.name

    first_flag = None
    try:
        ps = []
        for cmd in cmds or [None]:
            ps.append((cmd,
                       subprocess.Popen(smtlib_cmd(smt2_file, cmd), 
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True)))
        outs = []
        for (cmd,p) in ps:
            output, error = p.communicate()
            outs.append((cmd, (output+error).strip()))

        parsed = []
        for (cmd, output) in outs:
            output = output.replace(smt2_file, "tmp.smt2")

            print('### output' + ' for ' + cmd if cmd is not None else '')
            print(output)

            parsed.append((cmd, parse_output(output)))

        first_flag = None
        first_model = None
        for (cmd, (flag, model)) in parsed:
            if flag == 'sat':
                return flag, model
            elif flag == 'unsat':
                return flag, model
            elif first_flag is None:
                first_flag = flag
                first_model = model
        return first_flag, first_model
    finally:
        os.unlink(smt2_file)

def parse_output(output):
    # Parse output
    if output.startswith('sat'):
        model = _parse_model(output)
        return 'sat', model
    elif output.startswith('unsat'):
        return 'unsat', None
    else:
        return 'unknown', None

def _parse_model(output):
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
            if typ == 'String':
                value = from_smtlib_string(str(value))
            elif typ == 'Int':
                value = from_stmlib_int(value)
            _model[var_name] = value

    return _model

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
        elif self.op == "forall":
            vars = self.args[0]
            body = self.args[1]
            vars_str = " ".join(f"({var.args[0]} Int)" for var in vars)
            body_str = body.to_smt2() if isinstance(body, MockExpr) else str(body)
            return f"(forall ({vars_str}) {body_str})"

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
'str_multiply':
"""
(define-fun-rec str.rev ((s String)) String
  (ite (= s "")
       ""
       (str.++ (str.substr s (- (str.len s) 1) 1)
               (str.rev (str.substr s 0 (- (str.len s) 1))))))
(define-fun-rec str_multiply_helper ((s String) (n Int) (acc String)) String
  (ite (<= n 0)
    acc
    (str_multiply_helper s (- n 1) (str.++ acc s))))
(define-fun str_multiply ((s String) (n Int)) String
  (ite (< n 0)
    (str.rev (str_multiply_helper s (- 0 n) ""))
    (str_multiply_helper s n "")))
"""
,
'python.int.xor':
"""
(define-fun bool-to-int ((b Bool)) Int
  (ite b 1 0))

(define-fun int2bits ((x Int)) Bool
  (= (mod (abs x) 2) 1))

(define-fun python.int.xor ((x Int) (y Int)) Int
  (let ((bits (bool-to-int (xor (int2bits x) (int2bits y)))))
    bits))
"""
,
'python.int':
"""
(define-fun-rec str-to-int ((s String) (base Int)) Int
  (let ((len (str.len s)))
    (to_int 
      (ite (<= len 0) 
           0.0
           (+ (* (to_real (- (str.to_code (str.substr s (- len 1) 1)) 48))
                 (^ (to_real base) (to_real (- len 1))))
              (to_real (str-to-int (str.substr s 0 (- len 1)) base)))))))

(define-fun-rec bin-to-int ((s String)) Int
  (let ((len (str.len s)))
    (ite (<= len 0) 
         0
         (+ (* (ite (= (str.substr s (- len 1) 1) "1") 1 0)
               (to_int (^ 2.0 (to_real (- len 1)))))
            (bin-to-int (str.substr s 0 (- len 1)))))))

(define-fun python.int ((s String) (base Int)) Int
  (ite (= base 10) (str.to_int s) (ite (= base 2) (bin-to-int s) (str-to-int s base))))
"""
,
'python.str.at':
"""
(define-fun python.str.at ((s String) (start Int)) String
  (let ((start (ite (< start 0) (+ (str.len s) start) start)))
    (str.substr s start 1)))
"""
,
'python.str.substr':
"""
(define-fun python.str.substr ((s String) (start Int) (end Int)) String
  (let ((start (ite (< start 0) (+ (str.len s) start) start))
        (end (ite (< end 0) (+ (str.len s) end) end)))
    (str.substr s start (- end start))))
"""
,
'str.to.float':
"""
(define-fun str.to.float ((s String)) Real
  (let ((dot_pos (str.indexof s "." 0)))
    (ite (= dot_pos (- 1))
      ; No decimal point - convert whole string as integer
      (to_real (str.to_int s))
      ; Has decimal point - handle integer and decimal parts
      (let ((int_part (str.substr s 0 dot_pos))
            (dec_part (str.substr s (+ dot_pos 1) (- (str.len s) (+ dot_pos 1)))))
        (+ (to_real (str.to_int int_part))
           (/ (to_real (str.to_int dec_part))
              (^ 10.0 (- (str.len s) (+ dot_pos 1)))))))))
"""
,
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
'islower':
"""
(define-fun is-lower-char ((c String)) Bool
  (and (>= (str.to_code c) (str.to_code "a")) (<= (str.to_code c) (str.to_code "z")))
)

(define-fun-rec islower ((s String)) Bool
  (ite (= s "")
       true
       (and (is-lower-char (str.at s 0))
            (islower (str.substr s 1 (- (str.len s) 1))))
  )
)
"""
,
'str.upper':
"""
(define-fun-rec str.upper ((s String)) String
  (let ((len (str.len s)))
    (ite (= len 0) 
         ""
         (let ((first (str.at s 0)))
           (str.++ 
             (ite (and (str.< "a" first) (str.< first "z"))
                  (let ((offset (- (str.to_int first) (str.to_int "a"))))
                    (str.from.int (+ (str.to_int "A") offset)))
                  first)
             (str.upper (str.substr s 1 (- len 1))))))))
"""
,
'str.lower':
"""
(define-fun-rec str.lower ((s String)) String
  (let ((len (str.len s)))
    (ite (= len 0) 
         ""
         (let ((first (str.at s 0)))
           (str.++ 
             (ite (and (str.< "A" first) (str.< first "Z"))
                  (let ((offset (- (str.to_int first) (str.to_int "A"))))
                    (str.from.int (+ (str.to_int "a") offset)))
                  first)
             (str.lower (str.substr s 1 (- len 1))))))))
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
        self.declarations = []
        self.extra_text = ""
        self._model = {}

    def add_text(self, text):
        self.extra_text += "\n" + text

    def add(self, constraint):
        if str(constraint) == 'True':
            print('Skipping constant true constraint')
            return
        assert isinstance(constraint, MockExpr), "found bad constraint " + str(constraint) + " of type " + str(type(constraint))
        self.constraints.append(constraint)
    
    def model(self):
        return self._model

    def check(self, cmd=None):
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

        smt2 += self.extra_text

        flag, model = run_smt(smt2, cmd)
        self._model = model
        return flag

class Backend():
    def __init__(self, cmds=None):
        self.cmds = cmds
        self.operations = []
        self.solver = MockSolver()
        self.stack = []
        self.quantified_vars = set()

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
        result = self.solver.check(self.cmds)
        self._record("check")
        return result

    def Not(self, x):
        return self._record("not", x)

    def Int(self, name: str) -> MockExpr:
        if name not in self.quantified_vars:
            self.solver.declarations.append((name, 'Int'))
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

    def Xor(self, a, b) -> MockExpr:
        return self._record("python.int.xor", a, b)

    def Implies(self, a, b) -> MockExpr:
        return self._record("=>", a, b)

    def If(self, cond, t, f) -> MockExpr:
        return self._record("ite", cond, t, f)

    def ForAll(self, vars, body) -> MockExpr:
        return self._record("forall", vars, body)

    def Distinct(self, *args) -> MockExpr:
        return self._record("distinct", *args)

    def Mod(self, a, b) -> MockExpr:
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
        return self._record("str.from_int", x)

    def StrToCode(self, x) -> MockExpr:
        return self._record("str.to_code", x)

    def StrToInt(self, x, base=None) -> MockExpr:
        return self._record("python.int", x, base if base else self.IntVal(10))

    def StrToFloat(self, x) -> MockExpr:
        return self._record("str.to.float", x)

    def StrReplace(self, x, y, z) -> MockExpr:
        return self._record("str.replace", x, y, z)

    def StrIndex(self, x, y) -> MockExpr:
        return self._record("python.str.at", x, y)

    def StrLen(self, x) -> MockExpr:
        return self._record("str.len", x)

    def StrPrefixOf(self, x, y) -> MockExpr:
        return self._record("str.prefixof", x, y)

    def String(self, name: str) -> MockExpr:
        self.solver.declarations.append((name, 'String'))
        return self._record("String", name)

    def StringVal(self, val: str) -> MockExpr:
        return self._record("str.val", val)

    def StrReverse(self, s) -> MockExpr:
        return self._record("str.reverse", s)

    def StrCount(self, s, sub) -> MockExpr:
        return self._record("str.count", s, sub)

    def StrContains(self, x, y) -> MockExpr:
        return self._record("str.contains", x, y)

    def StrSubstr(self, s, a, b, variant="python.str.substr") -> MockExpr:
        return self._record(variant, s, a, b)

    def StrConcat(self, *args) -> MockExpr:
        return self._record("str.++", *args)

    def StrMul(self, s, n) -> MockExpr:
        return self._record("str_multiply", s, n)

    def StrSplit(self, s, sep) -> MockExpr:
        """Split string by separator"""
        return self._record("str.to.re", s, sep)

    def Bin(self, x) -> MockExpr:
        return self._record("bin", x)

    def IsUpper(self, x) -> MockExpr:
        return self._record("isupper", x)

    def IsLower(self, x) -> MockExpr:
        return self._record("islower", x)

    def StrUpper(self, x) -> MockExpr:
        return self._record("str.upper", x)

    def StrLower(self, x) -> MockExpr:
        return self._record("str.lower", x)

default_backend = Backend
