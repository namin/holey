"""
SMTLIB backend for collecting and displaying constraints.
"""
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, get_origin, get_args
import tempfile
import subprocess
import os
import sexpdata
import sys

TRUNCATE = os.environ.get('TRUNCATE', 'true') != 'false'

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

def from_stmlib_float(value):
    # Handle Real values, including fractions
    if isinstance(value, list) and len(value) == 3 and value[0].value() == '/':
        # It's a fraction like (/ 3223.0 25000.0)
        numerator = float(value[1])
        denominator = float(value[2])
        value = numerator / denominator
    elif isinstance(value, list) and len(value) == 2 and value[0].value() == '-':
        value = -from_stmlib_float(value[1])
    else:
        # It's a simple number
        value = float(str(value))
    return value

cmd_prefixes = {
    'z3': ['z3', '-T:2'],
    'cvc5': ['cvc5', '--tlimit=2000', '--produce-model', '--fmf-fun']
}
def smtlib_cmd(smt2_file, cmd=None):
    cmd = cmd or next(iter(cmd_prefixes.keys()))
    print('running backend', cmd)
    return cmd_prefixes[cmd] + [smt2_file]

def print_smt(smt2):
    lines = smt2.split('\n')
    if TRUNCATE:
        lines = [line if len(line) < 1005 else line[0:1000] + "..." for line in lines]
        lines = (lines[:50] + [";; ..."] + lines[-50:]) if len(lines) > 1000 else lines
    r = '\n'.join(lines)
    print(r)
    sys.stdout.flush()
    return r

def run_smt(smt2, cmds=None, puzzle_name=None, solver_stats=None):
    print('### smt2')
    print_smt(smt2)

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

        # Check if we should run all solvers for stats
        run_all = solver_stats is not None and solver_stats.run_all_solvers

        first_flag = None
        first_model = None
        result_flag = None
        result_model = None

        for (cmd, (flag, model)) in parsed:
            # Record solver result if stats tracking is enabled
            if solver_stats is not None and puzzle_name is not None and cmd is not None:
                solver_stats.add(puzzle_name, cmd, flag, model=model if flag == 'sat' else None)

            # Track first definitive result
            if result_flag is None:
                if flag == 'sat':
                    result_flag, result_model = flag, model
                elif flag == 'unsat':
                    result_flag, result_model = flag, model

            # Track first result of any kind
            if first_flag is None:
                first_flag = flag
                first_model = model

            # Early return if not collecting all stats
            if not run_all and result_flag is not None:
                return result_flag, result_model

        # Return best result found
        if result_flag is not None:
            return result_flag, result_model
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
    elif 'timeout' in output.lower():
        return 'timeout', None
    else:
        return 'unknown', None

def _parse_model(output):
    """Parse the SMT model output into a Python dictionary"""
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
    # (define-fun y () (List Int) (cons 1 (cons 2 nil)))
    for defn in model_sexp:
        if defn[0].value() == 'define-fun':
            var_name = defn[1].value()
            value = defn[-1]
            
            # Extract type information
            typ = defn[3]
            
            # Handle different types
            if isinstance(typ, sexpdata.Symbol):
                # Simple type like Int, String, etc.
                type_name = typ.value()
                if type_name == 'String':
                    value = from_smtlib_string(str(value))
                elif type_name == 'Int':
                    value = from_stmlib_int(value)
                elif type_name == 'Real':
                    value = from_stmlib_float(value)
                elif type_name == 'Bool':
                    value = str(value) == 'true'
            elif isinstance(typ, list) and len(typ) >= 2:
                # Complex type like (List Int)
                if isinstance(typ[0], sexpdata.Symbol) and typ[0].value() == 'List':
                    element_type = typ[1].value() if isinstance(typ[1], sexpdata.Symbol) else str(typ[1])
                    value = _parse_list_value(value)
            
            _model[var_name] = value

    return _model

def _parse_list_value(list_sexp, bindings=None):
    """Parse a list value from SMT-LIB output using a simple recursive approach"""
    if bindings is None:
        bindings = {}
    
    # Handle 'let' expressions: (let ((var val)...) body) -> parse body with substitutions
    if isinstance(list_sexp, list) and len(list_sexp) >= 3 and isinstance(list_sexp[0], sexpdata.Symbol) and list_sexp[0].value() == 'let':
        # Parse let bindings
        let_bindings = list_sexp[1]
        body = list_sexp[2]
        
        # Create new bindings dict with let-bound variables
        new_bindings = bindings.copy()
        for binding in let_bindings:
            if len(binding) >= 2:
                var_name = binding[0].value() if isinstance(binding[0], sexpdata.Symbol) else str(binding[0])
                var_value = _parse_sexp_value(binding[1], bindings)
                new_bindings[var_name] = var_value
        
        # Parse body with new bindings
        return _parse_list_value(body, new_bindings)
    
    # Handle variable references
    if isinstance(list_sexp, sexpdata.Symbol):
        sym_val = list_sexp.value()
        if sym_val in bindings:
            return bindings[sym_val]
        elif sym_val == 'nil':
            return []
        else:
            return list_sexp
    
    # Handle 'as' expressions for nil cases
    if isinstance(list_sexp, list) and len(list_sexp) >= 2 and isinstance(list_sexp[0], sexpdata.Symbol) and list_sexp[0].value() == 'as':
        # CVC5 format: (as nil (List Int)) with empty list as second element
        if len(list_sexp) >= 3 and list_sexp[1] == []:
            return []
        # Z3 format: (as nil (List Int)) -> extract nil
        elif len(list_sexp) >= 3 and isinstance(list_sexp[1], sexpdata.Symbol) and list_sexp[1].value() == 'nil':
            return []
        # Other as expressions - shouldn't happen for lists but handle gracefully
        else:
            return _parse_list_value(list_sexp[1], bindings)
    
    # Handle empty list represented as []
    if list_sexp == []:
        return []
        
    # Handle cons cell (cons head tail)
    if isinstance(list_sexp, list) and len(list_sexp) >= 3 and isinstance(list_sexp[0], sexpdata.Symbol) and list_sexp[0].value() == 'cons':
        head = list_sexp[1]
        tail = list_sexp[2]
        
        # Parse head value
        parsed_head = _parse_sexp_value(head, bindings)
        
        # Recursively parse the tail
        parsed_tail = _parse_list_value(tail, bindings)
        
        # Combine head and tail into a new list
        return [parsed_head] + parsed_tail
    
    # Handle CVC5 format: ((as cons (List Int)) head tail)
    if isinstance(list_sexp, list) and len(list_sexp) >= 3:
        # Check if first element is (as cons (List Int))
        first = list_sexp[0]
        if isinstance(first, list) and len(first) >= 3 and isinstance(first[0], sexpdata.Symbol) and first[0].value() == 'as':
            if isinstance(first[1], sexpdata.Symbol) and first[1].value() == 'cons':
                # This is a CVC5 cons cell
                head = list_sexp[1]
                tail = list_sexp[2]
                
                # Parse head value
                parsed_head = _parse_sexp_value(head, bindings)
                
                # Recursively parse the tail
                parsed_tail = _parse_list_value(tail, bindings)
                
                # Combine head and tail into a new list
                return [parsed_head] + parsed_tail
    
    # If we get here, it's an unexpected format
    print(f"Warning: Unexpected list format in SMT output: {list_sexp}")
    return []

def _parse_sexp_value(sexp, bindings=None):
    """Parse any S-expression value to an appropriate Python value"""
    if bindings is None:
        bindings = {}
    
    # Handle simple values
    if isinstance(sexp, (int, float, bool)):
        return sexp
    
    # Handle symbols
    if isinstance(sexp, sexpdata.Symbol):
        sym_val = sexp.value()
        # Check if it's a variable reference
        if sym_val in bindings:
            return bindings[sym_val]
        elif sym_val == 'true':
            return True
        elif sym_val == 'false':
            return False
        elif sym_val == 'nil':
            return []
        else:
            try:
                # Try to convert to int/float if possible
                return int(sym_val)
            except ValueError:
                try:
                    return float(sym_val)
                except ValueError:
                    return sym_val  # Return as string
    
    # Handle 'let' expressions
    if isinstance(sexp, list) and len(sexp) >= 3 and isinstance(sexp[0], sexpdata.Symbol) and sexp[0].value() == 'let':
        # Parse let bindings
        let_bindings = sexp[1]
        body = sexp[2]
        
        # Create new bindings dict with let-bound variables
        new_bindings = bindings.copy()
        for binding in let_bindings:
            if len(binding) >= 2:
                var_name = binding[0].value() if isinstance(binding[0], sexpdata.Symbol) else str(binding[0])
                var_value = _parse_sexp_value(binding[1], bindings)
                new_bindings[var_name] = var_value
        
        # Parse body with new bindings
        return _parse_sexp_value(body, new_bindings)
    
    # Handle 'as' expressions
    if isinstance(sexp, list) and len(sexp) >= 2 and isinstance(sexp[0], sexpdata.Symbol) and sexp[0].value() == 'as':
        # For list-related 'as' expressions, delegate to _parse_list_value
        # which already handles all the special cases
        return _parse_list_value(sexp, bindings)
    
    # Handle lists
    if isinstance(sexp, list):
        if len(sexp) >= 3 and isinstance(sexp[0], sexpdata.Symbol) and sexp[0].value() == 'cons':
            return _parse_list_value(sexp, bindings)
        elif len(sexp) >= 1 and isinstance(sexp[0], sexpdata.Symbol):
            # Handle other SMT-LIB expressions
            op = sexp[0].value()
            if op == '-' and len(sexp) == 2:
                # Unary minus
                return -_parse_sexp_value(sexp[1], bindings)
            elif op == '/' and len(sexp) == 3:
                # Division expression
                return _parse_sexp_value(sexp[1], bindings) / _parse_sexp_value(sexp[2], bindings)
        
        # Otherwise parse each element recursively
        return [_parse_sexp_value(elem, bindings) for elem in sexp]
    
    # String or other primitive
    return sexp

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
        elif self.op == "RealVal":
            val = self.args[0]
            # SMT-LIB2 doesn't understand scientific notation like 1e-06
            # Format as decimal or fraction
            if isinstance(val, float):
                # Use fixed-point notation with enough precision
                formatted = f"{val:.15f}".rstrip('0').rstrip('.')
                if '.' not in formatted:
                    formatted += '.0'
                return formatted
            return str(val)
        elif self.op == "BoolVal":
            return str(self.args[0]).lower()
        elif self.op == 'str.val':
            return to_smtlib_string(self.args[0])
        elif self.op in ["Int", "String", "Real", "Bool", "List"]:
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

    def contains_var(self, var_name: str) -> bool:
        """Check if this expression references the given variable name"""
        if self._name == var_name or (self.op == "Int" and self.args == [var_name]):
            return True
        return any(
            (arg.contains_var(var_name) if isinstance(arg, MockExpr) else arg == var_name)
            for arg in self.args
        )

# Import library definitions from separate module
from .backend_lib import library, library_deps, resolve_dependencies, emit_library

@dataclass
class MockSolver:
    def __init__(self, puzzle_name=None, solver_stats=None):
        self.constraints = []
        self.declarations = []
        self.extra_text = ""
        self._model = {}
        self.puzzle_name = puzzle_name
        self.solver_stats = solver_stats

    def add_text(self, text):
        self.extra_text += "\n" + text

    def add(self, constraint):
        if str(constraint) == 'True':
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
        smt2_lower = smt2.lower()
        # Find all library functions mentioned in the SMT code
        needed_keys = [key for key in library.keys() if key in smt2_lower]
        # Resolve dependencies and emit in topological order
        resolved_keys = resolve_dependencies(needed_keys)
        for key in resolved_keys:
            if key in library:
                smt2_preambule += library[key] + "\n"
        smt2 = smt2_preambule + smt2

        smt2 += self.extra_text

        flag, model = run_smt(smt2, cmd, self.puzzle_name, self.solver_stats)
        self._model = model
        return flag

class Backend():
    def __init__(self, cmds=None, puzzle_name=None, solver_stats=None):
        self.cmds = cmds
        self.puzzle_name = puzzle_name
        self.solver_stats = solver_stats
        self.operations = []
        self.solver = MockSolver(puzzle_name=puzzle_name, solver_stats=solver_stats)
        self.stack = []
        self.quantified_vars = set()
        self.id_counter = 0

    def next_id(self):
        self.id_counter += 1
        return self.id_counter

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

    def Real(self, name: str) -> MockExpr:
        if name not in self.quantified_vars:
            self.solver.declarations.append((name, 'Real'))
        return self._record("Real", name)

    def RealVal(self, val: float) -> MockExpr:
        return self._record("RealVal", val)

    def Bool(self, name: str) -> MockExpr:
        if name not in self.quantified_vars:
            self.solver.declarations.append((name, 'Bool'))
        return self._record("Bool", name)

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
    
    def Add(self, *args) -> MockExpr:
        return self._record("+", *args)
    
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
        # Use str.from_any_int to handle negative integers correctly
        # (str.from_int returns empty string for negative integers in SMT-LIB)
        return self._record("str.from_any_int", x)

    def RealToStr(self, x) -> MockExpr:
        return self._record("str.from_real", x)

    def StrToCode(self, x) -> MockExpr:
        return self._record("str.to_code", x)

    def CodeToStr(self, x) -> MockExpr:
        return self._record("str.from_code", x)

    def StrToInt(self, x, base=None) -> MockExpr:
        return self._record("python.int", x, base if base else self.IntVal(10))

    def StrToFloat(self, x) -> MockExpr:
        return self._record("str.to.float", x)
        
    def IntToFloat(self, x) -> MockExpr:
        return self._record("to_real", x)

    def StrReplace(self, x, y, z) -> MockExpr:
        return self._record("str.replace", x, y, z)

    def StrIndex(self, x, y) -> MockExpr:
        return self._record("python.str.at", x, y)

    def StrJoin(self, s, ss) -> MockExpr:
        return self._record("python.join", ss, s)

    def StrList(self, xs) -> MockExpr:
        if xs:
            return self._record("cons", xs[0], self.StrList(xs[1:]))
        else:
            return self._record("(as nil (List String))")

    def StrIndexOf(self, s, sub, start) -> MockExpr:
        return self._record("str.indexof", s, sub, start)

    def StrLT(self, a, b) -> MockExpr:
        return self._record("str.<", a, b)

    def StrLen(self, x) -> MockExpr:
        return self._record("str.len", x)

    def StrPrefixOf(self, x, y) -> MockExpr:
        return self._record("str.prefixof", x, y)

    def StrSuffixOf(self, x, y) -> MockExpr:
        return self._record("str.suffixof", x, y)

    def String(self, name: str) -> MockExpr:
        self.solver.declarations.append((name, 'String'))
        return self._record("String", name)

    def StringVal(self, val: str) -> MockExpr:
        return self._record("str.val", val)

    def StrReverse(self, s) -> MockExpr:
        return self._record("str.reverse", s)

    def StrSorted(self, s) -> MockExpr:
        return self._record("str.sorted", s)

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
        return self._record("str.split", s, sep)

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

    def StrStrip(self, x) -> MockExpr:
        return self._record("python.str.strip", x)

    def StrLStrip(self, x) -> MockExpr:
        return self._record("python.str.lstrip", x)

    def StrRStrip(self, x) -> MockExpr:
        return self._record("python.str.rstrip", x)

    def StrIsDigit(self, x) -> MockExpr:
        return self._record("str.isdigit", x)

    def StrIsAlpha(self, x) -> MockExpr:
        return self._record("str.isalpha", x)

    def SwapCase(self, x) -> MockExpr:
        return self._record("swapcase", x)

    def Type(self, typ: type) -> str:
        if typ == int:
            return "Int"
        if typ == bool:
            return "Bool"
        if typ == float:
            return "Real"
        if typ == str:
            return "String"
        if get_origin(typ) is list:
            elem_type = get_args(typ)[0]
            return f"(List {self.Type(elem_type)})"
        raise ValueError("Unsupported type " + str(typ))

    def _type_suffix(self, element_type: str) -> str:
        """Convert SMTLIB type string to valid function suffix.

        Examples:
            "Int" -> "int"
            "String" -> "string"
            "(List Int)" -> "list_int"
        """
        if element_type.startswith("(List "):
            inner = element_type[6:-1]  # extract inner type
            return "list_" + self._type_suffix(inner)
        return element_type.lower()

    def List(self, name: str, element_type: str) -> MockExpr:
        """Declare a list variable with elements of the given type"""
        full_type = f"(List {element_type})"
        if name not in self.quantified_vars:
            self.solver.declarations.append((name, full_type))
        return self._record("List", name, element_type)

    def ListVal(self, elements: list, element_type: str) -> MockExpr:
        """Create a list value with the given elements"""
        if not elements:
            return self._record(f"(as nil (List {element_type}))")

        result = self._record(f"(as nil (List {element_type}))")
        # Build the list in reverse order
        for element in reversed(elements):
            # Ensure the element is a MockExpr
            if not isinstance(element, MockExpr):
                if element_type == "Int":
                    element = self.IntVal(element)
                elif element_type == "Real":
                    element = self.RealVal(element)
                elif element_type == "Bool":
                    element = self.BoolVal(element)
                elif element_type == "String":
                    element = self.StringVal(element)

            result = self._record("cons", element, result)

        return result

    def ListLength(self, lst, element_type) -> MockExpr:
        """Get the length of a list"""
        return self._record("list.length."+self._type_suffix(element_type), lst)

    def ListSetLen(self, lst, element_type) -> MockExpr:
        """Get the number of distinct elements in a list (cardinality of set(list))"""
        return self._record("list.set_len."+self._type_suffix(element_type), lst)

    def ListGet(self, lst, idx, element_type) -> MockExpr:
        """Get an element from a list at the given index"""
        return self._record("list.get."+self._type_suffix(element_type), lst, idx)

    def ListSlice(self, lst, start, stop, step, element_type) -> MockExpr:
        return self._record("list.slice."+self._type_suffix(element_type), lst, start, stop, step)

    def ListContains(self, lst, val, element_type) -> MockExpr:
        """Check if a list contains a value"""
        return self._record("list.contains."+self._type_suffix(element_type), lst, val)

    def ListIndex(self, lst, val, element_type) -> MockExpr:
        return self._record("list.index."+self._type_suffix(element_type), lst, val)

    def ListSum(self, lst) -> MockExpr:
        """Get the sum of all elements in a list"""
        return self._record("list.sum.int", lst)

    def ListAppend(self, lst1, lst2, element_type) -> MockExpr:
        """Append two lists"""
        return self._record("list.append."+self._type_suffix(element_type), lst1, lst2)

    def ListMapAdd(self, lst, val) -> MockExpr:
        """Add a value to each element in a list"""
        return self._record("list.map_add", lst, val)

    def ListCons(self, head, tail) -> MockExpr:
        """Add an element to the front of a list"""
        return self._record("cons", head, tail)

    def ListNil(self, element_type: str) -> MockExpr:
        """Create an empty list of the given element type"""
        return self._record(f"(as nil (List {element_type}))")

    def ListCount(self, lst, val, element_type: str) -> MockExpr:
        """Count occurrences of a value in a list"""
        return self._record("list.count."+self._type_suffix(element_type), lst, val)

    def BoundedList(self, name: str, size: int, element_type: str) -> 'BoundedListVars':
        """Create a bounded list with individual variables"""
        return BoundedListVars(self, name, size, element_type)

class BoundedListVars:
    """Represents a bounded list using individual variables instead of recursive (List Int)"""
    def __init__(self, backend, name: str, size: int, element_type: str):
        self.backend = backend
        self.name = name
        self.size = size
        self.element_type = element_type
        self.vars = []

        # Create individual variables for each element
        for i in range(size):
            var_name = f"{name}_e{i}"
            if element_type == "Int":
                var = backend.Int(var_name)
            elif element_type == "String":
                var = backend.String(var_name)
            elif element_type == "Bool":
                var = backend.Bool(var_name)
            elif element_type == "Real":
                var = backend.Real(var_name)
            else:
                raise ValueError(f"Unsupported bounded list element type: {element_type}")
            self.vars.append(var)

    def get(self, idx):
        """Get element at index - handles both concrete and symbolic indices"""
        if isinstance(idx, int):
            if idx < 0:
                idx = self.size + idx
            if 0 <= idx < self.size:
                return self.vars[idx]
            raise IndexError(f"Index {idx} out of bounds for list of size {self.size}")

        # For symbolic index, build ITE chain
        if isinstance(idx, MockExpr):
            # Handle negative indices symbolically
            idx_expr = idx
            result = self.vars[0]  # Default to first element
            for i in range(self.size - 1, -1, -1):
                result = self.backend.If(
                    self.backend.Or(
                        self.backend.Eq(idx_expr, self.backend.IntVal(i)),
                        self.backend.Eq(idx_expr, self.backend.IntVal(i - self.size))
                    ),
                    self.vars[i],
                    result
                )
            return result

        raise ValueError(f"Unsupported index type: {type(idx)}")

    def sum(self):
        """Get sum of all elements"""
        if not self.vars:
            return self.backend.IntVal(0)
        if len(self.vars) == 1:
            return self.vars[0]
        return self.backend.Add(*self.vars)

    def length(self):
        """Get length of list"""
        return self.backend.IntVal(self.size)

    def prefix_sum(self, end_idx):
        """Get sum of elements from 0 to end_idx (exclusive)"""
        if isinstance(end_idx, int):
            if end_idx <= 0:
                return self.backend.IntVal(0)
            if end_idx >= self.size:
                return self.sum()
            if end_idx == 1:
                return self.vars[0]
            return self.backend.Add(*self.vars[:end_idx])

        # For symbolic end index, build ITE chain
        if isinstance(end_idx, MockExpr):
            result = self.backend.IntVal(0)
            for i in range(self.size, 0, -1):
                if i == 1:
                    prefix_sum = self.vars[0]
                else:
                    prefix_sum = self.backend.Add(*self.vars[:i])
                result = self.backend.If(
                    self.backend.Eq(end_idx, self.backend.IntVal(i)),
                    prefix_sum,
                    result
                )
            return result

        raise ValueError(f"Unsupported end_idx type: {type(end_idx)}")

    def to_smt2(self):
        """Return SMT2 representation for get-value"""
        return " ".join(var.to_smt2() for var in self.vars)


default_backend = Backend
