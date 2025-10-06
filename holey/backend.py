"""
SMTLIB backend for collecting and displaying constraints.
"""
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
import tempfile
import subprocess
import os
import sexpdata
import sys

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
    'cvc5': ['my-cvc5', '--tlimit=2000', '--produce-model', '--fmf-fun', '--fmf-fun-rlv', '--unroll=20']
}
def smtlib_cmd(smt2_file, cmd=None):
    cmd = cmd or next(iter(cmd_prefixes.keys()))
    print('running backend', cmd)
    return cmd_prefixes[cmd] + [smt2_file]

def print_smt(smt2):
    lines = smt2.split('\n')
    truncated_lines = [line if len(line) < 1005 else line[0:1000] + "..." for line in lines]
    r = '\n'.join(truncated_lines)
    print(r)
    sys.stdout.flush()
    return r

def run_smt(smt2, cmds=None):
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
            return str(self.args[0])
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

# from lib to users
library_deps = {
'list': ['str.split', 'python.join', 'list.slice']
}

library = {
'str.isdigit':
"""
(define-fun str.isdigit ((s String)) Bool
  (forall ((i Int))
    (=> (and (<= 0 i) (< i (str.len s)))
        (let ((c (str.at s i)))
          (str.is_digit c)))))
"""
,
'list':
"""
(declare-datatypes ((List 1)) 
    ((par (T) ((cons (head T) (tail (List T))) (nil)))))

(define-fun-rec list.length.int ((l (List Int))) Int
  (ite (= l (as nil (List Int)))
       0
       (+ 1 (list.length.int (tail l)))))

(define-fun-rec list.get.int ((l (List Int)) (idx Int)) Int
  (ite (<= idx 0)
       (head l)
       (list.get.int (tail l) (- idx 1))))

(define-fun-rec list.index.rec.int ((i Int) (l (List Int)) (val Int)) Int
  (ite (= l (as nil (List Int)))
       -1
       (ite (= (head l) val)
            i
            (list.index.rec.int (+ 1 i) (tail l) val))))

(define-fun list.index.int ((l (List Int)) (val Int)) Int
  (list.index.rec.int 0 l val))

(define-fun-rec list.length.string ((l (List String))) Int
  (ite (= l (as nil (List String)))
       0
       (+ 1 (list.length.string (tail l)))))

(define-fun-rec list.get.string ((l (List String)) (idx Int)) String
  (ite (<= idx 0)
       (head l)
       (list.get.string (tail l) (- idx 1))))

(define-fun-rec list.sum.int ((l (List Int))) Int
  (ite (= l (as nil (List Int)))
       0
       (+ (head l) (list.sum.int (tail l)))))

(define-fun-rec list.append.int ((l1 (List Int)) (l2 (List Int))) (List Int)
  (ite (= l1 (as nil (List Int)))
       l2
       (cons (head l1) (list.append.int (tail l1) l2))))

(define-fun-rec list.append.string ((l1 (List String)) (l2 (List String))) (List String)
  (ite (= l1 (as nil (List String)))
       l2
       (cons (head l1) (list.append.string (tail l1) l2))))

(define-fun-rec list.map_add.int ((l (List Int)) (val Int)) (List Int)
  (ite (= l (as nil (List Int)))
       (as nil (List Int))
       (cons (+ (head l) val) (list.map_add.int (tail l) val))))


(define-fun-rec list.count.int ((l (List Int)) (val Int)) Int
  (ite (= l (as nil (List Int)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.int (tail l) val))))

(define-fun list.contains.int ((l (List Int)) (val Int)) Bool
  (> (list.count.int l val) 0))

(define-fun-rec list.count.string ((l (List String)) (val String)) Int
  (ite (= l (as nil (List String)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.string (tail l) val))))

(define-fun list.contains.string ((l (List String)) (val String)) Bool
  (> (list.count.string l val) 0))

(define-fun-rec list.count.bool ((l (List Bool)) (val Bool)) Int
  (ite (= l (as nil (List Bool)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.bool (tail l) val))))

(define-fun-rec list.count.real ((l (List Real)) (val Real)) Int
  (ite (= l (as nil (List Real)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.real (tail l) val))))
"""
,
'list.slice':
"""
(define-fun list.adjust_index ((idx Int) (len Int)) Int
  (ite (< idx 0)
       (+ len idx)
       idx))
(define-fun list.valid_index.int ((l (List Int)) (idx Int)) Bool
  (and (>= idx 0) (< idx (list.length.int l))))

(define-fun-rec list.slice.int.helper ((l (List Int)) (curr Int) (stop Int) (step Int) (result (List Int))) (List Int)
  (ite (or (and (> step 0) (>= curr stop))     ;; Positive step and reached/passed stop
           (and (< step 0) (<= curr stop))     ;; Negative step and reached/passed stop
           (not (list.valid_index.int l curr))) ;; Index out of bounds
       result
       (let ((new_result (cons (list.get.int l curr) result)))
         (list.slice.int.helper l (+ curr step) stop step new_result))))

(define-fun-rec list.reverse.int ((l (List Int)) (acc (List Int))) (List Int)
  (ite (= l (as nil (List Int)))
       acc
       (list.reverse.int (tail l) (cons (head l) acc))))

(define-fun list.slice.int ((l (List Int)) (start Int) (stop Int) (step Int)) (List Int)
  (let ((len (list.length.int l)))
    (ite (= step 0)
         (as nil (List Int))  ;; Invalid step, return empty list
         (let ((adj_start (list.adjust_index start len))
               (adj_stop (list.adjust_index stop len)))
           (ite (> step 0)
                ;; For positive step
                (list.reverse.int 
                  (list.slice.int.helper l adj_start adj_stop step (as nil (List Int)))
                  (as nil (List Int)))
                ;; For negative step, reverse parameters
                (let ((real_start (- len 1 adj_start))
                      (real_stop (- len 1 adj_stop)))
                  (list.reverse.int 
                    (list.slice.int.helper l real_start real_stop (ite (< step 0) (- 0 step) step) (as nil (List Int)))
                    (as nil (List Int)))))))))
"""
,
'str.split':
"""
; Helper function to check if a character matches the delimiter
(define-fun is-delimiter ((c String) (delim String)) Bool
  (= c delim))

; Helper function to get the substring from start to end (exclusive)
(define-fun substring ((s String) (start Int) (end Int)) String
  (let ((len (- end start)))
    (ite (or (< start 0) (< len 0) (> end (str.len s)))
         ""
         (str.substr s start len))))

; Recursive helper function to do the actual splitting
; This simulates a loop through the string
(define-fun-rec loop-split ((s String) (delim String) (start Int) (pos Int) 
                            (result (List String)) (len Int)) (List String)
  (ite (>= pos len)
       ; If we reached the end of the string, add the final substring
       (let ((final-part (substring s start len)))
         (cons final-part result))
       ; If not at the end, check if current character is a delimiter
       (ite (is-delimiter (str.at s pos) delim)
            ; If it's a delimiter, add the substring to result and continue
            (let ((part (substring s start pos)))
              (let ((new-result (cons part result)))
                (loop-split s delim (+ pos 1) (+ pos 1) new-result len)))
            ; If not a delimiter, just continue
            (loop-split s delim start (+ pos 1) result len))))

(define-fun str.split ((s String) (delim String)) (List String)
  (let ((len (str.len s)))
    (ite (= len 0)
         (cons "" (as nil (List String)))
         (let ((result (as nil (List String)))
               (start 0))
           ; We need to manually iterate through the string
           ; and build our list of substrings
           (let ((result (loop-split s delim 0 0 result len)))
             result)))))
"""
,
'str.sorted':
"""
(define-fun-rec str.min_char ((s String)) String
  (let ((len (str.len s)))
    (ite (<= len 1)
         s
         (let ((first (str.at s 0))
               (rest_min (str.min_char (str.substr s 1 (- len 1)))))
           (ite (str.< first rest_min)
                first
                rest_min)))))

(define-fun-rec str.remove_first_occurrence ((s String) (c String)) String
  (let ((len (str.len s)))
    (ite (= len 0)
         ""
         (ite (= (str.at s 0) c)
              (str.substr s 1 (- len 1))
              (str.++ (str.substr s 0 1) 
                     (str.remove_first_occurrence (str.substr s 1 (- len 1)) c))))))

(define-fun-rec str.sorted ((s String)) String
  (let ((len (str.len s)))
    (ite (= len 0)
         ""
         (let ((min_c (str.min_char s)))
           (str.++ min_c (str.sorted (str.remove_first_occurrence s min_c)))))))
"""
,
'python.join':
"""
(define-fun-rec python.join ((lst (List String)) (delim String)) String
    (ite (= lst (as nil (List String)))
         ""
         (ite (= (tail lst) (as nil (List String)))
              (head lst)
              (str.++ (head lst) 
                     delim 
                     (python.join (tail lst) delim)))))
"""
,
'swapcase':
"""
(define-fun is_upper ((c String)) Bool
  (and 
    (>= (str.to_code c) 65)
    (<= (str.to_code c) 90)))

(define-fun is_lower ((c String)) Bool
  (and
    (>= (str.to_code c) 97)
    (<= (str.to_code c) 122)))

(define-fun to_lower ((c String)) String
  (let ((code (str.to_code c)))
    (str.from_code (+ code 32))))

(define-fun to_upper ((c String)) String
  (let ((code (str.to_code c)))
    (str.from_code (- code 32))))

(define-fun swapcase_char ((c String)) String
  (ite (is_upper c)
       (to_lower c)
       (ite (is_lower c)
            (to_upper c)
            c)))

(define-fun-rec swapcase_helper ((i Int) (n Int) (s String)) String
  (ite (>= i n)
       ""
       (str.++ (swapcase_char (str.at s i))
               (swapcase_helper (+ i 1) n s))))

(define-fun swapcase ((s String)) String
  (swapcase_helper 0 (str.len s) s))
"""
,
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
                    (str.from_int (+ (str.to_int "A") offset)))
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
                    (str.from_int (+ (str.to_int "a") offset)))
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
,
"str.from_real"
:
"""
(define-fun str.from_real ((r Real)) String
  (let ((is-negative (< r 0.0)))
    (let ((abs-r (ite is-negative (- 0.0 r) r)))
      (let ((int-part (to_int abs-r)))
        (let ((frac-part (- abs-r (to_real int-part))))
          (let ((int-str (str.from_int int-part)))
            (let ((sign-str (ite is-negative "-" "")))
              (let ((decimal-str "."))
                (let ((precision 6.0)) ;; Show 6 decimal places
                  (let ((frac-expanded (to_int (* frac-part (^ 10.0 precision)))))
                    (let ((frac-str (str.from_int frac-expanded)))
                      ;; Combine all parts: sign + integer part + decimal point + fraction part
                      (str.++ sign-str (str.++ int-str (str.++ decimal-str frac-str))))))))))))))
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
        smt2_lower = smt2.lower()
        for fun,defn in library.items():
            if fun in smt2_lower or any([user_fun in smt2_lower for user_fun in library_deps.get(fun, [])]):
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
        return self._record("str.from_int", x)

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

    def StrIsDigit(self, x) -> MockExpr:
        return self._record("str.isdigit", x)

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
        raise ValueError("Unsupported type " + str(typ))

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
        return self._record("list.length."+element_type.lower(), lst)

    def ListGet(self, lst, idx, element_type) -> MockExpr:
        """Get an element from a list at the given index"""
        return self._record("list.get."+element_type.lower(), lst, idx)

    def ListSlice(self, lst, start, stop, step, element_type) -> MockExpr:
        return self._record("list.slice."+element_type.lower(), lst, start, stop, step)

    def ListContains(self, lst, val, element_type) -> MockExpr:
        """Check if a list contains a value"""
        return self._record("list.contains."+element_type.lower(), lst, val)

    def ListIndex(self, lst, val, element_type) -> MockExpr:
        return self._record("list.index."+element_type.lower(), lst, val)

    def ListSum(self, lst) -> MockExpr:
        """Get the sum of all elements in a list"""
        return self._record("list.sum.int", lst)

    def ListAppend(self, lst1, lst2, element_type) -> MockExpr:
        """Append two lists"""
        return self._record("list.append."+element_type.lower(), lst1, lst2)

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
        return self._record("list.count."+element_type.lower(), lst, val)

default_backend = Backend
