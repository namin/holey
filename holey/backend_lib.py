"""
SMT-LIB2 library definitions for the Holey backend.

Each public-facing function is a separate entry with explicit dependencies.
Dependencies are listed as: entry -> [entries it requires]
"""

# =============================================================================
# ELEMENT TYPES for List operations
# =============================================================================

# Maps suffix -> (SMT type, nil expression)
ELEMENT_TYPES = {
    'int': ('Int', '(as nil (List Int))'),
    'string': ('String', '(as nil (List String))'),
    'real': ('Real', '(as nil (List Real))'),
    'bool': ('Bool', '(as nil (List Bool))'),
    'list_int': ('(List Int)', '(as nil (List (List Int)))'),
    'list_list_int': ('(List (List Int))', '(as nil (List (List (List Int))))'),
    'list_real': ('(List Real)', '(as nil (List (List Real)))'),
}

# =============================================================================
# TEMPLATES for List operations
# =============================================================================

def make_list_length(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    return f'''
(define-fun-rec list.length.{suffix} ((l {list_type})) Int
  (ite (= l {nil_expr})
       0
       (+ 1 (list.length.{suffix} (tail l)))))
'''

def make_list_get(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    return f'''
(define-fun-rec list.get.{suffix} ((l {list_type}) (idx Int)) {elem_type}
  (ite (< idx 0)
       (list.get.{suffix} l (+ (list.length.{suffix} l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.{suffix} (tail l) (- idx 1)))))
'''

def make_list_append(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    return f'''
(define-fun-rec list.append.{suffix} ((l1 {list_type}) (l2 {list_type})) {list_type}
  (ite (= l1 {nil_expr})
       l2
       (cons (head l1) (list.append.{suffix} (tail l1) l2))))
'''

def make_list_count(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    return f'''
(define-fun-rec list.count.{suffix} ((l {list_type}) (val {elem_type})) Int
  (ite (= l {nil_expr})
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.{suffix} (tail l) val))))
'''

def make_list_contains(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    return f'''
(define-fun list.contains.{suffix} ((l {list_type}) (val {elem_type})) Bool
  (> (list.count.{suffix} l val) 0))
'''

def make_list_reverse(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    return f'''
(define-fun-rec list.reverse.{suffix} ((l {list_type}) (acc {list_type})) {list_type}
  (ite (= l {nil_expr})
       acc
       (list.reverse.{suffix} (tail l) (cons (head l) acc))))
'''

def make_list_slice(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    return f'''
(define-fun list.valid_index.{suffix} ((l {list_type}) (idx Int)) Bool
  (and (>= idx 0) (< idx (list.length.{suffix} l))))

(define-fun-rec list.slice.{suffix}.helper ((l {list_type}) (curr Int) (stop Int) (step Int) (result {list_type})) {list_type}
  (ite (or (and (> step 0) (>= curr stop))
           (and (< step 0) (<= curr stop))
           (not (list.valid_index.{suffix} l curr)))
       result
       (let ((new_result (cons (list.get.{suffix} l curr) result)))
         (list.slice.{suffix}.helper l (+ curr step) stop step new_result))))

(define-fun list.slice.{suffix} ((l {list_type}) (start Int) (stop Int) (step Int)) {list_type}
  (let ((len (list.length.{suffix} l)))
    (ite (= step 0)
         {nil_expr}
         (let ((adj_start (list.adjust_index start len))
               (adj_stop (list.adjust_index stop len)))
           (ite (> step 0)
                (list.reverse.{suffix}
                  (list.slice.{suffix}.helper l adj_start adj_stop step {nil_expr})
                  {nil_expr})
                (let ((real_start (- len 1 adj_start))
                      (real_stop (- len 1 adj_stop)))
                  (list.reverse.{suffix}
                    (list.slice.{suffix}.helper l real_start real_stop (ite (< step 0) (- 0 step) step) {nil_expr})
                    {nil_expr})))))))
'''

# =============================================================================
# GENERATE list operations for all element types
# =============================================================================

def generate_list_library():
    """Generate library entries and dependencies for all list operations."""
    lib = {}
    deps = {}

    for suffix, (elem_type, nil_expr) in ELEMENT_TYPES.items():
        # list.length.{suffix}
        lib[f'list.length.{suffix}'] = make_list_length(suffix, elem_type, nil_expr)
        deps[f'list.length.{suffix}'] = ['list']

        # list.get.{suffix}
        lib[f'list.get.{suffix}'] = make_list_get(suffix, elem_type, nil_expr)
        deps[f'list.get.{suffix}'] = ['list', f'list.length.{suffix}']

        # list.append.{suffix}
        lib[f'list.append.{suffix}'] = make_list_append(suffix, elem_type, nil_expr)
        deps[f'list.append.{suffix}'] = ['list']

        # list.count.{suffix}
        lib[f'list.count.{suffix}'] = make_list_count(suffix, elem_type, nil_expr)
        deps[f'list.count.{suffix}'] = ['list']

        # list.contains.{suffix}
        lib[f'list.contains.{suffix}'] = make_list_contains(suffix, elem_type, nil_expr)
        deps[f'list.contains.{suffix}'] = ['list', f'list.count.{suffix}']

        # list.reverse.{suffix}
        lib[f'list.reverse.{suffix}'] = make_list_reverse(suffix, elem_type, nil_expr)
        deps[f'list.reverse.{suffix}'] = ['list']

        # list.slice.{suffix}
        lib[f'list.slice.{suffix}'] = make_list_slice(suffix, elem_type, nil_expr)
        deps[f'list.slice.{suffix}'] = ['list', 'list.adjust_index', f'list.length.{suffix}', f'list.get.{suffix}', f'list.reverse.{suffix}']

    return lib, deps

# =============================================================================
# STATIC library entries (non-templated)
# =============================================================================

library_static = {

# === DATATYPE ===

'list':
"""
(declare-datatypes ((List 1))
    ((par (T) ((cons (head T) (tail (List T))) (nil)))))
""",

# === LIST HELPERS (shared) ===

'list.adjust_index':
"""
(define-fun list.adjust_index ((idx Int) (len Int)) Int
  (ite (< idx 0)
       (+ len idx)
       idx))
""",

# === LIST[INT] SPECIAL FUNCTIONS ===

'list.index.int':
"""
(define-fun-rec list.index.rec.int ((i Int) (l (List Int)) (val Int)) Int
  (ite (= l (as nil (List Int)))
       -1
       (ite (= (head l) val)
            i
            (list.index.rec.int (+ 1 i) (tail l) val))))

(define-fun list.index.int ((l (List Int)) (val Int)) Int
  (list.index.rec.int 0 l val))
""",

'list.sum.int':
"""
(define-fun-rec list.sum.int ((l (List Int))) Int
  (ite (= l (as nil (List Int)))
       0
       (+ (head l) (list.sum.int (tail l)))))
""",

'list.map_add.int':
"""
(define-fun-rec list.map_add.int ((l (List Int)) (val Int)) (List Int)
  (ite (= l (as nil (List Int)))
       (as nil (List Int))
       (cons (+ (head l) val) (list.map_add.int (tail l) val))))
""",

# =============================================================================
# STRING FUNCTIONS
# =============================================================================

'str.isdigit':
"""
(define-fun str.isdigit ((s String)) Bool
  (forall ((i Int))
    (=> (and (<= 0 i) (< i (str.len s)))
        (let ((c (str.at s i)))
          (str.is_digit c)))))
""",

'str.isalpha':
"""
(define-fun str.isalpha ((s String)) Bool
  (and (> (str.len s) 0)
       (forall ((i Int))
         (=> (and (<= 0 i) (< i (str.len s)))
             (let ((c (str.to_code (str.at s i))))
               (or (and (>= c 65) (<= c 90))
                   (and (>= c 97) (<= c 122))))))))
""",

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
""",

'str.split':
"""
(define-fun is-delimiter ((c String) (delim String)) Bool
  (= c delim))

(define-fun substring ((s String) (start Int) (end Int)) String
  (let ((len (- end start)))
    (ite (or (< start 0) (< len 0) (> end (str.len s)))
         ""
         (str.substr s start len))))

(define-fun-rec loop-split ((s String) (delim String) (start Int) (pos Int)
                            (result (List String)) (len Int)) (List String)
  (ite (>= pos len)
       (let ((final-part (substring s start len)))
         (cons final-part result))
       (ite (is-delimiter (str.at s pos) delim)
            (let ((part (substring s start pos)))
              (let ((new-result (cons part result)))
                (loop-split s delim (+ pos 1) (+ pos 1) new-result len)))
            (loop-split s delim start (+ pos 1) result len))))

(define-fun str.split ((s String) (delim String)) (List String)
  (let ((len (str.len s)))
    (ite (= len 0)
         (cons "" (as nil (List String)))
         (let ((result (as nil (List String)))
               (start 0))
           (let ((result (loop-split s delim 0 0 result len)))
             result)))))
""",

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
""",

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
""",

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
""",

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
""",

'python.str.at':
"""
(define-fun python.str.at ((s String) (start Int)) String
  (let ((start (ite (< start 0) (+ (str.len s) start) start)))
    (str.substr s start 1)))
""",

'python.str.substr':
"""
(define-fun python.str.substr ((s String) (start Int) (end Int)) String
  (let ((start (ite (< start 0) (+ (str.len s) start) start))
        (end (ite (< end 0) (+ (str.len s) end) end)))
    (str.substr s start (- end start))))
""",

'str.to.float':
"""
(define-fun str.to.float ((s String)) Real
  (let ((dot_pos (str.indexof s "." 0)))
    (ite (= dot_pos (- 1))
      (to_real (str.to_int s))
      (let ((int_part (str.substr s 0 dot_pos))
            (dec_part (str.substr s (+ dot_pos 1) (- (str.len s) (+ dot_pos 1)))))
        (+ (to_real (str.to_int int_part))
           (/ (to_real (str.to_int dec_part))
              (^ 10.0 (- (str.len s) (+ dot_pos 1)))))))))
""",

'str.reverse':
"""
(define-fun-rec str.reverse ((s String)) String
  (ite (= s "")
       ""
       (str.++ (str.substr s (- (str.len s) 1) 1)
               (str.reverse (str.substr s 0 (- (str.len s) 1))))))
""",

'isupper':
"""
(define-fun is-upper-char ((c String)) Bool
  (and (>= (str.to_code c) 65) (<= (str.to_code c) 90)))

(define-fun-rec isupper ((s String)) Bool
  (ite (= s "")
       true
       (and (is-upper-char (str.at s 0))
            (isupper (str.substr s 1 (- (str.len s) 1))))))
""",

'islower':
"""
(define-fun is-lower-char ((c String)) Bool
  (and (>= (str.to_code c) (str.to_code "a")) (<= (str.to_code c) (str.to_code "z"))))

(define-fun-rec islower ((s String)) Bool
  (ite (= s "")
       true
       (and (is-lower-char (str.at s 0))
            (islower (str.substr s 1 (- (str.len s) 1))))))
""",

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
""",

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
""",

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
""",

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
""",

'str.from_real':
"""
(define-fun str.from_real ((r Real)) String
  (let ((is-negative (< r 0.0)))
    (let ((abs-r (ite is-negative (- 0.0 r) r)))
      (let ((int-part (to_int abs-r)))
        (let ((frac-part (- abs-r (to_real int-part))))
          (let ((int-str (str.from_int int-part)))
            (let ((sign-str (ite is-negative "-" "")))
              (let ((decimal-str "."))
                (let ((precision 6.0))
                  (let ((frac-expanded (to_int (* frac-part (^ 10.0 precision)))))
                    (let ((frac-str (str.from_int frac-expanded)))
                      (str.++ sign-str (str.++ int-str (str.++ decimal-str frac-str))))))))))))))
""",

# =============================================================================
# INT FUNCTIONS
# =============================================================================

'python.int.xor':
"""
(define-fun bool-to-int ((b Bool)) Int
  (ite b 1 0))

(define-fun int2bits ((x Int)) Bool
  (= (mod (abs x) 2) 1))

(define-fun python.int.xor ((x Int) (y Int)) Int
  (let ((bits (bool-to-int (xor (int2bits x) (int2bits y)))))
    bits))
""",

}

# Static dependencies
library_deps_static = {
    'list': [],
    'list.adjust_index': [],
    'list.index.int': ['list'],
    'list.sum.int': ['list'],
    'list.map_add.int': ['list'],
    'str.isdigit': [],
    'str.isalpha': [],
    'str.sorted': [],
    'str.split': [],
    'python.join': [],
    'swapcase': [],
    'str_multiply': [],
    'python.int': [],
    'python.str.at': [],
    'python.str.substr': [],
    'str.to.float': [],
    'str.reverse': [],
    'isupper': [],
    'islower': [],
    'str.upper': [],
    'str.lower': [],
    'str.count': [],
    'bin': [],
    'str.from_real': [],
    'python.int.xor': [],
}

# =============================================================================
# BUILD FINAL LIBRARY
# =============================================================================

# Generate list operations
_list_lib, _list_deps = generate_list_library()

# Combine static and generated
library = {**library_static, **_list_lib}
library_deps = {**library_deps_static, **_list_deps}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def resolve_dependencies(keys):
    """
    Given a set of library keys, return all keys needed (including transitive dependencies).
    Returns keys in topological order (dependencies first).
    """
    needed = set()
    order = []

    def visit(key):
        if key in needed:
            return
        needed.add(key)
        for dep in library_deps.get(key, []):
            visit(dep)
        order.append(key)

    for key in keys:
        visit(key)

    # Return in dependency order (dependencies before dependents)
    return order


def emit_library(keys):
    """
    Emit SMT-LIB2 code for the given library keys, including all dependencies.
    """
    resolved = resolve_dependencies(keys)
    return '\n'.join(library[k] for k in resolved if k in library)
