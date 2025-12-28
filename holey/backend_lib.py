"""
SMT-LIB2 library definitions for the Holey backend.

Each public-facing function is a separate entry with explicit dependencies.
Dependencies are listed as: entry -> [entries it requires]
"""

# Dependencies: maps library key -> list of keys it depends on
library_deps = {
    # === DATATYPE ===
    'list': [],

    # === STRING FUNCTIONS ===
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

    # === INT FUNCTIONS ===
    'python.int.xor': [],

    # === LIST[INT] FUNCTIONS ===
    'list.length.int': ['list'],
    'list.get.int': ['list', 'list.length.int'],
    'list.index.int': ['list'],
    'list.sum.int': ['list'],
    'list.append.int': ['list'],
    'list.map_add.int': ['list'],
    'list.count.int': ['list'],
    'list.contains.int': ['list', 'list.count.int'],
    'list.reverse.int': ['list'],
    'list.slice.int': ['list', 'list.length.int', 'list.get.int', 'list.reverse.int'],

    # === LIST[STRING] FUNCTIONS ===
    'list.length.string': ['list'],
    'list.get.string': ['list', 'list.length.string'],
    'list.append.string': ['list'],
    'list.count.string': ['list'],
    'list.contains.string': ['list', 'list.count.string'],
    'list.reverse.string': ['list'],
    'list.slice.string': ['list', 'list.length.string', 'list.get.string', 'list.reverse.string'],

    # === LIST[REAL] FUNCTIONS ===
    'list.length.real': ['list'],
    'list.get.real': ['list', 'list.length.real'],
    'list.append.real': ['list'],
    'list.count.real': ['list'],
    'list.contains.real': ['list', 'list.count.real'],
    'list.reverse.real': ['list'],
    'list.slice.real': ['list', 'list.length.real', 'list.get.real', 'list.reverse.real'],

    # === LIST[BOOL] FUNCTIONS ===
    'list.length.bool': ['list'],
    'list.get.bool': ['list', 'list.length.bool'],
    'list.append.bool': ['list'],
    'list.count.bool': ['list'],
    'list.contains.bool': ['list', 'list.count.bool'],
    'list.reverse.bool': ['list'],
    'list.slice.bool': ['list', 'list.length.bool', 'list.get.bool', 'list.reverse.bool'],

    # === LIST[LIST[INT]] FUNCTIONS ===
    'list.length.list_int': ['list'],
    'list.get.list_int': ['list', 'list.length.list_int'],
    'list.append.list_int': ['list'],
    'list.count.list_int': ['list'],
    'list.contains.list_int': ['list', 'list.count.list_int'],
    'list.reverse.list_int': ['list'],
    'list.slice.list_int': ['list', 'list.length.list_int', 'list.get.list_int', 'list.reverse.list_int'],

    # === LIST[LIST[LIST[INT]]] FUNCTIONS ===
    'list.length.list_list_int': ['list'],
    'list.get.list_list_int': ['list', 'list.length.list_list_int'],
    'list.append.list_list_int': ['list'],
    'list.count.list_list_int': ['list'],
    'list.contains.list_list_int': ['list', 'list.count.list_list_int'],
    'list.reverse.list_list_int': ['list'],
    'list.slice.list_list_int': ['list', 'list.length.list_list_int', 'list.get.list_list_int', 'list.reverse.list_list_int'],

    # === LIST[LIST[REAL]] FUNCTIONS ===
    'list.length.list_real': ['list'],
    'list.get.list_real': ['list', 'list.length.list_real'],
    'list.append.list_real': ['list'],
    'list.count.list_real': ['list'],
    'list.contains.list_real': ['list', 'list.count.list_real'],
    'list.reverse.list_real': ['list'],
    'list.slice.list_real': ['list', 'list.length.list_real', 'list.get.list_real', 'list.reverse.list_real'],
}

# Library definitions: maps key -> SMT-LIB2 code
library = {

# =============================================================================
# DATATYPE
# =============================================================================

'list':
"""
(declare-datatypes ((List 1))
    ((par (T) ((cons (head T) (tail (List T))) (nil)))))
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

# =============================================================================
# LIST[INT] FUNCTIONS
# =============================================================================

'list.length.int':
"""
(define-fun-rec list.length.int ((l (List Int))) Int
  (ite (= l (as nil (List Int)))
       0
       (+ 1 (list.length.int (tail l)))))
""",

'list.get.int':
"""
(define-fun-rec list.get.int ((l (List Int)) (idx Int)) Int
  (ite (< idx 0)
       (list.get.int l (+ (list.length.int l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.int (tail l) (- idx 1)))))
""",

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

'list.append.int':
"""
(define-fun-rec list.append.int ((l1 (List Int)) (l2 (List Int))) (List Int)
  (ite (= l1 (as nil (List Int)))
       l2
       (cons (head l1) (list.append.int (tail l1) l2))))
""",

'list.map_add.int':
"""
(define-fun-rec list.map_add.int ((l (List Int)) (val Int)) (List Int)
  (ite (= l (as nil (List Int)))
       (as nil (List Int))
       (cons (+ (head l) val) (list.map_add.int (tail l) val))))
""",

'list.count.int':
"""
(define-fun-rec list.count.int ((l (List Int)) (val Int)) Int
  (ite (= l (as nil (List Int)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.int (tail l) val))))
""",

'list.contains.int':
"""
(define-fun list.contains.int ((l (List Int)) (val Int)) Bool
  (> (list.count.int l val) 0))
""",

'list.reverse.int':
"""
(define-fun-rec list.reverse.int ((l (List Int)) (acc (List Int))) (List Int)
  (ite (= l (as nil (List Int)))
       acc
       (list.reverse.int (tail l) (cons (head l) acc))))
""",

'list.slice.int':
"""
(define-fun list.adjust_index ((idx Int) (len Int)) Int
  (ite (< idx 0)
       (+ len idx)
       idx))

(define-fun list.valid_index.int ((l (List Int)) (idx Int)) Bool
  (and (>= idx 0) (< idx (list.length.int l))))

(define-fun-rec list.slice.int.helper ((l (List Int)) (curr Int) (stop Int) (step Int) (result (List Int))) (List Int)
  (ite (or (and (> step 0) (>= curr stop))
           (and (< step 0) (<= curr stop))
           (not (list.valid_index.int l curr)))
       result
       (let ((new_result (cons (list.get.int l curr) result)))
         (list.slice.int.helper l (+ curr step) stop step new_result))))

(define-fun list.slice.int ((l (List Int)) (start Int) (stop Int) (step Int)) (List Int)
  (let ((len (list.length.int l)))
    (ite (= step 0)
         (as nil (List Int))
         (let ((adj_start (list.adjust_index start len))
               (adj_stop (list.adjust_index stop len)))
           (ite (> step 0)
                (list.reverse.int
                  (list.slice.int.helper l adj_start adj_stop step (as nil (List Int)))
                  (as nil (List Int)))
                (let ((real_start (- len 1 adj_start))
                      (real_stop (- len 1 adj_stop)))
                  (list.reverse.int
                    (list.slice.int.helper l real_start real_stop (ite (< step 0) (- 0 step) step) (as nil (List Int)))
                    (as nil (List Int)))))))))
""",

# =============================================================================
# LIST[STRING] FUNCTIONS
# =============================================================================

'list.length.string':
"""
(define-fun-rec list.length.string ((l (List String))) Int
  (ite (= l (as nil (List String)))
       0
       (+ 1 (list.length.string (tail l)))))
""",

'list.get.string':
"""
(define-fun-rec list.get.string ((l (List String)) (idx Int)) String
  (ite (< idx 0)
       (list.get.string l (+ (list.length.string l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.string (tail l) (- idx 1)))))
""",

'list.append.string':
"""
(define-fun-rec list.append.string ((l1 (List String)) (l2 (List String))) (List String)
  (ite (= l1 (as nil (List String)))
       l2
       (cons (head l1) (list.append.string (tail l1) l2))))
""",

'list.count.string':
"""
(define-fun-rec list.count.string ((l (List String)) (val String)) Int
  (ite (= l (as nil (List String)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.string (tail l) val))))
""",

'list.contains.string':
"""
(define-fun list.contains.string ((l (List String)) (val String)) Bool
  (> (list.count.string l val) 0))
""",

'list.reverse.string':
"""
(define-fun-rec list.reverse.string ((l (List String)) (acc (List String))) (List String)
  (ite (= l (as nil (List String)))
       acc
       (list.reverse.string (tail l) (cons (head l) acc))))
""",

'list.slice.string':
"""
(define-fun list.valid_index.string ((l (List String)) (idx Int)) Bool
  (and (>= idx 0) (< idx (list.length.string l))))

(define-fun-rec list.slice.string.helper ((l (List String)) (curr Int) (stop Int) (step Int) (result (List String))) (List String)
  (ite (or (and (> step 0) (>= curr stop))
           (and (< step 0) (<= curr stop))
           (not (list.valid_index.string l curr)))
       result
       (let ((new_result (cons (list.get.string l curr) result)))
         (list.slice.string.helper l (+ curr step) stop step new_result))))

(define-fun list.slice.string ((l (List String)) (start Int) (stop Int) (step Int)) (List String)
  (let ((len (list.length.string l)))
    (ite (= step 0)
         (as nil (List String))
         (let ((adj_start (list.adjust_index start len))
               (adj_stop (list.adjust_index stop len)))
           (ite (> step 0)
                (list.reverse.string
                  (list.slice.string.helper l adj_start adj_stop step (as nil (List String)))
                  (as nil (List String)))
                (let ((real_start (- len 1 adj_start))
                      (real_stop (- len 1 adj_stop)))
                  (list.reverse.string
                    (list.slice.string.helper l real_start real_stop (ite (< step 0) (- 0 step) step) (as nil (List String)))
                    (as nil (List String)))))))))
""",

# =============================================================================
# LIST[REAL] FUNCTIONS
# =============================================================================

'list.length.real':
"""
(define-fun-rec list.length.real ((l (List Real))) Int
  (ite (= l (as nil (List Real)))
       0
       (+ 1 (list.length.real (tail l)))))
""",

'list.get.real':
"""
(define-fun-rec list.get.real ((l (List Real)) (idx Int)) Real
  (ite (< idx 0)
       (list.get.real l (+ (list.length.real l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.real (tail l) (- idx 1)))))
""",

'list.append.real':
"""
(define-fun-rec list.append.real ((l1 (List Real)) (l2 (List Real))) (List Real)
  (ite (= l1 (as nil (List Real)))
       l2
       (cons (head l1) (list.append.real (tail l1) l2))))
""",

'list.count.real':
"""
(define-fun-rec list.count.real ((l (List Real)) (val Real)) Int
  (ite (= l (as nil (List Real)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.real (tail l) val))))
""",

'list.contains.real':
"""
(define-fun list.contains.real ((l (List Real)) (val Real)) Bool
  (> (list.count.real l val) 0))
""",

'list.reverse.real':
"""
(define-fun-rec list.reverse.real ((l (List Real)) (acc (List Real))) (List Real)
  (ite (= l (as nil (List Real)))
       acc
       (list.reverse.real (tail l) (cons (head l) acc))))
""",

'list.slice.real':
"""
(define-fun list.valid_index.real ((l (List Real)) (idx Int)) Bool
  (and (>= idx 0) (< idx (list.length.real l))))

(define-fun-rec list.slice.real.helper ((l (List Real)) (curr Int) (stop Int) (step Int) (result (List Real))) (List Real)
  (ite (or (and (> step 0) (>= curr stop))
           (and (< step 0) (<= curr stop))
           (not (list.valid_index.real l curr)))
       result
       (let ((new_result (cons (list.get.real l curr) result)))
         (list.slice.real.helper l (+ curr step) stop step new_result))))

(define-fun list.slice.real ((l (List Real)) (start Int) (stop Int) (step Int)) (List Real)
  (let ((len (list.length.real l)))
    (ite (= step 0)
         (as nil (List Real))
         (let ((adj_start (list.adjust_index start len))
               (adj_stop (list.adjust_index stop len)))
           (ite (> step 0)
                (list.reverse.real
                  (list.slice.real.helper l adj_start adj_stop step (as nil (List Real)))
                  (as nil (List Real)))
                (let ((real_start (- len 1 adj_start))
                      (real_stop (- len 1 adj_stop)))
                  (list.reverse.real
                    (list.slice.real.helper l real_start real_stop (ite (< step 0) (- 0 step) step) (as nil (List Real)))
                    (as nil (List Real)))))))))
""",

# =============================================================================
# LIST[BOOL] FUNCTIONS
# =============================================================================

'list.length.bool':
"""
(define-fun-rec list.length.bool ((l (List Bool))) Int
  (ite (= l (as nil (List Bool)))
       0
       (+ 1 (list.length.bool (tail l)))))
""",

'list.get.bool':
"""
(define-fun-rec list.get.bool ((l (List Bool)) (idx Int)) Bool
  (ite (< idx 0)
       (list.get.bool l (+ (list.length.bool l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.bool (tail l) (- idx 1)))))
""",

'list.append.bool':
"""
(define-fun-rec list.append.bool ((l1 (List Bool)) (l2 (List Bool))) (List Bool)
  (ite (= l1 (as nil (List Bool)))
       l2
       (cons (head l1) (list.append.bool (tail l1) l2))))
""",

'list.count.bool':
"""
(define-fun-rec list.count.bool ((l (List Bool)) (val Bool)) Int
  (ite (= l (as nil (List Bool)))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.bool (tail l) val))))
""",

'list.contains.bool':
"""
(define-fun list.contains.bool ((l (List Bool)) (val Bool)) Bool
  (> (list.count.bool l val) 0))
""",

'list.reverse.bool':
"""
(define-fun-rec list.reverse.bool ((l (List Bool)) (acc (List Bool))) (List Bool)
  (ite (= l (as nil (List Bool)))
       acc
       (list.reverse.bool (tail l) (cons (head l) acc))))
""",

'list.slice.bool':
"""
(define-fun list.valid_index.bool ((l (List Bool)) (idx Int)) Bool
  (and (>= idx 0) (< idx (list.length.bool l))))

(define-fun-rec list.slice.bool.helper ((l (List Bool)) (curr Int) (stop Int) (step Int) (result (List Bool))) (List Bool)
  (ite (or (and (> step 0) (>= curr stop))
           (and (< step 0) (<= curr stop))
           (not (list.valid_index.bool l curr)))
       result
       (let ((new_result (cons (list.get.bool l curr) result)))
         (list.slice.bool.helper l (+ curr step) stop step new_result))))

(define-fun list.slice.bool ((l (List Bool)) (start Int) (stop Int) (step Int)) (List Bool)
  (let ((len (list.length.bool l)))
    (ite (= step 0)
         (as nil (List Bool))
         (let ((adj_start (list.adjust_index start len))
               (adj_stop (list.adjust_index stop len)))
           (ite (> step 0)
                (list.reverse.bool
                  (list.slice.bool.helper l adj_start adj_stop step (as nil (List Bool)))
                  (as nil (List Bool)))
                (let ((real_start (- len 1 adj_start))
                      (real_stop (- len 1 adj_stop)))
                  (list.reverse.bool
                    (list.slice.bool.helper l real_start real_stop (ite (< step 0) (- 0 step) step) (as nil (List Bool)))
                    (as nil (List Bool)))))))))
""",

# =============================================================================
# LIST[LIST[INT]] FUNCTIONS
# =============================================================================

'list.length.list_int':
"""
(define-fun-rec list.length.list_int ((l (List (List Int)))) Int
  (ite (= l (as nil (List (List Int))))
       0
       (+ 1 (list.length.list_int (tail l)))))
""",

'list.get.list_int':
"""
(define-fun-rec list.get.list_int ((l (List (List Int))) (idx Int)) (List Int)
  (ite (< idx 0)
       (list.get.list_int l (+ (list.length.list_int l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.list_int (tail l) (- idx 1)))))
""",

'list.append.list_int':
"""
(define-fun-rec list.append.list_int ((l1 (List (List Int))) (l2 (List (List Int)))) (List (List Int))
  (ite (= l1 (as nil (List (List Int))))
       l2
       (cons (head l1) (list.append.list_int (tail l1) l2))))
""",

'list.count.list_int':
"""
(define-fun-rec list.count.list_int ((l (List (List Int))) (val (List Int))) Int
  (ite (= l (as nil (List (List Int))))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.list_int (tail l) val))))
""",

'list.contains.list_int':
"""
(define-fun list.contains.list_int ((l (List (List Int))) (val (List Int))) Bool
  (> (list.count.list_int l val) 0))
""",

'list.reverse.list_int':
"""
(define-fun-rec list.reverse.list_int ((l (List (List Int))) (acc (List (List Int)))) (List (List Int))
  (ite (= l (as nil (List (List Int))))
       acc
       (list.reverse.list_int (tail l) (cons (head l) acc))))
""",

'list.slice.list_int':
"""
(define-fun list.valid_index.list_int ((l (List (List Int))) (idx Int)) Bool
  (and (>= idx 0) (< idx (list.length.list_int l))))

(define-fun-rec list.slice.list_int.helper ((l (List (List Int))) (curr Int) (stop Int) (step Int) (result (List (List Int)))) (List (List Int))
  (ite (or (and (> step 0) (>= curr stop))
           (and (< step 0) (<= curr stop))
           (not (list.valid_index.list_int l curr)))
       result
       (let ((new_result (cons (list.get.list_int l curr) result)))
         (list.slice.list_int.helper l (+ curr step) stop step new_result))))

(define-fun list.slice.list_int ((l (List (List Int))) (start Int) (stop Int) (step Int)) (List (List Int))
  (let ((len (list.length.list_int l)))
    (ite (= step 0)
         (as nil (List (List Int)))
         (let ((adj_start (list.adjust_index start len))
               (adj_stop (list.adjust_index stop len)))
           (ite (> step 0)
                (list.reverse.list_int
                  (list.slice.list_int.helper l adj_start adj_stop step (as nil (List (List Int))))
                  (as nil (List (List Int))))
                (let ((real_start (- len 1 adj_start))
                      (real_stop (- len 1 adj_stop)))
                  (list.reverse.list_int
                    (list.slice.list_int.helper l real_start real_stop (ite (< step 0) (- 0 step) step) (as nil (List (List Int))))
                    (as nil (List (List Int))))))))))
""",

# =============================================================================
# LIST[LIST[LIST[INT]]] FUNCTIONS
# =============================================================================

'list.length.list_list_int':
"""
(define-fun-rec list.length.list_list_int ((l (List (List (List Int))))) Int
  (ite (= l (as nil (List (List (List Int)))))
       0
       (+ 1 (list.length.list_list_int (tail l)))))
""",

'list.get.list_list_int':
"""
(define-fun-rec list.get.list_list_int ((l (List (List (List Int)))) (idx Int)) (List (List Int))
  (ite (< idx 0)
       (list.get.list_list_int l (+ (list.length.list_list_int l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.list_list_int (tail l) (- idx 1)))))
""",

'list.append.list_list_int':
"""
(define-fun-rec list.append.list_list_int ((l1 (List (List (List Int)))) (l2 (List (List (List Int))))) (List (List (List Int)))
  (ite (= l1 (as nil (List (List (List Int)))))
       l2
       (cons (head l1) (list.append.list_list_int (tail l1) l2))))
""",

'list.count.list_list_int':
"""
(define-fun-rec list.count.list_list_int ((l (List (List (List Int)))) (val (List (List Int)))) Int
  (ite (= l (as nil (List (List (List Int)))))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.list_list_int (tail l) val))))
""",

'list.contains.list_list_int':
"""
(define-fun list.contains.list_list_int ((l (List (List (List Int)))) (val (List (List Int)))) Bool
  (> (list.count.list_list_int l val) 0))
""",

'list.reverse.list_list_int':
"""
(define-fun-rec list.reverse.list_list_int ((l (List (List (List Int)))) (acc (List (List (List Int))))) (List (List (List Int)))
  (ite (= l (as nil (List (List (List Int)))))
       acc
       (list.reverse.list_list_int (tail l) (cons (head l) acc))))
""",

'list.slice.list_list_int':
"""
(define-fun list.valid_index.list_list_int ((l (List (List (List Int)))) (idx Int)) Bool
  (and (>= idx 0) (< idx (list.length.list_list_int l))))

(define-fun-rec list.slice.list_list_int.helper ((l (List (List (List Int)))) (curr Int) (stop Int) (step Int) (result (List (List (List Int))))) (List (List (List Int)))
  (ite (or (and (> step 0) (>= curr stop))
           (and (< step 0) (<= curr stop))
           (not (list.valid_index.list_list_int l curr)))
       result
       (let ((new_result (cons (list.get.list_list_int l curr) result)))
         (list.slice.list_list_int.helper l (+ curr step) stop step new_result))))

(define-fun list.slice.list_list_int ((l (List (List (List Int)))) (start Int) (stop Int) (step Int)) (List (List (List Int)))
  (let ((len (list.length.list_list_int l)))
    (ite (= step 0)
         (as nil (List (List (List Int))))
         (let ((adj_start (list.adjust_index start len))
               (adj_stop (list.adjust_index stop len)))
           (ite (> step 0)
                (list.reverse.list_list_int
                  (list.slice.list_list_int.helper l adj_start adj_stop step (as nil (List (List (List Int)))))
                  (as nil (List (List (List Int)))))
                (let ((real_start (- len 1 adj_start))
                      (real_stop (- len 1 adj_stop)))
                  (list.reverse.list_list_int
                    (list.slice.list_list_int.helper l real_start real_stop (ite (< step 0) (- 0 step) step) (as nil (List (List (List Int)))))
                    (as nil (List (List (List Int)))))))))))
""",

# =============================================================================
# LIST[LIST[REAL]] FUNCTIONS
# =============================================================================

'list.length.list_real':
"""
(define-fun-rec list.length.list_real ((l (List (List Real)))) Int
  (ite (= l (as nil (List (List Real))))
       0
       (+ 1 (list.length.list_real (tail l)))))
""",

'list.get.list_real':
"""
(define-fun-rec list.get.list_real ((l (List (List Real))) (idx Int)) (List Real)
  (ite (< idx 0)
       (list.get.list_real l (+ (list.length.list_real l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.list_real (tail l) (- idx 1)))))
""",

'list.append.list_real':
"""
(define-fun-rec list.append.list_real ((l1 (List (List Real))) (l2 (List (List Real)))) (List (List Real))
  (ite (= l1 (as nil (List (List Real))))
       l2
       (cons (head l1) (list.append.list_real (tail l1) l2))))
""",

'list.count.list_real':
"""
(define-fun-rec list.count.list_real ((l (List (List Real))) (val (List Real))) Int
  (ite (= l (as nil (List (List Real))))
       0
       (+ (ite (= (head l) val) 1 0)
          (list.count.list_real (tail l) val))))
""",

'list.contains.list_real':
"""
(define-fun list.contains.list_real ((l (List (List Real))) (val (List Real))) Bool
  (> (list.count.list_real l val) 0))
""",

'list.reverse.list_real':
"""
(define-fun-rec list.reverse.list_real ((l (List (List Real))) (acc (List (List Real)))) (List (List Real))
  (ite (= l (as nil (List (List Real))))
       acc
       (list.reverse.list_real (tail l) (cons (head l) acc))))
""",

'list.slice.list_real':
"""
(define-fun list.valid_index.list_real ((l (List (List Real))) (idx Int)) Bool
  (and (>= idx 0) (< idx (list.length.list_real l))))

(define-fun-rec list.slice.list_real.helper ((l (List (List Real))) (curr Int) (stop Int) (step Int) (result (List (List Real)))) (List (List Real))
  (ite (or (and (> step 0) (>= curr stop))
           (and (< step 0) (<= curr stop))
           (not (list.valid_index.list_real l curr)))
       result
       (let ((new_result (cons (list.get.list_real l curr) result)))
         (list.slice.list_real.helper l (+ curr step) stop step new_result))))

(define-fun list.slice.list_real ((l (List (List Real))) (start Int) (stop Int) (step Int)) (List (List Real))
  (let ((len (list.length.list_real l)))
    (ite (= step 0)
         (as nil (List (List Real)))
         (let ((adj_start (list.adjust_index start len))
               (adj_stop (list.adjust_index stop len)))
           (ite (> step 0)
                (list.reverse.list_real
                  (list.slice.list_real.helper l adj_start adj_stop step (as nil (List (List Real))))
                  (as nil (List (List Real))))
                (let ((real_start (- len 1 adj_start))
                      (real_stop (- len 1 adj_stop)))
                  (list.reverse.list_real
                    (list.slice.list_real.helper l real_start real_stop (ite (< step 0) (- 0 step) step) (as nil (List (List Real))))
                    (as nil (List (List Real))))))))))
""",

}


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
