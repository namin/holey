"""
CHC (Constrained Horn Clauses) versions of SMT-LIB2 library definitions.

Uses declare-fun + Horn clause axioms designed for Z3's Spacer engine.
The key difference from relational predicates: axioms are structured as
proper Horn clauses (implication with single head) without :pattern
annotations, relying on Spacer's PDR algorithm instead of E-matching.

Gated behind CHC=true environment variable.

Functions not converted to CHC form are re-exported from backend_lib.
"""

from .backend_lib import (
    ELEMENT_TYPES,
    library_static,
    library_deps_static,
    make_list_reverse,
    make_list_slice,
    make_list_set_len,
)

# =============================================================================
# CHC TEMPLATES for List operations
# =============================================================================

def make_list_length_chc(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    return f'''
(declare-fun list.length.{suffix} ((List {elem_type})) Int)
; Base case
(assert (= (list.length.{suffix} {nil_expr}) 0))
; Step case (Horn clause: premise => head)
(assert (forall ((h {elem_type}) (t {list_type}))
  (= (list.length.{suffix} (cons h t)) (+ 1 (list.length.{suffix} t)))))
; Non-negativity
(assert (forall ((l {list_type}))
  (>= (list.length.{suffix} l) 0)))
'''

def make_list_get_chc(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    return f'''
(declare-fun list.get.{suffix} ((List {elem_type}) Int) {elem_type})
; Base case: index 0
(assert (forall ((h {elem_type}) (t {list_type}))
  (= (list.get.{suffix} (cons h t) 0) h)))
; Step case: index > 0
(assert (forall ((h {elem_type}) (t {list_type}) (i Int))
  (=> (> i 0)
      (= (list.get.{suffix} (cons h t) i) (list.get.{suffix} t (- i 1))))))
; Negative index
(assert (forall ((l {list_type}) (i Int))
  (=> (< i 0)
      (= (list.get.{suffix} l i) (list.get.{suffix} l (+ (list.length.{suffix} l) i))))))
'''

def make_list_append_chc(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    return f'''
(declare-fun list.append.{suffix} ((List {elem_type}) (List {elem_type})) (List {elem_type}))
; Base case
(assert (forall ((l2 {list_type}))
  (= (list.append.{suffix} {nil_expr} l2) l2)))
; Step case
(assert (forall ((h {elem_type}) (t {list_type}) (l2 {list_type}))
  (= (list.append.{suffix} (cons h t) l2) (cons h (list.append.{suffix} t l2)))))
'''

def make_list_count_chc(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    return f'''
(declare-fun list.count.{suffix} ((List {elem_type}) {elem_type}) Int)
; Base case
(assert (forall ((val {elem_type}))
  (= (list.count.{suffix} {nil_expr} val) 0)))
; Step case
(assert (forall ((h {elem_type}) (t {list_type}) (val {elem_type}))
  (= (list.count.{suffix} (cons h t) val)
     (+ (ite (= h val) 1 0) (list.count.{suffix} t val)))))
; Non-negativity
(assert (forall ((l {list_type}) (val {elem_type}))
  (>= (list.count.{suffix} l val) 0)))
'''

def make_list_contains_chc(suffix, elem_type, nil_expr):
    """Same as original — defined via count."""
    list_type = f'(List {elem_type})'
    return f'''
(define-fun list.contains.{suffix} ((l {list_type}) (val {elem_type})) Bool
  (> (list.count.{suffix} l val) 0))
'''

def make_list_index_chc(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    return f'''
(declare-fun list.index.rec.{suffix} (Int (List {elem_type}) {elem_type}) Int)
; Base case
(assert (forall ((i Int) (val {elem_type}))
  (= (list.index.rec.{suffix} i {nil_expr} val) (- 1))))
; Step case
(assert (forall ((i Int) (h {elem_type}) (t {list_type}) (val {elem_type}))
  (= (list.index.rec.{suffix} i (cons h t) val)
     (ite (= h val) i (list.index.rec.{suffix} (+ 1 i) t val)))))

(define-fun list.index.{suffix} ((l {list_type}) (val {elem_type})) Int
  (list.index.rec.{suffix} 0 l val))
'''

def make_list_sum_chc(suffix, elem_type, nil_expr, zero_val):
    list_type = f'(List {elem_type})'
    return f'''
(declare-fun list.sum.{suffix} ((List {elem_type})) {elem_type})
; Base case
(assert (= (list.sum.{suffix} {nil_expr}) {zero_val}))
; Step case
(assert (forall ((h {elem_type}) (t {list_type}))
  (= (list.sum.{suffix} (cons h t)) (+ h (list.sum.{suffix} t)))))
'''

# =============================================================================
# GENERATE CHC list operations for all element types
# =============================================================================

def generate_list_library_chc():
    lib = {}
    deps = {}

    for suffix, (elem_type, nil_expr, is_numeric) in ELEMENT_TYPES.items():
        lib[f'list.length.{suffix}'] = make_list_length_chc(suffix, elem_type, nil_expr)
        deps[f'list.length.{suffix}'] = ['list']

        lib[f'list.get.{suffix}'] = make_list_get_chc(suffix, elem_type, nil_expr)
        deps[f'list.get.{suffix}'] = ['list', f'list.length.{suffix}']

        lib[f'list.append.{suffix}'] = make_list_append_chc(suffix, elem_type, nil_expr)
        deps[f'list.append.{suffix}'] = ['list']

        lib[f'list.count.{suffix}'] = make_list_count_chc(suffix, elem_type, nil_expr)
        deps[f'list.count.{suffix}'] = ['list']

        lib[f'list.contains.{suffix}'] = make_list_contains_chc(suffix, elem_type, nil_expr)
        deps[f'list.contains.{suffix}'] = ['list', f'list.count.{suffix}']

        lib[f'list.index.{suffix}'] = make_list_index_chc(suffix, elem_type, nil_expr)
        deps[f'list.index.{suffix}'] = ['list']

        # Kept as recursive
        lib[f'list.set_len.{suffix}'] = make_list_set_len(suffix, elem_type, nil_expr)
        deps[f'list.set_len.{suffix}'] = ['list', f'list.contains.{suffix}']

        lib[f'list.reverse.{suffix}'] = make_list_reverse(suffix, elem_type, nil_expr)
        deps[f'list.reverse.{suffix}'] = ['list']

        lib[f'list.slice.{suffix}'] = make_list_slice(suffix, elem_type, nil_expr)
        deps[f'list.slice.{suffix}'] = ['list', 'list.adjust_index', f'list.length.{suffix}', f'list.get.{suffix}', f'list.reverse.{suffix}']

        if is_numeric:
            zero_val = '0' if elem_type == 'Int' else '0.0'
            lib[f'list.sum.{suffix}'] = make_list_sum_chc(suffix, elem_type, nil_expr, zero_val)
            deps[f'list.sum.{suffix}'] = ['list']

    return lib, deps


# =============================================================================
# BUILD FINAL LIBRARY
# =============================================================================

_list_lib_chc, _list_deps_chc = generate_list_library_chc()

library = {**library_static, **_list_lib_chc}
library_deps = {**library_deps_static, **_list_deps_chc}


# =============================================================================
# UTILITY FUNCTIONS (re-exported with same interface)
# =============================================================================

def resolve_dependencies(keys):
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

    return order


def emit_library(keys):
    resolved = resolve_dependencies(keys)
    return '\n'.join(library[k] for k in resolved if k in library)
