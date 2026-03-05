"""
Relational predicate versions of SMT-LIB2 library definitions.

Instead of define-fun-rec (recursive functions), this module uses
declare-fun + universally quantified axioms (relational predicates).
This can improve SMT solver performance via Z3's E-matching engine.

Gated behind RELATIONAL_PREDICATES=true environment variable.

Functions not converted to relational form are re-exported from backend_lib.
"""

from .backend_lib import (
    ELEMENT_TYPES,
    library_static,
    library_deps_static,
)

# =============================================================================
# RELATIONAL TEMPLATES for List operations
# =============================================================================

def _sort_list(*sorts):
    """Build SMT-LIB2 sort list for declare-fun: (Sort1 Sort2 ...)"""
    return '(' + ' '.join(sorts) + ')'

def make_list_length_rel(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    sorts = _sort_list(list_type)
    return f'''
(declare-fun list.length.{suffix} {sorts} Int)
(assert (= (list.length.{suffix} {nil_expr}) 0))
(assert (forall ((h {elem_type}) (t {list_type}))
  (! (= (list.length.{suffix} (cons h t)) (+ 1 (list.length.{suffix} t)))
     :pattern ((list.length.{suffix} (cons h t))))))
(assert (forall ((l {list_type}))
  (! (>= (list.length.{suffix} l) 0)
     :pattern ((list.length.{suffix} l)))))
'''

def make_list_get_rel(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    sorts = _sort_list(list_type, 'Int')
    return f'''
(declare-fun list.get.{suffix} {sorts} {elem_type})
(assert (forall ((h {elem_type}) (t {list_type}))
  (! (= (list.get.{suffix} (cons h t) 0) h)
     :pattern ((list.get.{suffix} (cons h t) 0)))))
(assert (forall ((h {elem_type}) (t {list_type}) (i Int))
  (! (=> (> i 0) (= (list.get.{suffix} (cons h t) i) (list.get.{suffix} t (- i 1))))
     :pattern ((list.get.{suffix} (cons h t) i)))))
(assert (forall ((l {list_type}) (i Int))
  (! (=> (< i 0) (= (list.get.{suffix} l i) (list.get.{suffix} l (+ (list.length.{suffix} l) i))))
     :pattern ((list.get.{suffix} l i)))))
'''

def make_list_append_rel(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    sorts = _sort_list(list_type, list_type)
    return f'''
(declare-fun list.append.{suffix} {sorts} {list_type})
(assert (forall ((l2 {list_type}))
  (! (= (list.append.{suffix} {nil_expr} l2) l2)
     :pattern ((list.append.{suffix} {nil_expr} l2)))))
(assert (forall ((h {elem_type}) (t {list_type}) (l2 {list_type}))
  (! (= (list.append.{suffix} (cons h t) l2) (cons h (list.append.{suffix} t l2)))
     :pattern ((list.append.{suffix} (cons h t) l2)))))
'''

def make_list_count_rel(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    sorts = _sort_list(list_type, elem_type)
    return f'''
(declare-fun list.count.{suffix} {sorts} Int)
(assert (forall ((val {elem_type}))
  (! (= (list.count.{suffix} {nil_expr} val) 0)
     :pattern ((list.count.{suffix} {nil_expr} val)))))
(assert (forall ((h {elem_type}) (t {list_type}) (val {elem_type}))
  (! (= (list.count.{suffix} (cons h t) val)
        (+ (ite (= h val) 1 0) (list.count.{suffix} t val)))
     :pattern ((list.count.{suffix} (cons h t) val)))))
(assert (forall ((l {list_type}) (val {elem_type}))
  (! (>= (list.count.{suffix} l val) 0)
     :pattern ((list.count.{suffix} l val)))))
'''

def make_list_contains_rel(suffix, elem_type, nil_expr):
    """Same as original — defined via count, which is now relational."""
    list_type = f'(List {elem_type})'
    return f'''
(define-fun list.contains.{suffix} ((l {list_type}) (val {elem_type})) Bool
  (> (list.count.{suffix} l val) 0))
'''

def make_list_index_rel(suffix, elem_type, nil_expr):
    list_type = f'(List {elem_type})'
    sorts = _sort_list('Int', list_type, elem_type)
    return f'''
(declare-fun list.index.rec.{suffix} {sorts} Int)
(assert (forall ((i Int) (val {elem_type}))
  (! (= (list.index.rec.{suffix} i {nil_expr} val) (- 1))
     :pattern ((list.index.rec.{suffix} i {nil_expr} val)))))
(assert (forall ((i Int) (h {elem_type}) (t {list_type}) (val {elem_type}))
  (! (= (list.index.rec.{suffix} i (cons h t) val)
        (ite (= h val) i (list.index.rec.{suffix} (+ 1 i) t val)))
     :pattern ((list.index.rec.{suffix} i (cons h t) val)))))

(define-fun list.index.{suffix} ((l {list_type}) (val {elem_type})) Int
  (list.index.rec.{suffix} 0 l val))
'''

def make_list_sum_rel(suffix, elem_type, nil_expr, zero_val):
    list_type = f'(List {elem_type})'
    sorts = _sort_list(list_type)
    return f'''
(declare-fun list.sum.{suffix} {sorts} {elem_type})
(assert (= (list.sum.{suffix} {nil_expr}) {zero_val}))
(assert (forall ((h {elem_type}) (t {list_type}))
  (! (= (list.sum.{suffix} (cons h t)) (+ h (list.sum.{suffix} t)))
     :pattern ((list.sum.{suffix} (cons h t))))))
'''

# =============================================================================
# GENERATE relational list operations for all element types
# =============================================================================

def generate_list_library_relational():
    """Generate relational library entries and dependencies for all list operations.

    Functions with relational versions: count, contains, index, sum, append.
    These are "heavy" recursive operations where quantified axioms can help the
    solver avoid unbounded recursion during search.

    Functions kept as recursive: length, get, reverse, slice, set_len, map_add.
    These are used with concrete indices/values and benefit from direct evaluation
    via recursive unfolding rather than quantifier instantiation.
    """
    from .backend_lib import (
        make_list_length, make_list_get, make_list_append,
        make_list_reverse, make_list_slice, make_list_set_len,
        make_list_sum,
    )

    lib = {}
    deps = {}

    for suffix, (elem_type, nil_expr, is_numeric) in ELEMENT_TYPES.items():
        # Keep as recursive — used with concrete values, direct evaluation is faster
        lib[f'list.length.{suffix}'] = make_list_length(suffix, elem_type, nil_expr)
        deps[f'list.length.{suffix}'] = ['list']

        lib[f'list.get.{suffix}'] = make_list_get(suffix, elem_type, nil_expr)
        deps[f'list.get.{suffix}'] = ['list', f'list.length.{suffix}']

        lib[f'list.append.{suffix}'] = make_list_append(suffix, elem_type, nil_expr)
        deps[f'list.append.{suffix}'] = ['list']

        lib[f'list.reverse.{suffix}'] = make_list_reverse(suffix, elem_type, nil_expr)
        deps[f'list.reverse.{suffix}'] = ['list']

        lib[f'list.slice.{suffix}'] = make_list_slice(suffix, elem_type, nil_expr)
        deps[f'list.slice.{suffix}'] = ['list', 'list.adjust_index', f'list.length.{suffix}', f'list.get.{suffix}', f'list.reverse.{suffix}']

        lib[f'list.set_len.{suffix}'] = make_list_set_len(suffix, elem_type, nil_expr)
        deps[f'list.set_len.{suffix}'] = ['list', f'list.contains.{suffix}']

        # Relational versions — search-heavy operations where quantified axioms
        # help the solver avoid unbounded recursion
        lib[f'list.count.{suffix}'] = make_list_count_rel(suffix, elem_type, nil_expr)
        deps[f'list.count.{suffix}'] = ['list']

        lib[f'list.contains.{suffix}'] = make_list_contains_rel(suffix, elem_type, nil_expr)
        deps[f'list.contains.{suffix}'] = ['list', f'list.count.{suffix}']

        lib[f'list.index.{suffix}'] = make_list_index_rel(suffix, elem_type, nil_expr)
        deps[f'list.index.{suffix}'] = ['list']

        if is_numeric:
            zero_val = '0' if elem_type == 'Int' else '0.0'
            lib[f'list.sum.{suffix}'] = make_list_sum(suffix, elem_type, nil_expr, zero_val)
            deps[f'list.sum.{suffix}'] = ['list']

    return lib, deps


# =============================================================================
# BUILD FINAL LIBRARY
# =============================================================================

# Generate relational list operations
_list_lib_rel, _list_deps_rel = generate_list_library_relational()

# All static entries (including str.count) stay as original recursive versions
library = {**library_static, **_list_lib_rel}
library_deps = {**library_deps_static, **_list_deps_rel}


# =============================================================================
# UTILITY FUNCTIONS (re-exported with same interface)
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

    return order


def emit_library(keys):
    """
    Emit SMT-LIB2 code for the given library keys, including all dependencies.
    """
    resolved = resolve_dependencies(keys)
    return '\n'.join(library[k] for k in resolved if k in library)
