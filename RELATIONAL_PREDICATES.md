# Relational Predicates Experiment

## Goal

Replace recursive SMT-LIB2 function definitions (`define-fun-rec`) in `backend_lib.py`
with relational predicates: `declare-fun` (uninterpreted function) + universally quantified
axioms (`forall` with `:pattern` annotations for E-matching).

The hypothesis was that Z3's E-matching engine and CVC5's quantifier instantiation
would handle axiom-based definitions more efficiently than recursive function unfolding,
especially for search-heavy list operations like `count`, `contains`, `index`, and `sum`.

## Implementation

Gated behind `RELATIONAL_PREDICATES=true` environment variable.

**File:** `holey/backend_lib_relational.py`

Recursive version (`define-fun-rec`):
```smt2
(define-fun-rec list.length.int ((l (List Int))) Int
  (ite (= l (as nil (List Int)))
       0
       (+ 1 (list.length.int (tail l)))))
```

Relational version (`declare-fun` + `forall`):
```smt2
(declare-fun list.length.int ((List Int)) Int)

; Base case
(assert (= (list.length.int (as nil (List Int))) 0))

; Inductive case (cons pattern)
(assert (forall ((h Int) (t (List Int)))
  (! (= (list.length.int (cons h t)) (+ 1 (list.length.int t)))
     :pattern ((list.length.int (cons h t))))))

; Non-negativity
(assert (forall ((l (List Int)))
  (! (>= (list.length.int l) 0)
     :pattern ((list.length.int l)))))
```

Functions converted: `list.length`, `list.get`, `list.append`, `list.count`,
`list.contains`, `list.index`, `list.sum` (for all element types).

Functions kept as `define-fun-rec`: `list.reverse`, `list.slice`, `list.set_len`,
`list.map_add`, and all string functions.

## Results

163 regressions out of 206 solved puzzles. 0 improvements. Both Z3 and CVC5
return `unknown` (timeout) on problems that the recursive encoding solves.

## Why It Fails

The root cause is a **pattern trigger problem** inherent to combining quantified
axioms with algebraic datatypes (ADTs).

### The chicken-and-egg problem

A typical puzzle constraint looks like:
```smt2
(declare-const x (List Int))
(assert (= (list.get.int x 0) 10))
```

The cons-pattern axiom for `list.get` is:
```smt2
(assert (forall ((h Int) (t (List Int)))
  (! (= (list.get.int (cons h t) 0) h)
     :pattern ((list.get.int (cons h t) 0)))))
```

This pattern only fires when a term of the form `(list.get.int (cons h t) 0)` exists
in the E-matching context. But the constraint uses `(list.get.int x 0)` where `x` is
an opaque variable — not a `(cons h t)` term. The pattern never fires.

With `define-fun-rec`, the solver can eagerly unfold:
`list.get.int(x, 0)` -> `ite(0=0, head(x), ...)` -> `head(x)`. No pattern matching needed.

### Accessor-form axioms don't help either

We tried rewriting axioms to use `head`/`tail` instead of cons patterns:
```smt2
(assert (forall ((l (List Int)))
  (! (=> (not (= l (as nil (List Int))))
         (= (list.get.int l 0) (head l)))
     :pattern ((list.get.int l 0)))))
```

This fires correctly on `(list.get.int x 0)` and produces `(head x) = 10`. But the
step case creates an infinite instantiation loop:

- `(list.get.int x 1)` fires the step axiom, creating `(list.get.int (tail x) (- 1 1))`
- The term `(- 1 1)` is an **expression**, not the literal `0`, so the base case
  pattern `(list.get.int l 0)` doesn't match
- The step axiom fires again on the new term, creating deeper `tail` nesting
- This continues until the solver times out

For `list.length`, the accessor form is even worse:
- `(list.length.int x)` fires the axiom, creating `(list.length.int (tail x))`
- Which fires again, creating `(list.length.int (tail (tail x)))`
- Infinite chain with no decreasing argument to stop it

Multi-patterns like `((list.length.int l) (tail l))` prevent the infinite loop but
are too restrictive — they only fire when `(tail l)` already exists from another source,
which rarely happens.

### The fundamental limitation

SMT solvers' quantifier instantiation engines (E-matching, MBQI) are not designed
for recursive reasoning over algebraic datatypes. `define-fun-rec` gives the solver
structural information (totality, termination, computational unfolding) that
`declare-fun` + axioms cannot express. CVC5's `--fmf-fun` mode specifically
optimizes for `define-fun-rec` with finite model finding — it essentially does
the relational approach internally but with more information.

## Confirmed via minimal test

```smt2
; This times out on both Z3 and CVC5:
(declare-fun my.get ((List Int) Int) Int)
(assert (forall ((l (List Int)))
  (=> (not (= l (as nil (List Int)))) (= (my.get l 0) (head l)))))
(assert (forall ((l (List Int)) (i Int))
  (=> (and (not (= l (as nil (List Int)))) (> i 0))
      (= (my.get l i) (my.get (tail l) (- i 1))))))
(declare-const x (List Int))
(assert (= (my.get x 0) 10))
(check-sat)  ; → timeout

; While this solves instantly:
(declare-const x (List Int))
(assert (= (head x) 10))
(check-sat)  ; → sat
```

## Conclusion

Relational predicates over ADTs are theoretically sound but practically unusable
with current SMT solvers (Z3 4.x, CVC5). The recursive `define-fun-rec` encoding
remains the right choice for this problem domain.
