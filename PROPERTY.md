# Property-Based Testing with Holey

Holey can be used for property-based testing by leveraging its symbolic execution and SMT solving capabilities to find counterexamples that violate properties.

## How It Works

The key insight is the duality between puzzle solving and property testing:

| Mode | Goal | SMT Query |
|------|------|-----------|
| **Puzzle Solving** | Find `x` such that `sat(x) = True` | Solve `sat(x)` |
| **Property Testing** | Find `x` such that `prop(x) = False` | Solve `NOT prop(x)` |

When testing a property, we **negate** it and use Holey to search for inputs where the property fails. If the SMT solver returns:
- **`sat`** → Counterexample found! The property is violated.
- **`unsat`** → No counterexample exists (within the encoding). The property holds.
- **`unknown`/`timeout`** → Inconclusive.

## Writing Properties

Properties are Python functions that return `True` when the property holds:

```python
def prop_add_commutative(x: int, y: int):
    """Addition is commutative."""
    return x + y == y + x

def prop_square_non_negative(x: int):
    """Squares are non-negative."""
    return x * x >= 0
```

### Requirements

1. **Type hints are required** - Each symbolic parameter must have a type annotation
2. **Function name starts with `prop_`** (configurable via `--prop-prefix`)
3. **Return a boolean** - `True` if property holds, `False` if violated

### Supported Types

- `int` - Integers
- `str` - Strings
- `float` - Floating point numbers
- `List[int]` - Lists of integers
- `List[str]` - Lists of strings

### Preconditions via Control Flow

Use early returns to handle preconditions (no special `assume()` needed):

```python
def prop_division_inverse(x: int, y: int):
    """For non-zero y: (x * y) // y == x."""
    if y == 0:
        return True  # Skip division by zero (vacuously true)
    return (x * y) // y == x

def prop_positive_square_greater(x: int):
    """For positive x, x² >= x."""
    if x <= 0:
        return True  # Only test positive numbers
    return x * x >= x
```

### Multiple Parameters

Properties can have multiple symbolic parameters:

```python
def prop_distribute(x: int, y: int, z: int):
    """Multiplication distributes over addition."""
    return x * (y + z) == x * y + x * z
```

### Parameters with Defaults (Concrete Inputs)

Parameters with default values are treated as concrete inputs, not symbolic:

```python
def prop_sum_bounds(li: List[int], n=10):
    """Sum of list elements is bounded."""
    if len(li) != n:
        return True
    return sum(li) <= n * 100
```

Here `li` is symbolic but `n=10` is concrete.

## Running the Property Tester

### Basic Usage

```bash
python property_tester.py --prop-file examples_property/arithmetic.py
```

### Options

```bash
python property_tester.py --help

Options:
  --prop-file FILE        Path to Python file containing properties (required)
  --prop-prefix PREFIX    Function name prefix (default: "prop_")
  --smtlib-backends       SMT solvers to use: z3, cvc5 (default: z3)
  --no-bounded-lists      Disable bounded list optimization
  --bounded-list-max-size Maximum size for bounded lists (default: 200)
  --no-ite                Disable ITE mode (use explicit branching)
  --all-solvers           Run all solvers and collect statistics
```

### Example Output

```
============================================================
Testing prop_add_commutative
============================================================
Property:
def prop_add_commutative(x: int, y: int):
    return x + y == y + x

Negated property:
def sat(x: int, y: int):
    return not x + y == y + x

### smt2
(set-logic ALL)
(declare-const x Int)
(declare-const y Int)
(assert (not (= (+ x y) (+ y x))))
(check-sat)
(get-model)

### output for z3
unsat
  ✓ No counterexample found (property may hold)

============================================================
Testing prop_bad_always_positive
============================================================
Property:
def prop_bad_always_positive(x: int):
    return x > 0

### output for z3
sat
  ✗ COUNTEREXAMPLE FOUND:
    x = 0

============================================================
SUMMARY
============================================================
Total properties tested: 2
  ✗ Counterexamples found: 1
  ✓ No counterexample: 1
```

## Examples

See [`examples_property/arithmetic.py`](examples_property/arithmetic.py) for example properties including:

- Basic arithmetic properties (commutativity, identity, distributivity)
- Properties that should fail (for testing the tester)
- Properties with preconditions

## Comparison with Traditional Property-Based Testing

| Feature | Holey | QuickCheck/Hypothesis |
|---------|-------|----------------------|
| **Search method** | SMT solving (exhaustive within encoding) | Random sampling |
| **Counterexamples** | Guaranteed minimal? No, but precise | Shrinking heuristics |
| **Coverage** | All paths (symbolic) | Statistical |
| **Performance** | Can timeout on complex properties | Fast but may miss bugs |
| **Quantifiers** | Native support (∀, ∃) | Manual generators |

### When to Use Holey for Property Testing

- Properties involving arithmetic constraints
- When you need **proof** that a property holds (not just confidence)
- Properties with complex preconditions
- Small, bounded domains

### When to Use Traditional PBT

- Properties involving complex data structures
- When random sampling is sufficient
- Properties that are expensive to encode symbolically
- Large search spaces where SMT times out

## How Negation Works

The property tester transforms:

```python
def prop_example(x: int):
    if x < 0:
        return True
    return x * x >= x
```

Into:

```python
def sat(x: int):
    if x < 0:
        return not True  # = False
    return not (x * x >= x)
```

This is then symbolically executed. The path conditions ensure that:
- If `x < 0`: the constraint `False` is added (unsatisfiable on this path)
- If `x >= 0`: the constraint `NOT (x² >= x)` is added

The SMT solver finds `x` satisfying any path, which would be a counterexample.
