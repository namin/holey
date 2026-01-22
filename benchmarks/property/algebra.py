"""
Algebraic properties for integers.

These properties test fundamental mathematical laws that should hold
for all integers. Most should pass (no counterexample).

Run with:
    python property_tester.py --prop-file benchmarks/property/algebra.py
"""

# =============================================================================
# Addition Properties
# =============================================================================

def prop_add_commutative(x: int, y: int):
    """Addition is commutative: x + y == y + x"""
    return x + y == y + x


def prop_add_associative(x: int, y: int, z: int):
    """Addition is associative: (x + y) + z == x + (y + z)"""
    return (x + y) + z == x + (y + z)


def prop_add_identity(x: int):
    """Zero is additive identity: x + 0 == x"""
    return x + 0 == x


def prop_add_inverse(x: int):
    """Additive inverse: x + (-x) == 0"""
    return x + (-x) == 0


# =============================================================================
# Multiplication Properties
# =============================================================================

def prop_mul_commutative(x: int, y: int):
    """Multiplication is commutative: x * y == y * x"""
    return x * y == y * x


def prop_mul_associative(x: int, y: int, z: int):
    """Multiplication is associative: (x * y) * z == x * (y * z)"""
    return (x * y) * z == x * (y * z)


def prop_mul_identity(x: int):
    """One is multiplicative identity: x * 1 == x"""
    return x * 1 == x


def prop_mul_zero(x: int):
    """Zero annihilates: x * 0 == 0"""
    return x * 0 == 0


# =============================================================================
# Distributivity
# =============================================================================

def prop_distribute_left(x: int, y: int, z: int):
    """Left distributivity: x * (y + z) == x * y + x * z"""
    return x * (y + z) == x * y + x * z


def prop_distribute_right(x: int, y: int, z: int):
    """Right distributivity: (x + y) * z == x * z + y * z"""
    return (x + y) * z == x * z + y * z


# =============================================================================
# Comparison Properties
# =============================================================================

def prop_eq_reflexive(x: int):
    """Equality is reflexive: x == x"""
    return x == x


def prop_lt_irreflexive(x: int):
    """Less-than is irreflexive: not (x < x)"""
    return not (x < x)


def prop_trichotomy(x: int, y: int):
    """Trichotomy: exactly one of x < y, x == y, x > y holds"""
    lt = x < y
    eq = x == y
    gt = x > y
    # Exactly one is true
    return (lt and not eq and not gt) or (not lt and eq and not gt) or (not lt and not eq and gt)


def prop_lt_transitive(x: int, y: int, z: int):
    """Less-than is transitive: x < y and y < z implies x < z"""
    if x < y and y < z:
        return x < z
    return True


# =============================================================================
# Absolute Value Properties
# =============================================================================

def prop_abs_non_negative(x: int):
    """Absolute value is non-negative"""
    if x >= 0:
        return x >= 0
    else:
        return -x >= 0


def prop_abs_idempotent(x: int):
    """Absolute value is idempotent: abs(abs(x)) == abs(x)"""
    if x >= 0:
        ax = x
    else:
        ax = -x
    # abs of ax
    if ax >= 0:
        aax = ax
    else:
        aax = -ax
    return aax == ax


def prop_abs_multiplicative(x: int, y: int):
    """Absolute value is multiplicative: |x * y| == |x| * |y|"""
    # Compute |x|
    ax = x if x >= 0 else -x
    # Compute |y|
    ay = y if y >= 0 else -y
    # Compute |x * y|
    xy = x * y
    axy = xy if xy >= 0 else -xy
    return axy == ax * ay


# =============================================================================
# Square Properties
# =============================================================================

def prop_square_non_negative(x: int):
    """Squares are non-negative: x^2 >= 0"""
    return x * x >= 0


def prop_square_even_function(x: int):
    """Square is an even function: (-x)^2 == x^2"""
    return (-x) * (-x) == x * x


# =============================================================================
# Division Properties (with preconditions)
# =============================================================================

def prop_div_identity(x: int):
    """Division by 1 is identity: x // 1 == x"""
    return x // 1 == x


def prop_div_self(x: int):
    """Division by self is 1 (for non-zero): x // x == 1"""
    if x == 0:
        return True  # Skip zero
    return x // x == 1


def prop_mod_bounds(x: int, y: int):
    """Modulo is bounded: 0 <= x % y < |y| (for positive y)"""
    if y <= 0:
        return True  # Only test positive divisors
    r = x % y
    return r >= 0 and r < y


def prop_div_mod_identity(x: int, y: int):
    """Division-modulo identity: x == (x // y) * y + (x % y)"""
    if y == 0:
        return True  # Skip zero
    return x == (x // y) * y + (x % y)


# =============================================================================
# Bitwise Properties
# =============================================================================

def prop_double_is_shift(x: int):
    """Doubling equals left shift: x * 2 == x + x"""
    return x * 2 == x + x


def prop_xor_self(x: int):
    """XOR with self is zero: x ^ x == 0"""
    return (x ^ x) == 0


def prop_xor_zero(x: int):
    """XOR with zero is identity: x ^ 0 == x"""
    return (x ^ 0) == x
