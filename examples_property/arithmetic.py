"""
Example properties for testing with property_tester.py

Run with:
    python property_tester.py --prop-file examples_property/arithmetic.py
"""

# =============================================================================
# Properties that should HOLD (no counterexample expected)
# =============================================================================

def prop_add_commutative(x: int, y: int):
    """Addition is commutative."""
    return x + y == y + x


def prop_mul_commutative(x: int, y: int):
    """Multiplication is commutative."""
    return x * y == y * x


def prop_add_zero_identity(x: int):
    """Adding zero is identity."""
    return x + 0 == x


def prop_mul_one_identity(x: int):
    """Multiplying by one is identity."""
    return x * 1 == x


def prop_double_is_add_self(x: int):
    """Doubling equals adding to self."""
    return x * 2 == x + x


def prop_square_non_negative(x: int):
    """Squares are non-negative."""
    return x * x >= 0


def prop_abs_non_negative(x: int):
    """Absolute value is non-negative."""
    if x >= 0:
        return x >= 0
    else:
        return -x >= 0


def prop_distribute_mul_over_add(x: int, y: int, z: int):
    """Multiplication distributes over addition."""
    return x * (y + z) == x * y + x * z


# =============================================================================
# Properties that should FAIL (counterexample expected)
# =============================================================================

def prop_bad_always_positive(x: int):
    """INTENTIONALLY WRONG: Claims all integers are positive.

    Should find counterexample like x = 0 or x = -1.
    """
    return x > 0


def prop_bad_always_less_than_100(x: int):
    """INTENTIONALLY WRONG: Claims all integers are less than 100.

    Should find counterexample like x = 100.
    """
    return x < 100


def prop_bad_square_equals_double(x: int):
    """INTENTIONALLY WRONG: Claims x^2 == 2x for all x.

    Only true for x = 0 and x = 2. Should find counterexample.
    """
    return x * x == 2 * x


def prop_bad_sum_greater(x: int, y: int):
    """INTENTIONALLY WRONG: Claims x + y > x for all x, y.

    Fails when y <= 0. Should find counterexample.
    """
    return x + y > x


# =============================================================================
# Properties with preconditions (via control flow)
# =============================================================================

def prop_positive_square_greater(x: int):
    """For positive x, x^2 >= x.

    Uses early return to handle precondition.
    """
    if x <= 0:
        return True  # Vacuously true for non-positive
    return x * x >= x


def prop_nonzero_square_positive(x: int):
    """For non-zero x, x^2 > 0."""
    if x == 0:
        return True  # Skip zero
    return x * x > 0


def prop_division_inverse(x: int, y: int):
    """For non-zero y: (x * y) // y == x.

    Note: This uses integer division, should hold.
    """
    if y == 0:
        return True  # Skip division by zero
    return (x * y) // y == x
