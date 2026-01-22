"""
Intentionally buggy properties for testing the property tester.

ALL properties in this file should FAIL (counterexample expected).
This validates that the property tester correctly finds violations.

Run with:
    python property_tester.py --prop-file benchmarks/property/buggy.py

Expected: All properties should have counterexamples found.
"""
from typing import List

# =============================================================================
# Off-by-One Errors
# =============================================================================

def prop_buggy_off_by_one_lt(x: int):
    """BUG: Claims x < x + 1 implies x <= x (should be x < x+1, always true)

    Actually tests wrong thing: x + 1 > x + 1 is always false.
    Counterexample: any x
    """
    return x + 1 > x + 1


def prop_buggy_off_by_one_bound(x: int):
    """BUG: Claims all integers are strictly less than 10.

    Counterexample: x = 10
    """
    return x < 10


def prop_buggy_off_by_one_ge(x: int):
    """BUG: Claims all integers are >= 1.

    Counterexample: x = 0 or x = -1
    """
    return x >= 1


# =============================================================================
# Sign Errors
# =============================================================================

def prop_buggy_always_positive(x: int):
    """BUG: Claims all integers are positive.

    Counterexample: x = 0 or any negative
    """
    return x > 0


def prop_buggy_always_negative(x: int):
    """BUG: Claims all integers are negative.

    Counterexample: x = 0 or any positive
    """
    return x < 0


def prop_buggy_always_nonzero(x: int):
    """BUG: Claims all integers are non-zero.

    Counterexample: x = 0
    """
    return x != 0


def prop_buggy_abs_equals_self(x: int):
    """BUG: Claims abs(x) == x for all x.

    Counterexample: any negative x
    """
    ax = x if x >= 0 else -x
    return ax == x


# =============================================================================
# Arithmetic Errors
# =============================================================================

def prop_buggy_square_equals_double(x: int):
    """BUG: Claims x^2 == 2*x for all x.

    Only true for x = 0 and x = 2.
    Counterexample: x = 1 or x = 3 or any other
    """
    return x * x == 2 * x


def prop_buggy_add_equals_mul(x: int, y: int):
    """BUG: Claims x + y == x * y for all x, y.

    Only true for specific pairs like (0,0), (2,2).
    Counterexample: x = 1, y = 1 (gives 2 != 1)
    """
    return x + y == x * y


def prop_buggy_sum_greater(x: int, y: int):
    """BUG: Claims x + y > x for all x, y.

    Fails when y <= 0.
    Counterexample: y = 0 or y = -1
    """
    return x + y > x


def prop_buggy_product_greater(x: int, y: int):
    """BUG: Claims x * y > x for all x, y.

    Fails for many cases: y <= 1, x <= 0, etc.
    Counterexample: x = 1, y = 1
    """
    return x * y > x


def prop_buggy_division_exact(x: int, y: int):
    """BUG: Claims (x // y) * y == x for all non-zero y.

    Only true when y divides x evenly.
    Counterexample: x = 5, y = 2 (gives 4 != 5)
    """
    if y == 0:
        return True
    return (x // y) * y == x


# =============================================================================
# Comparison Errors
# =============================================================================

def prop_buggy_lt_or_gt(x: int, y: int):
    """BUG: Claims x < y or x > y for all x, y.

    Misses the equality case.
    Counterexample: x = y (any equal values)
    """
    return x < y or x > y


def prop_buggy_transitive_wrong(x: int, y: int, z: int):
    """BUG: Claims x < y and y < z implies x < y (instead of x < z).

    This is trivially true (premise includes conclusion), but let's
    test a truly wrong version: claims x < z implies y < z.
    """
    if x < z:
        return y < z
    return True


# =============================================================================
# String Errors
# =============================================================================

def prop_buggy_str_never_empty(s: str):
    """BUG: Claims all strings are non-empty.

    Counterexample: s = ""
    """
    return len(s) > 0


def prop_buggy_str_short(s: str):
    """BUG: Claims all strings have length < 5.

    Counterexample: any string with 5+ chars
    """
    return len(s) < 5


def prop_buggy_str_equals_upper(s: str):
    """BUG: Claims s == s.upper() for all strings.

    Counterexample: any string with lowercase letters
    """
    return s == s.upper()


def prop_buggy_str_contains_a(s: str):
    """BUG: Claims all strings contain 'a'.

    Counterexample: s = "b" or s = ""
    """
    return 'a' in s


# =============================================================================
# List Errors
# =============================================================================

def prop_buggy_list_never_empty(li: List[int]):
    """BUG: Claims all lists are non-empty.

    Counterexample: li = []
    """
    return len(li) > 0


def prop_buggy_list_sum_positive(li: List[int]):
    """BUG: Claims sum of any list is positive.

    Counterexample: li = [] (sum = 0) or li = [-1]
    """
    return sum(li) > 0


def prop_buggy_list_min_positive(li: List[int]):
    """BUG: Claims minimum of any non-empty list is positive.

    Counterexample: li = [0] or li = [-1]
    """
    if len(li) == 0:
        return True
    return min(li) > 0


def prop_buggy_list_sorted_equals_self(li: List[int]):
    """BUG: Claims sorted(li) == li for all lists.

    Only true for already-sorted lists.
    Counterexample: li = [2, 1]
    """
    return sorted(li) == li


def prop_buggy_list_all_equal(li: List[int]):
    """BUG: Claims min(li) == max(li) for all non-empty lists.

    Only true when all elements are equal.
    Counterexample: li = [1, 2]
    """
    if len(li) == 0:
        return True
    return min(li) == max(li)


# =============================================================================
# Logic Errors
# =============================================================================

def prop_buggy_false(x: int):
    """BUG: Always returns False.

    Counterexample: any x
    """
    return False


def prop_buggy_contradiction(x: int):
    """BUG: Claims x < 0 and x >= 0 simultaneously.

    Counterexample: any x
    """
    return x < 0 and x >= 0
