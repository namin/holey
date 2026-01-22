"""
Collection properties for lists and strings.

These properties test invariants that should hold for list and string
operations. Tests requiring specific list sizes use preconditions.

Run with:
    python property_tester.py --prop-file benchmarks/property/collections.py
"""
from typing import List

# =============================================================================
# String Length Properties
# =============================================================================

def prop_str_len_non_negative(s: str):
    """String length is non-negative"""
    return len(s) >= 0


def prop_str_concat_len(s: str, t: str):
    """Concatenation length: len(s + t) == len(s) + len(t)"""
    return len(s + t) == len(s) + len(t)


def prop_str_repeat_len(s: str, n: int):
    """Repetition length: len(s * n) == len(s) * n (for n >= 0)"""
    if n < 0:
        return True  # s * negative = empty string, different semantics
    return len(s * n) == len(s) * n


# =============================================================================
# String Case Properties
# =============================================================================

def prop_str_upper_idempotent(s: str):
    """Upper is idempotent: s.upper().upper() == s.upper()"""
    return s.upper().upper() == s.upper()


def prop_str_lower_idempotent(s: str):
    """Lower is idempotent: s.lower().lower() == s.lower()"""
    return s.lower().lower() == s.lower()


def prop_str_upper_len(s: str):
    """Upper preserves length: len(s.upper()) == len(s)"""
    return len(s.upper()) == len(s)


def prop_str_lower_len(s: str):
    """Lower preserves length: len(s.lower()) == len(s)"""
    return len(s.lower()) == len(s)


# =============================================================================
# String Contains Properties
# =============================================================================

def prop_str_contains_self(s: str):
    """String contains itself: s in s"""
    return s in s


def prop_str_contains_empty(s: str):
    """String contains empty string: '' in s"""
    return '' in s


def prop_str_startswith_prefix(s: str):
    """String starts with empty prefix"""
    return s.startswith('')


def prop_str_endswith_suffix(s: str):
    """String ends with empty suffix"""
    return s.endswith('')


# =============================================================================
# String Index Properties
# =============================================================================

def prop_str_index_bounds(s: str, i: int):
    """Valid index access (within bounds)"""
    if len(s) == 0:
        return True  # Skip empty strings
    if i < 0 or i >= len(s):
        return True  # Only test valid indices
    c = s[i]
    return len(c) == 1


def prop_str_slice_len(s: str, i: int, j: int):
    """Slice length: len(s[i:j]) == max(0, j - i) for valid i, j"""
    if i < 0 or j < 0:
        return True  # Skip negative indices
    if i > len(s) or j > len(s):
        return True  # Skip out of bounds
    if i > j:
        return len(s[i:j]) == 0
    return len(s[i:j]) == j - i


# =============================================================================
# List Length Properties
# =============================================================================

def prop_list_len_non_negative(li: List[int]):
    """List length is non-negative"""
    return len(li) >= 0


def prop_list_concat_len(li: List[int], lj: List[int]):
    """Concatenation length: len(li + lj) == len(li) + len(lj)"""
    return len(li + lj) == len(li) + len(lj)


# =============================================================================
# List Element Properties
# =============================================================================

def prop_list_index_in_list(li: List[int], i: int):
    """Element at valid index is in list"""
    if len(li) == 0:
        return True
    if i < 0 or i >= len(li):
        return True  # Only test valid indices
    return li[i] in li


def prop_list_count_bounds(li: List[int], x: int):
    """Count is bounded: 0 <= count(x) <= len(li)"""
    c = li.count(x)
    return c >= 0 and c <= len(li)


def prop_list_count_zero_not_in(li: List[int], x: int):
    """Count zero implies not in list"""
    if li.count(x) == 0:
        return x not in li
    return True


def prop_list_count_positive_in(li: List[int], x: int):
    """Positive count implies in list"""
    if li.count(x) > 0:
        return x in li
    return True


# =============================================================================
# List Min/Max Properties (non-empty lists)
# =============================================================================

def prop_list_min_in_list(li: List[int]):
    """Minimum is in the list"""
    if len(li) == 0:
        return True  # Skip empty lists
    m = min(li)
    return m in li


def prop_list_max_in_list(li: List[int]):
    """Maximum is in the list"""
    if len(li) == 0:
        return True  # Skip empty lists
    m = max(li)
    return m in li


def prop_list_min_le_max(li: List[int]):
    """Minimum <= Maximum"""
    if len(li) == 0:
        return True
    return min(li) <= max(li)


def prop_list_min_le_all(li: List[int], i: int):
    """Minimum <= all elements"""
    if len(li) == 0:
        return True
    if i < 0 or i >= len(li):
        return True
    return min(li) <= li[i]


def prop_list_max_ge_all(li: List[int], i: int):
    """Maximum >= all elements"""
    if len(li) == 0:
        return True
    if i < 0 or i >= len(li):
        return True
    return max(li) >= li[i]


# =============================================================================
# List Sum Properties
# =============================================================================

def prop_list_sum_empty():
    """Sum of empty list is zero"""
    return sum([]) == 0


def prop_list_sum_singleton(x: int):
    """Sum of singleton is element: sum([x]) == x"""
    return sum([x]) == x


def prop_list_sum_bounds(li: List[int]):
    """Sum is between n*min and n*max"""
    if len(li) == 0:
        return True
    n = len(li)
    s = sum(li)
    return s >= n * min(li) and s <= n * max(li)


# =============================================================================
# List Sorted Properties
# =============================================================================

def prop_list_sorted_len(li: List[int]):
    """Sorting preserves length"""
    return len(sorted(li)) == len(li)


def prop_list_sorted_idempotent(li: List[int]):
    """Sorting is idempotent: sorted(sorted(li)) == sorted(li)"""
    return sorted(sorted(li)) == sorted(li)


def prop_list_sorted_min_first(li: List[int]):
    """First element of sorted list is minimum"""
    if len(li) == 0:
        return True
    return sorted(li)[0] == min(li)


def prop_list_sorted_max_last(li: List[int]):
    """Last element of sorted list is maximum"""
    if len(li) == 0:
        return True
    return sorted(li)[-1] == max(li)


# =============================================================================
# List Reverse Properties
# =============================================================================

def prop_list_reverse_len(li: List[int]):
    """Reversing preserves length"""
    return len(list(reversed(li))) == len(li)


def prop_list_reverse_involution(li: List[int]):
    """Reversing twice is identity"""
    return list(reversed(list(reversed(li)))) == li
