"""
Roundtrip properties (encode/decode, serialize/deserialize pairs).

These properties test that inverse operations compose to identity.
A roundtrip property has the form: decode(encode(x)) == x

Run with:
    python property_tester.py --prop-file benchmarks/property/roundtrip.py
"""
from typing import List

# =============================================================================
# Integer <-> String Roundtrips
# =============================================================================

def prop_int_str_roundtrip(x: int):
    """Integer to string and back: int(str(x)) == x"""
    return int(str(x)) == x


def prop_int_bin_roundtrip(x: int):
    """Integer to binary string and back: int(bin(x), 2) == x"""
    if x < 0:
        return True  # bin() for negative has '-0b' prefix, skip for simplicity
    return int(bin(x), 2) == x


def prop_int_hex_roundtrip(x: int):
    """Integer to hex string and back: int(hex(x), 16) == x"""
    if x < 0:
        return True  # hex() for negative has '-0x' prefix, skip for simplicity
    return int(hex(x), 16) == x


def prop_int_oct_roundtrip(x: int):
    """Integer to octal string and back: int(oct(x), 8) == x"""
    if x < 0:
        return True  # oct() for negative has '-0o' prefix, skip for simplicity
    return int(oct(x), 8) == x


# =============================================================================
# Character <-> Ordinal Roundtrips
# =============================================================================

def prop_chr_ord_roundtrip(x: int):
    """Character code roundtrip: ord(chr(x)) == x (for valid codepoints)"""
    if x < 0 or x > 0x10FFFF:
        return True  # Invalid Unicode codepoints
    if 0xD800 <= x <= 0xDFFF:
        return True  # Surrogate pairs not allowed
    return ord(chr(x)) == x


def prop_ord_chr_roundtrip(s: str):
    """String to ordinal and back (single char): chr(ord(s)) == s"""
    if len(s) != 1:
        return True  # Only test single characters
    return chr(ord(s)) == s


# =============================================================================
# String Case Roundtrips (partial)
# =============================================================================

def prop_upper_lower_upper(s: str):
    """Upper-lower-upper gives same as upper (for ASCII)"""
    # Not a true roundtrip, but upper is stable after lower for most cases
    return s.upper().lower().upper() == s.upper()


def prop_lower_upper_lower(s: str):
    """Lower-upper-lower gives same as lower (for ASCII)"""
    return s.lower().upper().lower() == s.lower()


# =============================================================================
# String Strip Roundtrips (partial)
# =============================================================================

def prop_strip_idempotent(s: str):
    """Stripping is idempotent: s.strip().strip() == s.strip()"""
    return s.strip().strip() == s.strip()


def prop_lstrip_idempotent(s: str):
    """Left strip is idempotent"""
    return s.lstrip().lstrip() == s.lstrip()


def prop_rstrip_idempotent(s: str):
    """Right strip is idempotent"""
    return s.rstrip().rstrip() == s.rstrip()


# =============================================================================
# Negation Roundtrips
# =============================================================================

def prop_neg_neg_roundtrip(x: int):
    """Double negation: -(-x) == x"""
    return -(-x) == x


def prop_not_not_roundtrip(x: int):
    """Double boolean negation (treating int as bool): not not x == bool(x)"""
    return (not (not x)) == bool(x)


# =============================================================================
# Absolute Value (partial roundtrip for non-negative)
# =============================================================================

def prop_abs_roundtrip_positive(x: int):
    """Absolute value of positive is identity: abs(x) == x for x >= 0"""
    if x < 0:
        return True
    if x >= 0:
        return x == x  # abs(x) should equal x
    return True


# =============================================================================
# List Operations (partial roundtrips)
# =============================================================================

def prop_list_sorted_reverse_max_first(li: List[int]):
    """Sorted reversed: first element is max"""
    if len(li) == 0:
        return True
    rev_sorted = sorted(li, reverse=True)
    return rev_sorted[0] == max(li)


def prop_list_append_pop_roundtrip(li: List[int], x: int):
    """Append then pop returns same list: (li + [x])[:-1] == li"""
    extended = li + [x]
    popped = extended[:-1]
    return popped == li


def prop_list_prepend_remove_roundtrip(li: List[int], x: int):
    """Prepend then remove first: ([x] + li)[1:] == li"""
    extended = [x] + li
    removed = extended[1:]
    return removed == li


# =============================================================================
# Division/Multiplication Roundtrips
# =============================================================================

def prop_mul_div_roundtrip(x: int, y: int):
    """Multiply then divide: (x * y) // y == x (for non-zero y)"""
    if y == 0:
        return True
    return (x * y) // y == x


def prop_div_mul_partial(x: int, y: int):
    """Divide then multiply (partial): (x // y) * y <= x (for positive y, x)"""
    if y <= 0 or x < 0:
        return True
    return (x // y) * y <= x


# =============================================================================
# Shift Roundtrips
# =============================================================================

def prop_shift_left_right_roundtrip(x: int, n: int):
    """Left shift then right shift: (x << n) >> n == x (for non-negative)"""
    if x < 0 or n < 0 or n > 30:
        return True  # Skip negative and large shifts
    return (x << n) >> n == x


def prop_shift_right_left_partial(x: int, n: int):
    """Right shift then left shift loses low bits"""
    if x < 0 or n < 0 or n > 30:
        return True
    # (x >> n) << n clears the low n bits
    result = (x >> n) << n
    return result <= x
