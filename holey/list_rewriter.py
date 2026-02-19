"""
Post-processing pass that rewrites recursive List ADT encodings to native SMT-LIB theories.

Analyzes which list operations appear in the SMT-LIB text and selects the best
native encoding (Array, Seq, or fallback to original recursive ADT).
"""
import re
from typing import Dict, Set, Tuple, Optional, List


# Maps element type suffix to the SMT-LIB element type name
SUFFIX_TO_SMT_TYPE = {
    'int': 'Int',
    'string': 'String',
    'real': 'Real',
    'bool': 'Bool',
}


def rewrite_lists_to_native(smt2: str) -> str:
    """Main entry point: rewrite recursive list encoding to native SMT-LIB theories.

    Analyzes the SMT-LIB text, selects the best encoding per list variable,
    and rewrites declarations, operations, and logic line accordingly.
    Returns the rewritten SMT-LIB text, or the original if no rewriting is possible.
    """
    # Step 1: Find all list variables and their element types
    list_vars = _find_list_variables(smt2)
    if not list_vars:
        return smt2

    # Step 2: Analyze which operations are used
    ops = _analyze_operations(smt2)

    # Step 3: Select encoding
    encoding = _select_encoding(ops, list_vars, smt2)
    if encoding == 'original':
        return smt2

    # Step 3b: Check compatibility — fall back if assertions use features
    # incompatible with the quantifier-free logic we'd select
    if not _is_compatible(encoding, list_vars, smt2):
        return smt2

    # Step 4: Rewrite
    if encoding == 'array':
        return _rewrite_to_array(smt2, list_vars, ops)
    elif encoding == 'seq':
        return _rewrite_to_seq(smt2, list_vars, ops)

    return smt2


def _find_list_variables(smt2: str) -> Dict[str, str]:
    """Find all declared list variables and their element types.

    Returns dict mapping variable name -> element type (e.g. 'Int', 'String').
    """
    pattern = r'\(declare-const\s+(\S+)\s+\(List\s+(\S+)\)\)'
    result = {}
    for m in re.finditer(pattern, smt2):
        var_name = m.group(1)
        elem_type = m.group(2)
        result[var_name] = elem_type
    return result


def _analyze_operations(smt2: str) -> Dict[str, Set[str]]:
    """Analyze which list operations appear in the SMT-LIB text.

    Returns dict mapping operation category -> set of full operation names found.
    Categories: 'get', 'length', 'count', 'contains', 'sum', 'append',
                'index', 'set_len', 'map_add', 'slice', 'reverse',
                'cons_nil', 'join', 'split'
    """
    ops: Dict[str, Set[str]] = {
        'get': set(),
        'length': set(),
        'count': set(),
        'contains': set(),
        'sum': set(),
        'append': set(),
        'index': set(),
        'set_len': set(),
        'map_add': set(),
        'slice': set(),
        'reverse': set(),
        'cons_nil': set(),
        'join': set(),
        'split': set(),
    }

    # Only scan the assertion section (after declare-const), not library definitions
    assertion_section = _get_assertion_section(smt2)

    # Scan for list.op.suffix patterns in assertions only
    for m in re.finditer(r'list\.(get|length|count|contains|sum|append|index|set_len|map_add|slice|reverse)\.(\w+)', assertion_section):
        category = m.group(1)
        full_name = m.group(0)
        if category in ops:
            ops[category].add(full_name)

    # Check for direct cons/nil usage in assertions
    if re.search(r'\bcons\b', assertion_section):
        ops['cons_nil'].add('cons')
    if re.search(r'\bnil\b', assertion_section):
        ops['cons_nil'].add('nil')

    # Check for join/split in assertions
    if 'python.join' in assertion_section:
        ops['join'].add('python.join')
    if 'str.split' in assertion_section:
        ops['split'].add('str.split')

    return ops


def _get_assertion_section(smt2: str) -> str:
    """Extract the part of the SMT-LIB text after the library definitions.

    This is roughly everything from the first (declare-const onward.
    """
    m = re.search(r'\(declare-const\s', smt2)
    if m:
        return smt2[m.start():]
    return smt2


def _has_complex_ops(ops: Dict[str, Set[str]]) -> bool:
    """Check if operations that prevent native encoding are used."""
    return bool(
        ops['append'] or ops['cons_nil'] or ops['join'] or ops['split']
        or ops['slice'] or ops['reverse'] or ops['map_add'] or ops['set_len']
    )


def _all_get_indices_concrete(smt2: str) -> bool:
    """Check if all list.get calls use concrete (literal integer) indices."""
    assertion_section = _get_assertion_section(smt2)
    # Match list.get.suffix patterns and check their index argument
    # Pattern: (list.get.{suffix} varname index) — index is a non-paren token
    for m in re.finditer(r'\(list\.get\.\w+\s+\S+\s+([^()\s]+)', assertion_section):
        idx_str = m.group(1)
        # Check if it's a concrete integer
        if not re.match(r'^-?\d+$', idx_str):
            return False
    return True


def _is_compatible(encoding: str, list_vars: Dict[str, str], smt2: str) -> bool:
    """Check if the assertions are compatible with the target QF logic.

    Falls back to original if the constraints use features that the
    quantifier-free logic can't handle: quantifiers, nonlinear arithmetic,
    division/modulo, or Real element types with Seq encoding.
    """
    assertion_section = _get_assertion_section(smt2)
    elem_types = set(list_vars.values())

    # Quantifiers are incompatible with any QF_* logic
    if re.search(r'\bforall\b', assertion_section) or re.search(r'\bexists\b', assertion_section):
        return False

    # Nonlinear arithmetic: div, mod, and / (integer or real division)
    if re.search(r'\(\s*div\s', assertion_section) or re.search(r'\(\s*mod\s', assertion_section):
        return False
    if re.search(r'\(\s*/\s', assertion_section):
        return False

    # Recursive function definitions require quantifiers, incompatible with QF_*
    # Only check for non-list recursive defs (list.* ones get stripped by the rewriter)
    for m in re.finditer(r'\(define-fun-rec\s+(\S+)', smt2):
        fname = m.group(1)
        if not fname.startswith('list.') and not fname.startswith('python.join'):
            return False

    # Real element types with Seq: cvc5 has poor support for (Seq Real)
    if encoding == 'seq' and 'Real' in elem_types:
        return False

    # If Real type appears anywhere in declarations, likely nonlinear — skip
    if re.search(r'\(declare-const\s+\S+\s+Real\)', smt2):
        return False

    return True


def _select_encoding(ops: Dict[str, Set[str]], list_vars: Dict[str, str], smt2: str) -> str:
    """Select the best encoding based on operations used.

    Returns: 'array', 'seq', or 'original'
    """
    # If complex/structural operations are used, fall back
    if _has_complex_ops(ops):
        return 'original'

    has_get = bool(ops['get'])
    has_length = bool(ops['length'])
    has_count = bool(ops['count'])
    has_contains = bool(ops['contains'])
    has_sum = bool(ops['sum'])
    has_index = bool(ops['index'])

    # sum, index, and count are hard to express natively — fall back
    if has_sum or has_index or has_count:
        return 'original'

    # Only get with all concrete indices and no length → Array
    if has_get and not has_length and not has_count and not has_contains:
        if _all_get_indices_concrete(smt2):
            return 'array'

    # get + length → Seq  (also handles get with non-concrete indices)
    if has_get and has_length:
        return 'seq'

    # get only (concrete indices) + length → Seq
    if has_get:
        # If indices are concrete, array is fine; otherwise seq
        if _all_get_indices_concrete(smt2):
            return 'array'
        return 'seq'

    # Only length (no element access) → Seq
    if has_length and not has_get and not has_contains:
        return 'seq'

    # contains without structural ops → Seq
    if has_contains:
        return 'seq'

    return 'original'


def _extract_get_indices(smt2: str, var_name: str, suffix: str) -> List[int]:
    """Extract all concrete indices used in list.get calls for a given variable."""
    indices = set()
    pattern = rf'\(list\.get\.{re.escape(suffix)}\s+{re.escape(var_name)}\s+(-?\d+)\)'
    for m in re.finditer(pattern, smt2):
        indices.add(int(m.group(1)))
    return sorted(indices)


def _determine_logic(encoding: str, list_vars: Dict[str, str], ops: Dict[str, Set[str]], smt2: str) -> str:
    """Determine the appropriate SMT-LIB logic string."""
    elem_types = set(list_vars.values())

    if encoding == 'array':
        # Check if we need nonlinear arithmetic
        has_nonlinear = bool(re.search(r'\(\*\s', _get_assertion_section(smt2)))
        if 'String' in elem_types:
            return 'QF_AS' if not has_nonlinear else 'QF_AS'
        if has_nonlinear:
            return 'QF_ANIA'
        return 'QF_ALIA'
    elif encoding == 'seq':
        # QF_SLIA if any integer arithmetic or comparisons are present.
        # seq.len returns Int, so length constraints always need LIA.
        assertion_section = _get_assertion_section(smt2)
        has_int_ops = bool(
            ops['length']  # seq.len returns Int
            or 'Int' in elem_types  # (Seq Int) needs integer reasoning
            or re.search(r'(?<!\w)[><]=?\s', assertion_section)  # comparisons
            or re.search(r'(?<!\w)[\+\-\*]\s', assertion_section)  # arithmetic
        )
        if has_int_ops:
            return 'QF_SLIA'
        return 'QF_S'
    return 'ALL'


def _strip_library_definitions(smt2: str) -> str:
    """Remove all recursive list library definitions and the List datatype declaration.

    Strips:
    - (declare-datatypes ((List 1)) ...)
    - (define-fun-rec list.* ...)
    - (define-fun list.* ...)
    - Any (define-fun or define-fun-rec that uses (List ...) types in its signature
    - (define-fun-rec python.join ...) and similar List-dependent helpers
    """
    lines = smt2.split('\n')
    result = []
    skip_depth = 0
    skipping = False

    for line in lines:
        stripped = line.strip()

        # Start skipping multi-line definitions
        if not skipping:
            should_skip = False
            if stripped.startswith('(declare-datatypes ((List'):
                should_skip = True
            elif re.match(r'\(define-fun(?:-rec)?\s+list\.', stripped):
                should_skip = True
            elif re.match(r'\(define-fun(?:-rec)?\s+python\.join\b', stripped):
                should_skip = True
            elif re.match(r'\(define-fun(?:-rec)?\s+\S+.*\(List\s', stripped):
                # Any function whose signature references (List ...)
                should_skip = True

            if should_skip:
                skipping = True
                skip_depth = 0
                skip_depth += stripped.count('(') - stripped.count(')')
                if skip_depth <= 0:
                    skipping = False
                continue

        if skipping:
            skip_depth += stripped.count('(') - stripped.count(')')
            if skip_depth <= 0:
                skipping = False
            continue

        result.append(line)

    return '\n'.join(result)


def _rewrite_to_array(smt2: str, list_vars: Dict[str, str], ops: Dict[str, Set[str]]) -> str:
    """Rewrite SMT-LIB text to use Array encoding."""
    # Determine logic
    logic = _determine_logic('array', list_vars, ops, smt2)

    # Strip library definitions
    smt2 = _strip_library_definitions(smt2)

    # Replace (set-logic ALL) with the new logic
    smt2 = re.sub(r'\(set-logic\s+\S+\)', f'(set-logic {logic})', smt2)

    # Add produce-models option after set-logic
    if '(set-option :produce-models true)' not in smt2:
        smt2 = smt2.replace(f'(set-logic {logic})',
                            f'(set-logic {logic})\n(set-option :produce-models true)')

    for var_name, elem_type in list_vars.items():
        suffix = _type_to_suffix(elem_type)

        # Replace declaration: (declare-const x (List T)) -> (declare-const x (Array Int T))
        smt2 = smt2.replace(
            f'(declare-const {var_name} (List {elem_type}))',
            f'(declare-const {var_name} (Array Int {elem_type}))')

        # Extract indices used and create define-const helpers
        indices = _extract_get_indices(smt2, var_name, suffix)
        defines = []
        for idx in indices:
            defines.append(f'(define-const {var_name}{idx} {elem_type} (select {var_name} {idx}))')

        # Insert defines after the declaration
        if defines:
            decl = f'(declare-const {var_name} (Array Int {elem_type}))'
            smt2 = smt2.replace(decl, decl + '\n' + '\n'.join(defines))

        # Replace list.get.suffix calls: (list.get.{suffix} var idx) -> (select var idx)
        # Use a function to handle the replacement properly
        smt2 = _replace_list_get_with_select(smt2, var_name, suffix)

        # Remove any length constraints for this variable (arrays are unbounded)
        smt2 = _remove_length_constraints(smt2, var_name, suffix)

    # Clean up empty lines
    smt2 = re.sub(r'\n{3,}', '\n\n', smt2)

    return smt2


def _rewrite_to_seq(smt2: str, list_vars: Dict[str, str], ops: Dict[str, Set[str]]) -> str:
    """Rewrite SMT-LIB text to use Seq encoding."""
    # Determine logic
    logic = _determine_logic('seq', list_vars, ops, smt2)

    # Strip library definitions
    smt2 = _strip_library_definitions(smt2)

    # Replace (set-logic ALL) with the new logic
    smt2 = re.sub(r'\(set-logic\s+\S+\)', f'(set-logic {logic})', smt2)

    # Add produce-models option after set-logic
    if '(set-option :produce-models true)' not in smt2:
        smt2 = smt2.replace(f'(set-logic {logic})',
                            f'(set-logic {logic})\n(set-option :produce-models true)')

    for var_name, elem_type in list_vars.items():
        suffix = _type_to_suffix(elem_type)

        # Replace declaration: (declare-const x (List T)) -> (declare-const x (Seq T))
        smt2 = smt2.replace(
            f'(declare-const {var_name} (List {elem_type}))',
            f'(declare-const {var_name} (Seq {elem_type}))')

        # Replace list.get.suffix calls: (list.get.{suffix} var idx) -> (seq.nth var idx)
        smt2 = _replace_list_get_with_seq_nth(smt2, var_name, suffix)

        # Replace list.length.suffix calls: (list.length.{suffix} var) -> (seq.len var)
        smt2 = _replace_list_length_with_seq_len(smt2, var_name, suffix)

        # Replace list.contains.suffix calls:
        # (list.contains.{suffix} var val) -> (seq.contains var (seq.unit val))
        smt2 = _replace_list_contains_with_seq(smt2, var_name, suffix)

        # Replace list.count.suffix calls with seq-based counting
        # This is harder to do natively, so if count is present we might
        # need to keep it or approximate. For now, leave count constraints
        # that reference this var — the solver may still handle them.

    # Clean up empty lines
    smt2 = re.sub(r'\n{3,}', '\n\n', smt2)

    return smt2


def _type_to_suffix(elem_type: str) -> str:
    """Convert SMT element type to the suffix used in library function names."""
    for suffix, smt_type in SUFFIX_TO_SMT_TYPE.items():
        if smt_type == elem_type:
            return suffix
    return elem_type.lower()


def _replace_list_get_with_select(smt2: str, var_name: str, suffix: str) -> str:
    """Replace (list.get.{suffix} var idx) with (select var idx)."""
    # Handle concrete index: (list.get.{suffix} varname 42)
    pattern = rf'\(list\.get\.{re.escape(suffix)}\s+{re.escape(var_name)}\s+(-?\d+)\)'
    smt2 = re.sub(pattern, rf'(select {var_name} \1)', smt2)
    return smt2


def _replace_list_get_with_seq_nth(smt2: str, var_name: str, suffix: str) -> str:
    """Replace (list.get.{suffix} var idx) with (seq.nth var idx)."""
    # Handle any index expression — we need balanced-paren matching for complex indices
    # Simple case: concrete index
    pattern = rf'\(list\.get\.{re.escape(suffix)}\s+{re.escape(var_name)}\s+(-?\d+)\)'
    smt2 = re.sub(pattern, rf'(seq.nth {var_name} \1)', smt2)
    # Complex case: expression index — use a more general replacement
    # (list.get.{suffix} var (expr)) -> (seq.nth var (expr))
    pattern = rf'\(list\.get\.{re.escape(suffix)}\s+{re.escape(var_name)}\s+'
    smt2 = re.sub(pattern, f'(seq.nth {var_name} ', smt2)
    return smt2


def _replace_list_length_with_seq_len(smt2: str, var_name: str, suffix: str) -> str:
    """Replace (list.length.{suffix} var) with (seq.len var)."""
    pattern = rf'\(list\.length\.{re.escape(suffix)}\s+{re.escape(var_name)}\)'
    smt2 = re.sub(pattern, f'(seq.len {var_name})', smt2)
    return smt2


def _replace_list_contains_with_seq(smt2: str, var_name: str, suffix: str) -> str:
    """Replace (list.contains.{suffix} var val) with (seq.contains var (seq.unit val)).

    This handles simple value arguments. For complex expressions, a balanced-paren
    approach would be needed.
    """
    # Simple value: (list.contains.{suffix} var simpleVal)
    pattern = rf'\(list\.contains\.{re.escape(suffix)}\s+{re.escape(var_name)}\s+(\S+)\)'
    smt2 = re.sub(pattern, rf'(seq.contains {var_name} (seq.unit \1))', smt2)
    return smt2


def _remove_length_constraints(smt2: str, var_name: str, suffix: str) -> str:
    """Remove assertions that only constrain the length of a list variable.

    For Array encoding, length is meaningless, so we strip:
    (assert (= N (list.length.{suffix} var)))
    (assert (= (list.length.{suffix} var) N))
    """
    # Pattern: (assert (= <expr> (list.length.{suffix} var)))
    pattern = rf'\(assert\s+\(=\s+\S+\s+\(list\.length\.{re.escape(suffix)}\s+{re.escape(var_name)}\)\)\)\n?'
    smt2 = re.sub(pattern, '', smt2)
    # Reverse order: (assert (= (list.length.{suffix} var) <expr>))
    pattern = rf'\(assert\s+\(=\s+\(list\.length\.{re.escape(suffix)}\s+{re.escape(var_name)}\)\s+\S+\)\)\n?'
    smt2 = re.sub(pattern, '', smt2)
    return smt2
