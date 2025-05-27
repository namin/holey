"""
Analyze list puzzles to determine if bounded encoding would help
"""

import ast
import re

class ListPuzzleAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze list operations"""
    
    def __init__(self):
        self.has_recursive_ops = False
        self.has_count = False
        self.has_length_constraint = False
        self.has_index_access = False
        self.has_list_comprehension = False
        self.has_sorting = False
        self.has_all_any = False
        self.max_explicit_index = 0
        self.max_range_value = 0
        self.length_constraints = []
        
    def visit_Call(self, node):
        """Visit function calls"""
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method == 'count':
                self.has_count = True
            elif method == 'index':
                self.has_recursive_ops = True
            elif method in ['sort', 'sorted']:
                self.has_sorting = True
                
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name == 'len':
                self.has_length_constraint = True
                # Try to extract length value if it's a comparison
                parent = getattr(node, '_parent', None)
                if parent and isinstance(parent, ast.Compare):
                    self._extract_length_constraint(parent)
                    
            elif func_name == 'sorted':
                self.has_sorting = True
            elif func_name in ['all', 'any']:
                self.has_all_any = True
            elif func_name == 'range':
                # Extract range value
                if node.args and isinstance(node.args[0], ast.Constant):
                    self.max_range_value = max(self.max_range_value, node.args[0].value)
                    
        self.generic_visit(node)
        
    def visit_Subscript(self, node):
        """Visit list[index] access"""
        self.has_index_access = True
        
        # Extract index value if constant
        if isinstance(node.slice, ast.Constant):
            self.max_explicit_index = max(self.max_explicit_index, node.slice.value)
        elif isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Constant):
            self.max_explicit_index = max(self.max_explicit_index, node.slice.value.value)
            
        self.generic_visit(node)
        
    def visit_ListComp(self, node):
        """Visit list comprehensions"""
        self.has_list_comprehension = True
        self.generic_visit(node)
        
    def visit_Compare(self, node):
        """Visit comparisons to extract constraints"""
        # Store parent reference for child nodes
        for child in ast.walk(node):
            child._parent = node
        self.generic_visit(node)
        
    def _extract_length_constraint(self, compare_node):
        """Extract length constraint from comparison"""
        if len(compare_node.ops) == 1 and isinstance(compare_node.ops[0], ast.Eq):
            if len(compare_node.comparators) == 1:
                comp = compare_node.comparators[0]
                if isinstance(comp, ast.Constant):
                    self.length_constraints.append(('=', comp.value))
                    

def analyze_list_puzzle(sat_func):
    """Analyze a SAT function to determine list characteristics"""
    
    # First, do regex-based analysis for quick checks
    regex_analysis = {
        'has_count': bool(re.search(r'\.count\s*\(', sat_func)),
        'has_length': bool(re.search(r'len\s*\(', sat_func)),
        'has_index': bool(re.search(r'\[\s*\d+\s*\]', sat_func)),
        'has_sorted': bool(re.search(r'sorted\s*\(', sat_func)),
        'has_all_any': bool(re.search(r'\b(all|any)\s*\(', sat_func)),
        'has_sum': bool(re.search(r'sum\s*\(', sat_func)),
        'has_in': bool(re.search(r'\sin\s', sat_func)),
    }
    
    # Extract numeric patterns
    indices = [int(m) for m in re.findall(r'\[\s*(\d+)\s*\]', sat_func)]
    ranges = [int(m) for m in re.findall(r'range\s*\((\d+)\)', sat_func)]
    length_eq = re.findall(r'len\s*\([^)]+\)\s*==\s*(\d+)', sat_func)
    
    # Now do AST analysis for more complex patterns
    try:
        tree = ast.parse(sat_func)
        analyzer = ListPuzzleAnalyzer()
        analyzer.visit(tree)
        
        ast_analysis = {
            'has_recursive_ops': analyzer.has_recursive_ops,
            'has_count': analyzer.has_count,
            'has_length_constraint': analyzer.has_length_constraint,
            'has_index_access': analyzer.has_index_access,
            'has_list_comprehension': analyzer.has_list_comprehension,
            'has_sorting': analyzer.has_sorting,
            'has_all_any': analyzer.has_all_any,
            'max_explicit_index': analyzer.max_explicit_index,
            'max_range_value': analyzer.max_range_value,
            'length_constraints': analyzer.length_constraints,
        }
    except:
        ast_analysis = {}
    
    # Combine analyses
    return {
        'regex': regex_analysis,
        'ast': ast_analysis,
        'indices': indices,
        'ranges': ranges,
        'length_values': [int(l) for l in length_eq] if length_eq else [],
        'max_index': max(indices) if indices else 0,
        'max_range': max(ranges) if ranges else 0,
    }


def should_use_bounded_encoding(analysis):
    """Decide if bounded encoding should be used based on analysis"""
    
    # Reasons to use bounded encoding:
    reasons = []
    
    # 1. Has count operations (recursive in unbounded)
    if analysis['regex']['has_count'] or analysis['ast'].get('has_count', False):
        reasons.append('count_operations')
        
    # 2. Has explicit length constraints
    if analysis['length_values']:
        reasons.append('explicit_length')
        
    # 3. Uses all/any with list operations
    if analysis['regex']['has_all_any'] or analysis['ast'].get('has_all_any', False):
        reasons.append('all_any_operations')
        
    # 4. Has sorting (very expensive with recursive lists)
    if analysis['regex']['has_sorted'] or analysis['ast'].get('has_sorting', False):
        reasons.append('sorting')
        
    # 5. Large index access
    if analysis['max_index'] > 10:
        reasons.append('large_indices')
        
    # 6. List comprehensions (often generate many constraints)
    if analysis['ast'].get('has_list_comprehension', False):
        reasons.append('list_comprehension')
        
    # Reasons NOT to use bounded encoding:
    # - Very simple constraints with small indices
    # - No recursive operations
    
    # Decision
    use_bounded = len(reasons) > 0
    
    # Estimate max length needed
    max_length = 100  # default
    
    if analysis['length_values']:
        max_length = max(analysis['length_values']) + 10
    elif analysis['max_index'] > 0:
        max_length = analysis['max_index'] + 10
    elif analysis['max_range'] > 0:
        max_length = analysis['max_range'] * 2
        
    # Special case for count patterns like Study_5:0
    # If we have count(i) == i for i in range(n), we need sum(range(n)) elements
    if 'count_operations' in reasons and analysis['max_range'] > 0:
        n = analysis['max_range']
        # Sum of 0 + 1 + 2 + ... + (n-1) = n*(n-1)/2
        needed_length = n * (n - 1) // 2
        max_length = max(max_length, needed_length + 10)
        
    return {
        'use_bounded': use_bounded,
        'reasons': reasons,
        'suggested_max_length': min(max_length, 1000),  # Cap at 1000
    }


# Example usage
if __name__ == "__main__":
    test_cases = [
        "len(li) == 10 and li.count(li[3]) == 2",
        "all([li.count(i) == i for i in range(10)])",
        "sorted(li) == list(range(999)) and all(li[i] != i for i in range(len(li)))",
        "ls[1234] in ls[1235] and ls[1234] != ls[1235]",
        "sum(li) == 100 and len(li) == 20",
        "li[0] < li[1] < li[2]",  # Simple case - might not need bounded
    ]
    
    for sat_func in test_cases:
        print(f"\nAnalyzing: {sat_func}")
        analysis = analyze_list_puzzle(sat_func)
        decision = should_use_bounded_encoding(analysis)
        
        print(f"  Max index: {analysis['max_index']}")
        print(f"  Max range: {analysis['max_range']}")
        print(f"  Has count: {analysis['regex']['has_count']}")
        print(f"  Decision: {'BOUNDED' if decision['use_bounded'] else 'RECURSIVE'}")
        print(f"  Reasons: {decision['reasons']}")
        print(f"  Suggested max length: {decision['suggested_max_length']}")