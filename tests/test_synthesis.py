"""
Tests for expression synthesis functionality.
"""
import pytest
from holey.synthesis import SymbolicHole
from holey.core import SymbolicTracer

def test_basic_hole():
    """Test basic hole creation and constraint adding"""
    tracer = SymbolicTracer()
    hole = SymbolicHole("test", int, tracer)
    
    # Add constraint that hole must be positive
    hole.add_constraint(hole.sym_var > 0, "test")
    
    # Add constraint that hole must be less than 10
    hole.add_constraint(hole.sym_var < 10, "test")
    
    # Try synthesis
    expr = hole.synthesize()
    assert expr is not None
    
    # Verify constraints
    value = eval(expr)
    assert isinstance(value, int)
    assert 0 < value < 10

def test_test_case_constraints():
    """Test adding test cases as constraints"""
    tracer = SymbolicTracer()
    hole = SymbolicHole("double", int, tracer)
    
    # Add test cases for doubling function
    test_cases = [
        ({"x": 1}, 2),
        ({"x": 2}, 4),
        ({"x": 3}, 6)
    ]
    
    for inputs, output in test_cases:
        hole.add_test_case(inputs, output)
        
    # Synthesize should find x * 2
    expr = hole.synthesize()
    assert expr is not None
    
    # Verify against test cases
    for inputs, expected in test_cases:
        result = eval(expr, {}, inputs)
        assert result == expected

def test_string_hole():
    """Test hole synthesis for strings"""
    tracer = SymbolicTracer()
    hole = SymbolicHole("concat", str, tracer)
    
    # Add test cases for string concatenation
    test_cases = [
        ({"a": "hello", "b": "world"}, "helloworld"),
        ({"a": "test", "b": "123"}, "test123")
    ]
    
    for inputs, output in test_cases:
        hole.add_test_case(inputs, output)
        
    # Synthesize
    expr = hole.synthesize()
    assert expr is not None
    
    # Verify 
    for inputs, expected in test_cases:
        result = eval(expr, {}, inputs)
        assert result == expected

def test_multiple_constraints():
    """Test combining multiple types of constraints"""
    tracer = SymbolicTracer()
    hole = SymbolicHole("abs", int, tracer)
    
    # Combine test cases and other constraints
    test_cases = [
        ({"x": -1}, 1),
        ({"x": 1}, 1),
        ({"x": 0}, 0)
    ]
    
    for inputs, output in test_cases:
        hole.add_test_case(inputs, output)
        
    # Add constraint that output must be non-negative
    hole.add_constraint(hole.sym_var >= 0, "type")
    
    # Synthesize
    expr = hole.synthesize()
    assert expr is not None
    
    # Verify
    for inputs, expected in test_cases:
        result = eval(expr, {}, inputs)
        assert result == expected
        assert result >= 0