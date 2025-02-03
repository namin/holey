"""
Tests for the core functionality.
"""
import pytest
from holey import SymbolicTracer, make_symbolic
from holey.backends import Z3Backend, MockBackend

def test_mock_backend_basic():
    tracer = SymbolicTracer(backend=MockBackend())
    x = make_symbolic(int, "x", tracer)
    y = make_symbolic(int, "y", tracer)
    
    result = x + y
    tracer.add_constraint(result.z3_expr == 10)
    tracer.add_constraint(x.z3_expr > 0)
    tracer.add_constraint(y.z3_expr > 0)
    
    # Mock backend always returns 'sat' but prints constraints
    assert tracer.backend.is_sat(tracer.check())

@pytest.mark.skipif(not Z3Backend.HAS_Z3, reason="Z3 not available")
def test_z3_backend_basic():
    tracer = SymbolicTracer(backend=Z3Backend())
    x = make_symbolic(int, "x", tracer)
    y = make_symbolic(int, "y", tracer)
    
    result = x + y
    tracer.add_constraint(result.z3_expr == 10)
    tracer.add_constraint(x.z3_expr > 0)
    tracer.add_constraint(y.z3_expr > 0)
    
    assert tracer.backend.is_sat(tracer.check())
    model = tracer.model()
    x_val = model[x.z3_expr].as_long()
    y_val = model[y.z3_expr].as_long()
    assert x_val + y_val == 10

def test_mock_backend_branching():
    tracer = SymbolicTracer(backend=MockBackend())
    x = make_symbolic(int, "x", tracer)
    
    def abs_value(a):
        if a > 0:
            return a
        return -a
    
    result = abs_value(x)
    tracer.add_constraint(result.z3_expr < 0)
    
    # Mock backend will show the contradiction in the constraints
    tracer.check()

@pytest.mark.skipif(not Z3Backend.HAS_Z3, reason="Z3 not available")
def test_z3_backend_branching():
    tracer = SymbolicTracer(backend=Z3Backend())
    x = make_symbolic(int, "x", tracer)
    
    def abs_value(a):
        if a > 0:
            return a
        return -a
    
    result = abs_value(x)
    tracer.add_constraint(result.z3_expr < 0)
    
    # Should be unsatisfiable since abs value is never negative
    assert not tracer.backend.is_sat(tracer.check())

def test_mock_backend_boolean_operations():
    tracer = SymbolicTracer(backend=MockBackend())
    x = make_symbolic(int, "x", tracer)
    y = make_symbolic(int, "y", tracer)
    
    condition = (x > 0) & (y > 0)
    tracer.add_constraint(condition.z3_expr)
    
    # Mock backend will show the combined conditions
    tracer.check()

@pytest.mark.skipif(not Z3Backend.HAS_Z3, reason="Z3 not available")
def test_z3_backend_boolean_operations():
    tracer = SymbolicTracer(backend=Z3Backend())
    x = make_symbolic(int, "x", tracer)
    y = make_symbolic(int, "y", tracer)
    
    condition = (x > 0) & (y > 0)
    tracer.add_constraint(condition.z3_expr)
    
    assert tracer.backend.is_sat(tracer.check())
    model = tracer.model()
    x_val = model[x.z3_expr].as_long()
    y_val = model[y.z3_expr].as_long()
    assert x_val > 0 and y_val > 0
