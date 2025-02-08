"""
Tests for the core functionality.
"""
import pytest
from holey import SymbolicTracer, make_symbolic
from holey.backends import Z3Backend, MockBackend

def backend_basic(backend):
    tracer = SymbolicTracer(backend=backend)
    x = make_symbolic(int, "x", tracer)
    y = make_symbolic(int, "y", tracer)
    
    result = x + y
    tracer.add_constraint(result.z3_expr == 10)
    tracer.add_constraint(x.z3_expr > 0)
    tracer.add_constraint(y.z3_expr > 0)
    
    # Mock backend always returns 'sat' but prints constraints
    assert tracer.backend.is_sat(tracer.check())

def test_mock_backend_basic():
    backend_basic(MockBackend())

@pytest.mark.skipif(not Z3Backend.HAS_Z3, reason="Z3 not available")
def test_z3_backend_basic():
    backend_basic(Z3Backend())

def backend_branching(backend):
    tracer = SymbolicTracer(backend=backend)
    x = make_symbolic(int, "x", tracer)
    
    def abs_value(a):
        if a > 0:
            return a
        return -a

    def thunk():
        tracer.add_constraint(abs_value(x).z3_expr < 0)
        return True

    tracer.driver(thunk)
    
    # Should be unsatisfiable since abs value is never negative
    assert not tracer.backend.is_sat(tracer.check())

def test_mock_backend_branching():
    backend_branching(MockBackend())

@pytest.mark.skipif(not Z3Backend.HAS_Z3, reason="Z3 not available")
def test_z3_backend_branching():
    backend_branching(Z3Backend())

def backend_boolean_operations(backend):
    tracer = SymbolicTracer(backend=backend)
    x = make_symbolic(int, "x", tracer)
    y = make_symbolic(int, "y", tracer)
    
    condition = (x > 0) & (y > 0)
    tracer.add_constraint(condition.z3_expr)
    
    tracer.check()
    return tracer, x, y

def test_mock_backend_boolean_operations():
    backend_boolean_operations(MockBackend())

@pytest.mark.skipif(not Z3Backend.HAS_Z3, reason="Z3 not available")
def test_z3_backend_boolean_operations():
    tracer, x, y = backend_boolean_operations(Z3Backend())
    model = tracer.model()
    x_val = model[x.z3_expr].as_long()
    y_val = model[y.z3_expr].as_long()
    assert x_val > 0 and y_val > 0
