"""
Tests for the core functionality.
"""
import pytest
from holey import SymbolicTracer, make_symbolic
from holey.backend import default_backend

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

def test_default_backend_basic():
    backend_basic(default_backend())

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

def test_default_backend_branching():
    backend_branching(default_backend())

def backend_boolean_operations(backend):
    tracer = SymbolicTracer(backend=backend)
    x = make_symbolic(int, "x", tracer)
    y = make_symbolic(int, "y", tracer)
    
    condition = (x > 0) & (y > 0)
    tracer.add_constraint(condition.z3_expr)
    
    tracer.check()

    model = tracer.model()
    x_val = model[x.name]
    y_val = model[y.name]
    assert x_val > 0 and y_val > 0

def test_default_backend_boolean_operations():
    backend_boolean_operations(default_backend())
