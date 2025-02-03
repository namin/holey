from typing import Any, Optional, List, Type
from dataclasses import dataclass
from contextlib import contextmanager
import z3
from .backends import default_backend, Backend

class SymbolicTracer:
    """Tracer for symbolic execution"""

    def __init__(self, backend: Optional[Backend] = None):
        """Initialize tracer with optional backend"""
        self.backend = backend or default_backend()
        self.solver = self.backend.Solver()
        self.path_conditions = []
        self._stack = []
        
    def __enter__(self):
        self._stack.append((self.path_conditions.copy(), self.solver.assertions()))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stack:
            old_conditions, old_assertions = self._stack.pop()
            self.path_conditions = old_conditions
            self.solver = self.backend.Solver()
            self.solver.add(old_assertions)
    
    def add_constraint(self, constraint):
        self.path_conditions.append(constraint)
        self.solver.add(constraint)
    
    def check(self):
        return self.solver.check()
    
    def model(self):
        return self.solver.model()
    
    @contextmanager
    def branch(self):
        """Context manager for handling branches in symbolic execution"""
        old_conditions = self.path_conditions.copy()
        old_solver = self.backend.Solver()
        old_solver.add(self.solver.assertions())
        try:
            yield
        finally:
            self.path_conditions = old_conditions
            self.solver = old_solver
