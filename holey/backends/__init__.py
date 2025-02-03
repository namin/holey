"""
Backend implementations for symbolic execution and synthesis.
"""
from .base import Backend
from .z3_backend import Z3Backend
from .mock_backend import MockBackend

# Default to Z3 if available, otherwise use Mock
try:
    import z3
    default_backend = Z3Backend
except ImportError:
    default_backend = MockBackend

__all__ = ['Backend', 'Z3Backend', 'MockBackend', 'default_backend']
