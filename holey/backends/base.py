"""
Base backend implementation.
"""
from typing import Any, Optional

class Backend:
    def __init__(self):
        self.base_backend = self

    def get_base_backend(self):
        """Get the underlying base backend for low-level operations"""
        return self.base_backend
