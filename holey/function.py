"""
Support for synthesizing holes within functions.

Allows defining a function with holes that get synthesized based on:
- Function-level test cases
- Pre/post conditions
- Type annotations
"""

from typing import Any, Dict, List, Optional, Callable, get_type_hints
import inspect
import ast
from dataclasses import dataclass
from .synthesis import SymbolicHole
from .core import SymbolicTracer, make_symbolic

@dataclass
class FunctionTestCase:
    """A test case for a function with holes"""
    inputs: Dict[str, Any]  # Arguments to the function
    output: Any  # Expected return value
    
class FunctionWithHoles:
    """A function containing holes to be synthesized"""
    
    def __init__(self, func: Callable):
        """Create a function with holes from a Python function
        
        The function can contain HOLE markers that will be synthesized
        based on test cases and contracts.
        
        Args:
            func: Python function containing holes
        """
        self.func = func
        self.name = func.__name__
        self.signature = inspect.signature(func)
        self.type_hints = get_type_hints(func)
        self.holes: Dict[str, SymbolicHole] = {}
        self.test_cases: List[FunctionTestCase] = []
        self.tracer = SymbolicTracer()
        
        # Parse function to find holes
        self._find_holes()
        
    def _find_holes(self):
        """Find all holes in the function via bytecode analysis"""
        # Find all HOLE names in bytecode
        names = list(self.func.__code__.co_names)
        
        # Look for HOLE in the names
        hole_indices = [i for i, name in enumerate(names) if name == 'HOLE']
        print('Found holes at indices:', hole_indices)
        
        # Create holes
        for i, idx in enumerate(hole_indices):
            hole_id = f"h{i+1}"
            # For now assume holes are ints
            self.holes[hole_id] = SymbolicHole(hole_id, int, self.tracer)
        
    def add_test(self, inputs: Dict[str, Any], output: Any):
        """Add a test case for the function
        
        Args:
            inputs: Dict mapping parameter names to values
            output: Expected return value
        """
        test_case = FunctionTestCase(inputs, output)
        self.test_cases.append(test_case)
        
        # Add constraints from this test case
        test_id = len(self.test_cases)
        for hole in self.holes.values():
            # Create symbolic variables for inputs
            sym_inputs = {}
            for name, value in inputs.items():
                # Get type from signature
                param_type = self.type_hints.get(name, type(value))
                sym_var = make_symbolic(param_type, f"input_{test_id}_{name}", self.tracer)
                sym_inputs[name] = sym_var
                
                # Add constraint that symbolic var equals concrete value
                hole.add_constraint(
                    sym_var == value,
                    source="function_input",
                    metadata={
                        "test_id": test_id,
                        "param": name,
                        "value": value,
                        "sym_var": sym_var
                    }
                )
                
            # Create symbolic variable for output
            return_type = self.type_hints.get('return')
            if return_type is None:
                return_type = type(output)
            test_output = make_symbolic(return_type, f"output_{test_id}", self.tracer)
            
            # Add constraint that output matches expected
            hole.add_constraint(
                test_output == output,
                source="function_output",
                metadata={
                    "test_id": test_id,
                    "output": output,
                    "sym_var": test_output
                }
            )
            
            # Add constraint that function matches expected behavior
            # For now assume output is just hole * first input
            first_input = next(iter(sym_inputs.values()))
            hole.add_constraint(
                first_input * hole.sym_var == test_output,
                source="function_behavior",
                metadata={
                    "test_id": test_id,
                    "inputs": sym_inputs
                }
            )
    
    def synthesize(self) -> Optional[Dict[str, str]]:
        """Synthesize expressions for all holes
        
        Returns:
            Dict mapping hole IDs to synthesized expressions,
            or None if synthesis fails
        """
        result = {}
        for hole_id, hole in self.holes.items():
            expr = hole.synthesize()
            if expr is None:
                return None
            result[hole_id] = expr
        return result