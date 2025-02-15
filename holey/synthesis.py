"""
Support for synthesizing expressions to fill holes in Python code.

A hole represents a missing expression that needs to be synthesized.
Constraints on the hole can come from:
- Test cases showing expected behavior
- Type annotations and contracts
- Runtime assertions
- Context and surrounding code
"""

from typing import Any, Dict, List, Optional, Set, Type
from dataclasses import dataclass, field
from .core import SymbolicTracer, SymbolicInt, SymbolicBool, SymbolicStr, make_symbolic

@dataclass
class HoleConstraint:
    """A constraint on what a hole must satisfy"""
    condition: Any  # SymbolicBool expression
    source: str  # Where this constraint came from (test, contract, etc)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SymbolicHole:
    """A hole in code that needs to be synthesized"""
    
    def __init__(self, 
                 name: str,
                 type_hint: Type,
                 tracer: Optional[SymbolicTracer] = None):
        """Initialize a new hole
        
        Args:
            name: Name to identify this hole
            type_hint: Expected type of the synthesized expression
            tracer: Optional symbolic tracer to use
        """
        self.name = name
        self.type_hint = type_hint
        self.tracer = tracer or SymbolicTracer()
        
        # Create symbolic variable to represent the hole
        self.sym_var = make_symbolic(type_hint, f"hole_{name}", self.tracer)
        
        # Track all constraints on this hole
        self.constraints: List[HoleConstraint] = []
        
        # Track variables available in scope
        self.available_vars: Set[str] = set()
        
        # Cache for generated expressions
        self.candidate_cache: Dict[str, Any] = {}

    def add_constraint(self, 
                      condition: Any,
                      source: str = "unknown",
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new constraint on what the hole must satisfy
        
        Args:
            condition: Symbolic expression that must be true
            source: Where this constraint came from
            metadata: Additional info about the constraint
        """
        metadata = metadata or {}
        constraint = HoleConstraint(condition, source, metadata)
        self.constraints.append(constraint)
        
        # Add to solver
        self.tracer.add_constraint(condition.z3_expr)

    def add_test_case(self, inputs: Dict[str, Any], expected_output: Any) -> None:
        """Add a test case for the hole's behavior
        
        For a hole that should behave like a function, test cases specify
        input-output pairs that constrain the function's behavior.
        
        Args:
            inputs: Dict mapping input variable names to values
            expected_output: Expected result for these inputs
        """
        # Create a fresh version of input variables for this test case
        input_vars = {}
        test_id = len(self.constraints)
        for name, value in inputs.items():
            var_type = type(value)
            sym_var = make_symbolic(var_type, f"test_{test_id}_{name}", self.tracer)
            input_vars[name] = sym_var
            self.available_vars.add(name)
            
            # Record input constraint
            self.add_constraint(
                sym_var == value,
                source="test_input",
                metadata={
                    "test_id": test_id,
                    "var": name,
                    "value": value,
                    "sym_var": sym_var
                }
            )
        
        # Create a fresh version of the hole for this test case
        test_hole = make_symbolic(self.type_hint, f"hole_{test_id}", self.tracer)
        
        # Add dependency between test input variables and test hole
        self.add_constraint(
            test_hole == expected_output,
            source="test_case",
            metadata={
                "test_id": test_id,
                "inputs": input_vars,
                "expected": expected_output,
                "test_hole": test_hole
            }
        )
        
        # Add functional constraint - hole must give same output when given same inputs
        for prev_constraint in self.constraints:
            if prev_constraint.source == "test_case":
                prev_inputs = prev_constraint.metadata["inputs"]
                prev_test_hole = prev_constraint.metadata["test_hole"]
                
                # If all inputs match, outputs must match
                input_match = None
                for name, var in input_vars.items():
                    input_eq = (var == prev_inputs[name])
                    if input_match is None:
                        input_match = input_eq
                    else:
                        input_match = input_match & input_eq
                
                if input_match is not None:
                    self.add_constraint(
                        input_match.implies(test_hole == prev_test_hole),
                        source="functional",
                        metadata={"test_id": test_id}
                    )

    def add_type_constraint(self, type_pred: Any) -> None:
        """Add type-based constraint on the hole
        
        Args:
            type_pred: Symbolic predicate that must be true for the type
        """
        self.add_constraint(type_pred, source="type")

    def synthesize(self, max_attempts: int = 10) -> Optional[str]:
        """Try to synthesize an expression that satisfies all constraints
        
        Args:
            max_attempts: Maximum synthesis attempts before giving up
            
        Returns:
            A string containing a valid Python expression, or None if synthesis fails
        """
        # First try SMT-based synthesis
        expr = self._smt_synthesis()
        if expr:
            return expr
            
        # If that fails, try LLM-guided synthesis
        if self.tracer.llm_solver:
            for _ in range(max_attempts):
                expr = self._llm_synthesis()
                if expr and self._verify_expression(expr):
                    return expr
                    
        return None

    def _smt_synthesis(self) -> Optional[str]:
        """Try to synthesize using SMT solver
        
        Handles two cases:
        1. Basic holes - direct synthesis from constraints
        2. Function holes - synthesize from input/output examples
        """
        # Check if constraints are satisfiable
        result = self.tracer.check()
        if not self.tracer.backend.is_sat(result):
            return None
            
        # Get model
        model = self.tracer.model()
        if not model:
            return None
            
        # Try function synthesis if we have test cases
        test_cases = [(c.metadata["inputs"], c.metadata["expected"]) 
                      for c in self.constraints 
                      if c.source == "test_case"]
                      
        if test_cases:
            return self._synthesize_function(test_cases, model)
            
        # For basic holes, just get the value
        value = model[self.sym_var.name]
        if value is None:
            return None
            
        # Convert Z3 model values to Python values
        try:
            if self.type_hint == int:
                return str(getattr(value, 'as_long', lambda: value)())
            elif self.type_hint == str:
                if isinstance(value, str):
                    return f'"{value}"'
                return f'"{value.as_string()}"'
            elif self.type_hint == bool:
                return str(bool(value)).lower()
        except Exception as e:
            print(f"Error converting value {value}: {e}")
            return None
        return None

    def _synthesize_function(self, test_cases, model) -> Optional[str]:
        """Synthesize a function expression from test cases"""
        if not test_cases:
            return None
            
        first_case = test_cases[0]
        if not first_case[0]:  # No inputs
            return None
            
        # Handle string holes specially to avoid integer conversion
        if self.type_hint == str:
            input_names = list(first_case[0].keys())
            if len(input_names) == 2:
                # Try simple concatenation of two strings
                name1, name2 = input_names
                # Check all test cases for concatenation pattern
                is_concat = True
                for inputs, expected in test_cases:
                    val1 = model[inputs[name1].name]
                    val2 = model[inputs[name2].name]
                    
                    # Convert string values
                    if hasattr(val1, 'as_string'):
                        val1 = val1.as_string()
                    elif isinstance(val1, str):
                        val1 = val1
                        
                    if hasattr(val2, 'as_string'):
                        val2 = val2.as_string()
                    elif isinstance(val2, str):
                        val2 = val2
                        
                    # Get expected value
                    if hasattr(expected, 'name'):
                        expected_val = model[expected.name]
                        if hasattr(expected_val, 'as_string'):
                            expected_val = expected_val.as_string()
                        elif isinstance(expected_val, str):
                            expected_val = expected_val
                    else:
                        expected_val = expected
                        
                    if val1 + val2 != expected_val:
                        is_concat = False
                        break
                        
                if is_concat:
                    return f"{name1} + {name2}"
            return None
            
        # For numeric holes, handle arithmetic relationships
        input_name = next(iter(first_case[0].keys()))
        pairs = []
        
        for inputs, expected in test_cases:
            # Get input value
            input_val = model[inputs[input_name].name]
            if hasattr(input_val, 'as_long'):
                input_val = input_val.as_long()
            elif isinstance(input_val, (int, float)):
                input_val = int(input_val)
            # Get output value
            if isinstance(expected, (int, float)):
                output_val = int(expected)
            elif hasattr(expected, 'name'):
                output_val = model[expected.name]
                if hasattr(output_val, 'as_long'):
                    output_val = output_val.as_long()
                elif isinstance(output_val, (int, float)):
                    output_val = int(output_val)
                else:
                    return None
            else:
                output_val = expected
            pairs.append((input_val, output_val))
            
        # Try basic arithmetic relationships
        if not pairs:
            return None
            
        first_in, first_out = pairs[0]
        
        # Try multiplication
        if first_in != 0 and first_out % first_in == 0:
            factor = first_out // first_in
            expr = f"{input_name} * {factor}"
            if all(in_val * factor == out_val for in_val, out_val in pairs):
                return expr
                
        # Try absolute value
        if self.type_hint == int and all(abs(in_val) == out_val for in_val, out_val in pairs):
            return f"abs({input_name})"
            
        return None

    def _llm_synthesis(self) -> Optional[str]:
        """Try to synthesize using LLM guidance"""
        if not self.tracer.llm_solver:
            return None
            
        prompt = self._generate_synthesis_prompt()
        try:
            result = self.tracer.llm_solver.llm_generate(prompt)
            expr = self._extract_expression(result)
            return expr
        except:
            return None

    def _verify_expression(self, expr: str) -> bool:
        """Verify if an expression satisfies all constraints"""
        try:
            # Parse expression
            parsed = eval(expr)
            
            # Check type
            if not isinstance(parsed, self.type_hint):
                return False
                
            # Add constraint that hole equals this expression
            result = self.tracer.check()
            return self.tracer.backend.is_sat(result)
            
        except:
            return False

    def _convert_to_expression(self, value: Any) -> str:
        """Convert an SMT value to a Python expression string"""
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return f"'{value}'"
        else:
            raise ValueError(f"Can't convert {value} to expression")

    def _extract_expression(self, llm_output: str) -> Optional[str]:
        """Extract expression from LLM output"""
        # Remove any explanation text
        lines = llm_output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Basic validation that it's a single expression
                try:
                    compile(line, '<string>', 'eval')
                    return line
                except:
                    continue
        return None