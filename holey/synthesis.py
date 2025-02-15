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
        """Add a test case that the synthesized expression must satisfy
        
        Args:
            inputs: Dict mapping input variable names to values
            expected_output: Expected result for these inputs
        """
        # Create symbolic variables for inputs
        input_vars = {}
        for name, value in inputs.items():
            var_type = type(value)
            sym_var = make_symbolic(var_type, f"test_{name}", self.tracer)
            input_vars[name] = sym_var
            
            # Track available variables
            self.available_vars.add(name)
            
            # Add constraint that symbolic var equals concrete value
            self.add_constraint(
                sym_var == value,
                source="test_input",
                metadata={"var": name, "value": value}
            )
            
        # Add constraint that hole equals expected output
        output_type = type(expected_output)
        if output_type != self.type_hint:
            raise TypeError(f"Expected output type {output_type} doesn't match hole type {self.type_hint}")
            
        self.add_constraint(
            self.sym_var == expected_output,
            source="test_output",
            metadata={"expected": expected_output}
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
        """Try to synthesize using SMT solver"""
        # Check if constraints are satisfiable
        if not self.tracer.backend.is_sat(self.tracer.check()):
            return None
            
        # Get model
        model = self.tracer.model()
        if not model:
            return None
            
        # Extract value for hole
        value = model[self.sym_var.name]
        if value is None:
            return None
            
        # Convert to Python expression
        return self._convert_to_expression(value)

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

    def _generate_synthesis_prompt(self) -> str:
        """Generate prompt for LLM synthesis"""
        prompt = [
            f"Synthesize a Python expression of type {self.type_hint.__name__}",
            "that satisfies these constraints:\n"
        ]
        
        # Add constraints
        for c in self.constraints:
            prompt.append(f"- {c.condition} (from {c.source})")
            
        # Add available variables
        if self.available_vars:
            prompt.append("\nAvailable variables:")
            for var in sorted(self.available_vars):
                prompt.append(f"- {var}")
                
        return "\n".join(prompt)

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