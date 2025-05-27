"""
Bounded array encoding for lists to avoid recursive definitions
"""

class BoundedListEncoding:
    """Encode lists as bounded arrays with explicit length"""
    
    def __init__(self, backend, name, element_type, max_length=100):
        self.backend = backend
        self.name = name
        self.element_type = element_type
        self.max_length = max_length
        
        # Create array representation
        self.length_var = backend.Int(f"{name}_length")
        
        # Store elements as individual variables for better solver performance
        self.elements = []
        for i in range(max_length):
            if element_type == "Int":
                self.elements.append(backend.Int(f"{name}_elem_{i}"))
            elif element_type == "String":
                self.elements.append(backend.String(f"{name}_elem_{i}"))
            elif element_type == "Bool":
                self.elements.append(backend.Bool(f"{name}_elem_{i}"))
            elif element_type == "Real":
                self.elements.append(backend.Real(f"{name}_elem_{i}"))
        
        # Add basic constraints
        backend.add(backend.And(
            self.length_var >= 0,
            self.length_var <= max_length
        ))
    
    def get(self, index):
        """Get element at index (bounded)"""
        # Build cascading if-then-else for concrete index
        if hasattr(index, 'concrete') and index.concrete is not None:
            idx = index.concrete
            if 0 <= idx < self.max_length:
                return self.elements[idx]
        
        # For symbolic index, create if-then-else chain
        result = self.elements[0]  # default
        for i in range(self.max_length):
            result = self.backend.If(
                self.backend.Eq(index, self.backend.IntVal(i)),
                self.elements[i],
                result
            )
        return result
    
    def count(self, value):
        """Count occurrences of value"""
        # Sum up matches
        count_terms = []
        for i in range(self.max_length):
            count_terms.append(
                self.backend.If(
                    self.backend.And(
                        self.backend.LT(self.backend.IntVal(i), self.length_var),
                        self.backend.Eq(self.elements[i], value)
                    ),
                    self.backend.IntVal(1),
                    self.backend.IntVal(0)
                )
            )
        
        # Build sum tree to avoid deep nesting
        return self._build_sum_tree(count_terms)
    
    def _build_sum_tree(self, terms):
        """Build balanced sum tree to avoid deep nesting"""
        if len(terms) == 0:
            return self.backend.IntVal(0)
        if len(terms) == 1:
            return terms[0]
        
        # Pair up terms
        new_terms = []
        for i in range(0, len(terms), 2):
            if i + 1 < len(terms):
                new_terms.append(self.backend.Add(terms[i], terms[i+1]))
            else:
                new_terms.append(terms[i])
        
        return self._build_sum_tree(new_terms)
    
    def length(self):
        """Get list length"""
        return self.length_var
    
    def set_concrete_list(self, concrete_list):
        """Set constraints for a concrete list"""
        self.backend.add(self.length_var == self.backend.IntVal(len(concrete_list)))
        
        for i, val in enumerate(concrete_list):
            if i < self.max_length:
                if self.element_type == "Int":
                    self.backend.add(self.elements[i] == self.backend.IntVal(val))
                elif self.element_type == "String":
                    self.backend.add(self.elements[i] == self.backend.StringVal(val))
                elif self.element_type == "Bool":
                    self.backend.add(self.elements[i] == self.backend.BoolVal(val))
    
    def slice(self, start, stop, step=1):
        """Handle list slicing (simplified)"""
        # For now, return a new bounded list with appropriate constraints
        new_list = BoundedListEncoding(
            self.backend, 
            f"{self.name}_slice", 
            self.element_type,
            self.max_length
        )
        
        # Add slicing constraints
        # This is a simplified version - full slicing logic would be more complex
        return new_list
    
    def to_smt2_list(self):
        """Convert to SMT2 list representation for final output"""
        # Build nested cons structure
        result = "(as nil (List " + self.element_type + "))"
        
        # Build from back to front
        for i in range(self.max_length - 1, -1, -1):
            result = self.backend.If(
                self.backend.LT(self.backend.IntVal(i), self.length_var),
                f"(cons {self.elements[i].to_smt2()} {result})",
                result
            )
        
        return result


def create_bounded_list(backend, name, element_type, max_length=None):
    """Factory function to create bounded list encoding"""
    if max_length is None:
        # Heuristic: use reasonable default based on problem
        max_length = 100
    
    return BoundedListEncoding(backend, name, element_type, max_length)