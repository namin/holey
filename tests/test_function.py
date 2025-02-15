"""
Tests for function hole synthesis.
"""

from holey.function import FunctionWithHoles

def test_basic_function():
    """Test basic function with one hole"""
    
    def double_with_hole(x: int) -> int:
        # TODO: Actually support HOLE syntax
        # return HOLE * x
        return x  # Just placeholder until we implement HOLE parsing
        
    func = FunctionWithHoles(double_with_hole)
    
    # Add test cases
    func.add_test({"x": 1}, 2)
    func.add_test({"x": 2}, 4)
    func.add_test({"x": 3}, 6)
    
    # Synthesize holes
    result = func.synthesize()
    assert result is not None
    assert result["h1"] == "2"  # Should synthesize multiplication by 2
    
def test_multiple_calls():
    """Test function called with multiple args"""
    
    def triple_with_hole(x: int) -> int:
        # TODO: Actually support HOLE syntax
        # return HOLE * x
        return x  # Just placeholder until we implement HOLE parsing
        
    func = FunctionWithHoles(triple_with_hole)
    
    # Add test cases
    func.add_test({"x": 1}, 3)
    func.add_test({"x": 2}, 6)
    func.add_test({"x": 4}, 12)
    
    # Synthesize holes
    result = func.synthesize()
    assert result is not None
    assert result["h1"] == "3"  # Should synthesize multiplication by 3