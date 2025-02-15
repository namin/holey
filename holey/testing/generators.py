from typing import List
from .systematic import TestCase
from ..llm import generate as llm_generate

class TestGenerator:
    def generate_string_puzzles(self) -> List[TestCase]:
        """Generate comprehensive string puzzle test cases"""
        base_cases = [
            # Length constraints
            "def sat(s: str): return len(s) == 5",
            # Character counting
            "def sat(s: str): return s.count('a') == 3",
            # Substring constraints
            "def sat(s: str): return 'hello' in s",
            # Multiple constraints
            "def sat(s: str): return len(s) == 5 and s.count('a') == 2",
        ]
        
        variations = []
        for case in base_cases:
            variations.extend(self._generate_variations(case))
            
        return variations

    def generate_numeric_puzzles(self) -> List[TestCase]:
        """Generate comprehensive numeric puzzle test cases"""
        base_cases = [
            # Basic arithmetic
            "def sat(x: int): return x * 2 == 10",
            # Multiple constraints
            "def sat(x: int): return x % 3 == 0 and x < 100",
            # Complex expressions
            "def sat(x: int): return x**2 + x - 12 == 0",
        ]
        
        variations = []
        for case in base_cases:
            variations.extend(self._generate_variations(case))
            
        return variations

    def _generate_variations(self, base_case: str) -> List[TestCase]:
        """Generate variations of a base case"""
        prompt = f"""Given this base puzzle:
{base_case}

Generate 5 variations by:
1. Changing constants
2. Adding additional constraints
3. Modifying operations
4. Varying complexity

Return each variation as a complete sat function."""

        variations = llm_generate(prompt).strip().split('\n\n')
        return [TestCase(v, self._infer_type(v), f"var_{i}")
                for i, v in enumerate(variations)]

    def _infer_type(self, sat_func: str) -> str:
        """Infer answer type from sat function"""
        if 'str' in sat_func.split('(')[1].split(')')[0]:
            return 'str'
        return 'int'

class TestTypes:
    @staticmethod
    def generate_regression_suite() -> List[TestCase]:
        """Generate regression test suite"""
        return [
            TestCase(
                "def sat(s: str): return s.count('a') == 5",
                "str",
                "basic_count",
                difficulty=1.0
            ),
            TestCase(
                "def sat(s: str): return len(s) == 10 and s.count('a') == 3",
                "str",
                "length_and_count",
                difficulty=2.0
            ),
            TestCase(
                "def sat(x: int): return x > 0 and x % 7 == 0 and x < 100",
                "int", 
                "basic_arithmetic",
                difficulty=1.0
            ),
        ]

    @staticmethod
    def generate_edge_cases() -> List[TestCase]:
        """Generate edge case test suite"""
        return [
            TestCase(
                "def sat(s: str): return len(s) == 0",
                "str",
                "empty_string",
                difficulty=1.0
            ),
            TestCase(
                "def sat(s: str): return len(s) == 1000",
                "str",
                "very_long_string",
                difficulty=4.0
            ),
            TestCase(
                "def sat(x: int): return x == 2**31 - 1",
                "int",
                "max_int32",
                difficulty=2.0
            ),
        ]

    @staticmethod
    def generate_performance_tests() -> List[TestCase]:
        """Generate performance test suite"""
        return [
            TestCase(
                "def sat(s: str): return len(s) == 100 and s.count('a') * 2 == s.count('b')",
                "str",
                "large_constraint",
                difficulty=3.0
            ),
            TestCase(
                "def sat(x: int): return x > 0 and all(x % i != 0 for i in range(2, 100))",
                "int",
                "prime_check",
                difficulty=4.0
            ),
        ]