"""
Solver implementations for the systematic testing framework.
Includes progressive solving strategies combining symbolic execution and LLMs.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import time
from func_timeout import func_timeout, FunctionTimedOut
import re
from ..core import SymbolicTracer
from ..llm import generate as llm_generate
from ..backend import run_smt
from .systematic import TestCase

@dataclass
class SolverState:
    """Track solver state and history"""
    smt_queries: List[str] = None
    timeouts: int = 0
    failed_constraints: List[str] = None
    successful_patterns: Dict[str, int] = None

class Strategy:
    """Base class for solving strategies"""
    def __init__(self):
        self.weight = 1.0
        self.successes = 0
        self.attempts = 0

    def simplify(self, constraints: Dict[str, Any]) -> str:
        """Convert constraints to SMT format with strategy-specific optimizations"""
        raise NotImplementedError

    def get_pattern(self) -> str:
        """Get string identifier for this strategy's pattern"""
        raise NotImplementedError

class DirectStrategy(Strategy):
    """Direct SMT encoding without optimizations"""
    def simplify(self, constraints: Dict[str, Any]) -> str:
        smt = ["(set-logic ALL)"]
        
        # Declare variables
        for var, type_info in constraints['variables'].items():
            smt.append(f"(declare-const {var} {type_info['type']})")
            
        # Add constraints
        for constraint in constraints['operations']:
            smt.append(f"(assert {constraint})")
            
        smt.append("(check-sat)")
        smt.append("(get-model)")
        
        return "\n".join(smt)

    def get_pattern(self) -> str:
        return "direct_smt"

class BoundedStrategy(Strategy):
    """Add bounds on variables for better solving"""
    def simplify(self, constraints: Dict[str, Any]) -> str:
        smt = ["(set-logic ALL)"]
        
        # Declare variables with bounds
        for var, type_info in constraints['variables'].items():
            smt.append(f"(declare-const {var} {type_info['type']})")
            if type_info['type'] == 'Int':
                smt.append(f"(assert (>= {var} -1000))")
                smt.append(f"(assert (<= {var} 1000))")
            elif type_info['type'] == 'String':
                smt.append(f"(assert (<= (str.len {var}) 1000))")
                
        # Add original constraints
        for constraint in constraints['operations']:
            smt.append(f"(assert {constraint})")
            
        smt.append("(check-sat)")
        smt.append("(get-model)")
        
        return "\n".join(smt)

    def get_pattern(self) -> str:
        return "bounded_smt"

class ProgressiveSolver:
    def __init__(self):
        self.state = SolverState()
        self.successful_patterns = {}
        self.strategies = [
            DirectStrategy(),
            BoundedStrategy()
        ]

    def symbolic_solve_progressive(self, test_case: TestCase) -> Optional[Any]:
        """Progressive SMT solving with increasing complexity"""
        constraints = self._analyze_constraints(test_case.sat_func)
        
        # Try each strategy in order of weight
        sorted_strategies = sorted(
            self.strategies,
            key=lambda s: s.weight,
            reverse=True
        )
        
        for strategy in sorted_strategies:
            try:
                simplified = strategy.simplify(constraints)
                result = self._try_solve_with_timeout(simplified)
                if result:
                    self._record_success(strategy, constraints)
                    return result
            except TimeoutError:
                self.state.timeouts += 1
                continue
                
        return None

    def _try_solve_with_timeout(self, constraints: str, 
                              base_timeout: int = 1) -> Optional[Any]:
        """Try solving with increasing timeouts"""
        for multiplier in [1, 2, 4]:
            timeout = base_timeout * multiplier
            try:
                result = func_timeout(
                    timeout,
                    self._run_smt,
                    args=(constraints,)
                )
                if result:
                    return result
            except FunctionTimedOut:
                self.state.timeouts += 1
        return None

    def _run_smt(self, constraints: str) -> Optional[Any]:
        """Run SMT solver with given constraints"""
        self.state.smt_queries.append(constraints)
        result, model = run_smt(constraints)
        if result == "sat":
            return model
        return None

    def _analyze_constraints(self, sat_func: str) -> Dict[str, Any]:
        """Analyze constraints for optimization opportunities"""
        analysis = {
            'variables': self._extract_variables(sat_func),
            'operations': self._extract_operations(sat_func),
            'constants': self._extract_constants(sat_func),
            'patterns': self._identify_patterns(sat_func)
        }
        
        # Identify constraint types
        analysis['types'] = {
            'has_strings': 'str' in sat_func,
            'has_arithmetic': any(op in sat_func for op in ['+', '-', '*', '/']),
            'has_comparisons': any(op in sat_func for op in ['<', '>', '==']),
            'has_modulo': '%' in sat_func
        }
        
        return analysis

    def _extract_variables(self, sat_func: str) -> Dict[str, Dict[str, str]]:
        """Extract variables and their types from function"""
        vars = {}
        
        # Parse function signature
        sig_match = re.search(r'def sat\((.*?)\):', sat_func)
        if sig_match:
            params = sig_match.group(1).split(',')
            for param in params:
                name, type_hint = param.strip().split(':')
                type_hint = type_hint.strip()
                if type_hint == 'str':
                    vars[name] = {'type': 'String'}
                elif type_hint == 'int':
                    vars[name] = {'type': 'Int'}
                    
        return vars

    def _extract_operations(self, sat_func: str) -> List[str]:
        """Extract SMT operations from function body"""
        ops = []
        
        # Get function body
        body = sat_func.split(':')[1].strip()
        
        # Convert Python operations to SMT
        if 'len(' in body:
            ops.append('str.len')
        if 'count(' in body:
            ops.append('str.count')
        if '%' in body:
            ops.append('mod')
            
        return ops

    def _extract_constants(self, sat_func: str) -> Dict[str, Any]:
        """Extract constant values from function"""
        constants = {
            'integers': [],
            'strings': []
        }
        
        # Find integers
        constants['integers'].extend(
            int(n) for n in re.findall(r'\d+', sat_func)
        )
        
        # Find strings
        constants['strings'].extend(
            s[1:-1] for s in re.findall(r"'([^']*)'", sat_func)
        )
        
        return constants

    def _identify_patterns(self, sat_func: str) -> List[str]:
        """Identify common constraint patterns"""
        patterns = []
        
        if 'len(' in sat_func and '==' in sat_func:
            patterns.append('length_equality')
            
        if 'count(' in sat_func:
            patterns.append('string_counting')
            
        if '%' in sat_func:
            patterns.append('modulo')
            
        return patterns

    def _record_success(self, strategy: Strategy, constraints: Dict[str, Any]):
        """Record successful solving pattern"""
        pattern = strategy.get_pattern()
        self.successful_patterns[pattern] = \
            self.successful_patterns.get(pattern, 0) + 1
        
        strategy.successes += 1
        strategy.attempts += 1
        strategy.weight = strategy.successes / strategy.attempts

    def update_strategy_weights(self):
        """Update strategy weights based on success patterns"""
        stats = self._analyze_solving_patterns()
        
        # Adjust strategy selection based on success rates
        for strategy in self.strategies:
            pattern = strategy.get_pattern()
            if pattern in stats:
                strategy.weight = stats[pattern]

    def _analyze_solving_patterns(self) -> Dict[str, float]:
        """Analyze which solving patterns work best"""
        stats = {}
        
        for pattern, successes in self.successful_patterns.items():
            total = len(self.state.smt_queries)
            success_rate = successes / total if total > 0 else 0
            stats[pattern] = success_rate
            
        return stats