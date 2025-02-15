# Systematic Testing Framework for Holey

This testing framework provides comprehensive testing capabilities for the Holey symbolic execution and synthesis system. The framework focuses on systematically evaluating and improving solving strategies by combining symbolic execution, SMT solving, and LLM approaches.

## Key Components

### 1. Test Case Generation
- Regression test suite for core functionality
- Edge case generation for boundary conditions
- Performance tests for scalability
- Automatically generated variations of base cases

### 2. Solving Strategies
The framework implements multiple solving strategies:
- Pure symbolic execution using Z3
- LLM-based solving
- Hybrid approaches combining symbolic and LLM methods
- Progressive timeout and complexity management
- Decomposition of complex problems into simpler sub-problems

### 3. Analysis & Monitoring
- Success rate tracking per strategy
- Failure pattern analysis
- Performance metrics
- Strategy effectiveness measurement
- Learning from successful patterns

### 4. Key Files

- `systematic.py`: Core testing infrastructure
- `solvers.py`: Implementation of different solving strategies
- `generators.py`: Test case generation
- `analysis.py`: Results analysis and visualization

## Running Tests

Basic usage:
```bash
python test_runner.py --attempts 3 --test-types regression edge
```

Options:
- `--attempts`: Number of attempts per test case
- `--test-types`: Types of tests to run (regression, edge, performance)
- `--output`: Output file for detailed results
- `--log-dir`: Directory for detailed logs

## Strategy Selection

The framework automatically selects solving strategies based on problem characteristics:

1. Simple Constraints
   - Length checks
   - Basic counting
   - Simple arithmetic
   → Use symbolic execution

2. Large-Scale Problems
   - Long strings
   - Complex arithmetic
   → Start with LLM approach

3. Mixed Constraints
   - Multiple conditions
   - Complex patterns
   → Use hybrid approach

## Learning System

The framework includes a learning component that:
1. Tracks successful solving patterns
2. Updates strategy weights based on performance
3. Adapts to different types of problems
4. Provides feedback for strategy improvement

## Results Analysis

Test runs generate:
1. Success rates per strategy
2. Timing analysis
3. Failure pattern identification
4. Performance bottleneck detection
5. Strategy effectiveness metrics

## Extension Points

The framework is designed to be extensible:
1. Add new solving strategies
2. Define custom test generators
3. Implement specialized analysis
4. Create new hybrid approaches

## Current Status

Based on initial testing:
- Symbolic execution: Good for simple constraints
- LLM approach: Helpful for creative solutions
- Hybrid methods: Most promising for complex cases
- Main challenge: Smooth integration between approaches