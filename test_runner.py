#!/usr/bin/env python3
from holey.testing.driver import TestDriver
from holey.testing.generators import TestGenerator, TestTypes
import argparse
import logging
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run systematic tests for holey puzzle solver')
    parser.add_argument('--output-dir', default='test_output', help='Output directory for results')
    parser.add_argument('--attempts', type=int, default=3, help='Number of attempts per test case')
    parser.add_argument('--test-types', nargs='+', 
                       choices=['string', 'numeric', 'regression', 'edge', 'performance'],
                       default=['regression', 'edge'],
                       help='Types of tests to run')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    args = parser.parse_args()

    # Initialize test driver
    driver = TestDriver(
        output_dir=args.output_dir,
        log_level=getattr(logging, args.log_level)
    )

    # Generate test cases
    generator = TestGenerator()
    test_cases = []
    
    if 'string' in args.test_types:
        test_cases.extend(generator.generate_string_puzzles())
        
    if 'numeric' in args.test_types:
        test_cases.extend(generator.generate_numeric_puzzles())
        
    if 'regression' in args.test_types:
        test_cases.extend(TestTypes.generate_regression_suite())
        
    if 'edge' in args.test_types:
        test_cases.extend(TestTypes.generate_edge_cases())
        
    if 'performance' in args.test_types:
        test_cases.extend(TestTypes.generate_performance_tests())

    # Run tests
    try:
        results = driver.run_test_suite(
            test_cases,
            tag=f"test_run_{datetime.now():%Y%m%d_%H%M%S}"
        )
        
        # Print summary to console
        print("\nTest Run Summary")
        print("=" * 50)
        
        for strategy, stats in results['by_strategy'].items():
            print(f"\n{strategy.upper()} Strategy:")
            print(f"Success Rate: {stats['success_rate']:.2%}")
            print(f"Average Time: {stats['avg_time']:.2f}s")
            print(f"Total Attempts: {stats['total_attempts']}")
        
        if results.get('improvements'):
            print("\nImprovements from last run:")
            for category, changes in results['improvements'].items():
                if isinstance(changes, dict):
                    print(f"\n{category}:")
                    for strategy, data in changes.items():
                        print(f"- {strategy}: {data['status']} ({data['change']:+.2%})")
                else:
                    print(f"Overall: {changes}")

    except Exception as e:
        logging.error(f"Test run failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()