#!/usr/bin/env python3
from holey.testing import SystematicTester, TestGenerator, TestTypes
from holey import LLMSolver
import json
import argparse
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Run systematic tests for holey puzzle solver')
    parser.add_argument('--output', default='test_results.json', help='Output file for test results')
    parser.add_argument('--attempts', type=int, default=10, help='Number of attempts per test case')
    parser.add_argument('--test-types', nargs='+', 
                       choices=['string', 'numeric', 'regression', 'edge', 'performance'],
                       default=['string', 'numeric', 'regression'],
                       help='Types of tests to run')
    parser.add_argument('--log-dir', default='logs', help='Directory for detailed logs')
    args = parser.parse_args()

    # Setup logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'test_run_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logging.info(f"Starting test run with {args.attempts} attempts per case")

    # Initialize components
    solver = LLMSolver()
    tester = SystematicTester(solver, attempts_per_case=args.attempts)
    generator = TestGenerator()
    
    # Generate test cases based on selected types
    all_tests = []
    
    if 'string' in args.test_types:
        string_puzzles = generator.generate_string_puzzles()
        all_tests.extend(string_puzzles)
        logging.info(f"Generated {len(string_puzzles)} string puzzles")
        
    if 'numeric' in args.test_types:
        numeric_puzzles = generator.generate_numeric_puzzles()
        all_tests.extend(numeric_puzzles)
        logging.info(f"Generated {len(numeric_puzzles)} numeric puzzles")
        
    if 'regression' in args.test_types:
        regression_suite = TestTypes.generate_regression_suite()
        all_tests.extend(regression_suite)
        logging.info(f"Added {len(regression_suite)} regression tests")
        
    if 'edge' in args.test_types:
        edge_cases = TestTypes.generate_edge_cases()
        all_tests.extend(edge_cases)
        logging.info(f"Added {len(edge_cases)} edge cases")
        
    if 'performance' in args.test_types:
        performance_tests = TestTypes.generate_performance_tests()
        all_tests.extend(performance_tests)
        logging.info(f"Added {len(performance_tests)} performance tests")

    logging.info(f"Running {len(all_tests)} total test cases")
    
    # Run tests
    try:
        results = tester.run_reliability_test(all_tests)
        
        # Generate report
        report = tester.generate_report()
        report_file = log_dir / f'report_{timestamp}.md'
        report_file.write_text(report)
        logging.info(f"Written test report to {report_file}")
        
        # Save detailed results
        results_file = Path(args.output)
        results_file.parent.mkdir(exist_ok=True)
        with results_file.open('w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Written detailed results to {args.output}")
        
        # Print summary
        print("\nTest Run Summary:")
        print("=" * 50)
        for strategy, stats in results['by_strategy'].items():
            print(f"\n{strategy.upper()} Strategy:")
            print(f"Success Rate: {stats['success_rate']:.2%}")
            print(f"Average Time: {stats['avg_time']:.2f}s")
            print(f"Total Attempts: {stats['total_attempts']}")
            
        print("\nTop Failure Patterns:")
        for error, count in sorted(results['failure_patterns'].items(), 
                                 key=lambda x: x[1], reverse=True)[:3]:
            print(f"- {error}: {count} occurrences")
            
        print("\nKey Recommendations:")
        for rec in results['recommendations']:
            print(f"- {rec}")
            
    except Exception as e:
        logging.error(f"Test run failed: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    main()