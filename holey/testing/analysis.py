"""
Analysis utilities for test results.
Provides visualization and insight generation from test runs.
"""

from typing import Dict, List, Any
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import json

class ResultAnalyzer:
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.strategy_data = defaultdict(list)
        self._process_results()

    def _process_results(self):
        """Process raw results into analyzable data"""
        for strategy, attempts in self.results['by_strategy'].items():
            self.strategy_data[strategy] = {
                'success_rates': [],
                'times': [],
                'errors': defaultdict(int)
            }
            
            for attempt in attempts:
                self.strategy_data[strategy]['success_rates'].append(int(attempt['success']))
                self.strategy_data[strategy]['times'].append(attempt['time_taken'])
                if not attempt['success'] and attempt.get('error'):
                    self.strategy_data[strategy]['errors'][attempt['error']] += 1

    def analyze_strategy_performance(self) -> Dict[str, Any]:
        """Analyze detailed strategy performance"""
        analysis = {}
        
        for strategy, data in self.strategy_data.items():
            success_rate = np.mean(data['success_rates'])
            avg_time = np.mean(data['times'])
            
            # Calculate stability metrics
            time_std = np.std(data['times'])
            success_consistency = np.std(data['success_rates'])
            
            analysis[strategy] = {
                'success_rate': success_rate,
                'avg_time': avg_time,
                'time_stability': 1.0 / (1.0 + time_std),  # Higher is more stable
                'success_consistency': 1.0 - success_consistency,  # Higher is more consistent
                'common_errors': dict(sorted(
                    data['errors'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3])
            }
            
        return analysis

    def analyze_correlations(self) -> Dict[str, float]:
        """Analyze correlations between metrics"""
        correlations = {}
        
        for strategy, data in self.strategy_data.items():
            # Time vs success correlation
            if len(data['success_rates']) > 1 and len(data['times']) > 1:
                corr = np.corrcoef(data['success_rates'], data['times'])[0, 1]
                correlations[f'{strategy}_time_vs_success'] = corr
                
        return correlations

    def generate_plots(self, output_dir: str = 'analysis'):
        """Generate analysis plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        self._plot_success_rates(output_dir / 'success_rates.png')
        self._plot_timing_distribution(output_dir / 'timing_dist.png')
        self._plot_error_distribution(output_dir / 'error_dist.png')
        self._plot_strategy_comparison(output_dir / 'strategy_comparison.png')

    def _plot_success_rates(self, output_path: Path):
        """Plot success rates by strategy"""
        strategies = list(self.strategy_data.keys())
        success_rates = [np.mean(self.strategy_data[s]['success_rates']) 
                        for s in strategies]

        plt.figure(figsize=(10, 6))
        plt.bar(strategies, success_rates)
        plt.title('Success Rates by Strategy')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)
        
        for i, v in enumerate(success_rates):
            plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
            
        plt.savefig(output_path)
        plt.close()

    def _plot_timing_distribution(self, output_path: Path):
        """Plot timing distribution by strategy"""
        plt.figure(figsize=(12, 6))
        
        positions = []
        times = []
        labels = []
        
        for i, (strategy, data) in enumerate(self.strategy_data.items()):
            positions.append(i)
            times.append(data['times'])
            labels.append(strategy)
            
        plt.boxplot(times, positions=positions, labels=labels)
        plt.title('Solving Time Distribution by Strategy')
        plt.ylabel('Time (seconds)')
        plt.yscale('log')
        
        plt.savefig(output_path)
        plt.close()

    def _plot_error_distribution(self, output_path: Path):
        """Plot error distribution by strategy"""
        plt.figure(figsize=(15, 8))
        
        strategies = list(self.strategy_data.keys())
        error_types = set()
        for data in self.strategy_data.values():
            error_types.update(data['errors'].keys())
        error_types = list(error_types)
        
        data = []
        for strategy in strategies:
            strategy_errors = self.strategy_data[strategy]['errors']
            data.append([strategy_errors[error] for error in error_types])
            
        data = np.array(data)
        
        bottom = np.zeros(len(strategies))
        for i, error in enumerate(error_types):
            plt.bar(strategies, data[:, i], bottom=bottom, label=error)
            bottom += data[:, i]
            
        plt.title('Error Distribution by Strategy')
        plt.ylabel('Number of Errors')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(output_path)
        plt.close()

    def _plot_strategy_comparison(self, output_path: Path):
        """Plot multi-metric strategy comparison"""
        strategies = list(self.strategy_data.keys())
        metrics = ['Success Rate', 'Avg Time', 'Time Stability', 'Success Consistency']
        
        performance = self.analyze_strategy_performance()
        
        # Normalize metrics
        data = []
        for strategy in strategies:
            perf = performance[strategy]
            row = [
                perf['success_rate'],
                1.0 / (1.0 + perf['avg_time']),  # Inverse time (higher is better)
                perf['time_stability'],
                perf['success_consistency']
            ]
            data.append(row)
            
        data = np.array(data)
        
        # Normalize each metric to [0,1]
        data_normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        
        # Plot radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Close the polygon
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for i, strategy in enumerate(strategies):
            values = np.concatenate((data_normalized[i], [data_normalized[i][0]]))
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy)
            ax.fill(angles, values, alpha=0.25)
            
        ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics)
        ax.set_title('Strategy Comparison Across Metrics')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.savefig(output_path)
        plt.close()

    def generate_insights(self) -> List[str]:
        """Generate insights from the analysis"""
        insights = []
        performance = self.analyze_strategy_performance()
        correlations = self.analyze_correlations()
        
        # Strategy insights
        best_strategy = max(performance.items(), key=lambda x: x[1]['success_rate'])[0]
        fastest_strategy = min(performance.items(), key=lambda x: x[1]['avg_time'])[0]
        most_stable = max(performance.items(), key=lambda x: x[1]['time_stability'])[0]
        
        insights.append(f"Best performing strategy: {best_strategy} "
                       f"({performance[best_strategy]['success_rate']:.2%} success rate)")
        insights.append(f"Fastest strategy: {fastest_strategy} "
                       f"({performance[fastest_strategy]['avg_time']:.2f}s average)")
        insights.append(f"Most stable strategy: {most_stable} "
                       f"(stability score: {performance[most_stable]['time_stability']:.2f})")
        
        # Correlation insights
        for metric, corr in correlations.items():
            if abs(corr) > 0.7:
                insights.append(f"Strong correlation ({corr:.2f}) found in {metric}")
                
        # Error patterns
        for strategy, data in performance.items():
            if data['common_errors']:
                top_error, count = list(data['common_errors'].items())[0]
                insights.append(f"{strategy}: Most common error '{top_error}' "
                              f"occurred {count} times")
                
        return insights

    def save_analysis(self, output_dir: str = 'analysis'):
        """Save complete analysis to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate and save plots
        self.generate_plots(output_dir)
        
        # Save performance analysis
        performance = self.analyze_strategy_performance()
        with open(output_dir / 'performance.json', 'w') as f:
            json.dump(performance, f, indent=2)
            
        # Save correlations
        correlations = self.analyze_correlations()
        with open(output_dir / 'correlations.json', 'w') as f:
            json.dump(correlations, f, indent=2)
            
        # Save insights
        insights = self.generate_insights()
        with open(output_dir / 'insights.md', 'w') as f:
            f.write("# Analysis Insights\n\n")
            for insight in insights:
                f.write(f"- {insight}\n")

    def generate_report(self) -> str:
        """Generate a complete analysis report"""
        report = ["# Test Results Analysis Report\n"]
        
        # Overall statistics
        report.append("## Overall Statistics\n")
        total_tests = sum(len(data['success_rates']) 
                         for data in self.strategy_data.values())
        total_success = sum(sum(data['success_rates']) 
                          for data in self.strategy_data.values())
        
        report.append(f"Total tests run: {total_tests}")
        report.append(f"Overall success rate: {total_success/total_tests:.2%}\n")
        
        # Strategy performance
        report.append("## Strategy Performance\n")
        performance = self.analyze_strategy_performance()
        for strategy, stats in performance.items():
            report.append(f"### {strategy}")
            report.append(f"- Success rate: {stats['success_rate']:.2%}")
            report.append(f"- Average time: {stats['avg_time']:.2f}s")
            report.append(f"- Time stability: {stats['time_stability']:.2f}")
            report.append(f"- Success consistency: {stats['success_consistency']:.2f}")
            report.append("\nCommon errors:")
            for error, count in stats['common_errors'].items():
                report.append(f"- {error}: {count} occurrences")
            report.append("")
            
        # Correlations
        report.append("## Metric Correlations\n")
        correlations = self.analyze_correlations()
        for metric, corr in correlations.items():
            report.append(f"- {metric}: {corr:.2f}")
        report.append("")
        
        # Insights
        report.append("## Key Insights\n")
        insights = self.generate_insights()
        for insight in insights:
            report.append(f"- {insight}")
            
        return "\n".join(report)