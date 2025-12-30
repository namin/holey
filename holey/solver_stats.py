"""
Solver statistics tracking for SMT-LIB puzzle solving.

Tracks per-puzzle, per-solver outcomes when SMT-LIB is generated.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import defaultdict


@dataclass
class SolverResult:
    """Result from a single solver attempt on a puzzle."""
    puzzle_name: str
    solver: str  # 'z3' or 'cvc5'
    status: str  # 'sat', 'unsat', 'unknown', 'timeout', 'error'
    verified: Optional[bool] = None  # Did result pass check_result?
    time_ms: Optional[float] = None  # Optional timing
    model: Optional[Dict] = None  # The model if sat


class SolverStats:
    """Collects and reports solver statistics across puzzles."""

    def __init__(self, run_all_solvers: bool = False):
        self.results: List[SolverResult] = []
        # Track which puzzle used which solver for the final result
        self._puzzle_solvers: Dict[str, Dict[str, SolverResult]] = defaultdict(dict)
        # When True, run all solvers even after one returns sat/unsat
        self.run_all_solvers = run_all_solvers

    def add(self, puzzle_name: str, solver: str, status: str,
            verified: Optional[bool] = None, time_ms: Optional[float] = None,
            model: Optional[Dict] = None):
        """Record a solver result."""
        result = SolverResult(
            puzzle_name=puzzle_name,
            solver=solver,
            status=status,
            verified=verified,
            time_ms=time_ms,
            model=model
        )
        self.results.append(result)
        self._puzzle_solvers[puzzle_name][solver] = result

    def update_verified(self, puzzle_name: str, verified: bool, solver: Optional[str] = None):
        """Update verification status for results of a puzzle.

        If solver is specified, only update that solver's result.
        Otherwise, update all sat results for the puzzle.
        """
        for result in self.results:
            if result.puzzle_name == puzzle_name and result.status == 'sat':
                if solver is None or result.solver == solver:
                    result.verified = verified

    def get_sat_results(self, puzzle_name: str) -> List[SolverResult]:
        """Get all sat results for a puzzle (for verification)."""
        return [r for r in self.results
                if r.puzzle_name == puzzle_name and r.status == 'sat']

    def get_solvers(self) -> List[str]:
        """Get list of all solvers that have been used."""
        solvers = set()
        for result in self.results:
            solvers.add(result.solver)
        return sorted(solvers)

    def get_puzzles(self) -> List[str]:
        """Get list of all puzzles that generated SMT-LIB."""
        return list(self._puzzle_solvers.keys())

    def summary_table(self) -> str:
        """Generate a markdown table of solver results."""
        if not self.results:
            return "No SMT-LIB puzzles recorded.\n"

        solvers = self.get_solvers()
        puzzles = self.get_puzzles()

        if not puzzles:
            return "No SMT-LIB puzzles recorded.\n"

        # Build data rows first to calculate column widths
        headers = ["Puzzle"] + solvers + ["Verified"]
        data_rows = []

        for puzzle in puzzles:
            row_parts = [puzzle]
            puzzle_results = self._puzzle_solvers[puzzle]
            any_verified = False

            for solver in solvers:
                if solver in puzzle_results:
                    result = puzzle_results[solver]
                    status_symbol = self._status_symbol(result.status)
                    # Append verification mark to sat results
                    if result.status == 'sat' and result.verified is True:
                        status_symbol += "✓"
                        any_verified = True
                    elif result.status == 'sat' and result.verified is False:
                        status_symbol += "✗"
                    row_parts.append(status_symbol)
                else:
                    row_parts.append("-")

            # Add overall verified column (any solver verified)
            if any_verified:
                row_parts.append("✓")
            else:
                row_parts.append("-")

            data_rows.append(row_parts)

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in data_rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        # Build formatted table
        def format_row(cells):
            return "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells)) + " |"

        header_line = format_row(headers)
        separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
        row_lines = [format_row(row) for row in data_rows]

        # Build summary
        summary = self._summary_counts(solvers)

        table = "\n".join([header_line, separator] + row_lines)
        return f"## Solver Statistics (SMT-LIB generated)\n\n{table}\n\n{summary}"

    def _status_symbol(self, status: str) -> str:
        """Convert status to display symbol."""
        symbols = {
            'sat': 'sat',
            'unsat': 'unsat',
            'unknown': '?',
            'timeout': 'T/O',
            'error': 'ERR'
        }
        return symbols.get(status, status)

    def _summary_counts(self, solvers: List[str]) -> str:
        """Generate summary counts for each solver."""
        counts = {solver: {'sat': 0, 'unsat': 0, 'unknown': 0, 'total': 0}
                  for solver in solvers}

        for puzzle, puzzle_results in self._puzzle_solvers.items():
            for solver in solvers:
                if solver in puzzle_results:
                    result = puzzle_results[solver]
                    counts[solver]['total'] += 1
                    if result.status in counts[solver]:
                        counts[solver][result.status] += 1

        lines = []
        for solver in solvers:
            c = counts[solver]
            if c['total'] > 0:
                lines.append(f"- **{solver}**: {c['sat']}/{c['total']} sat, "
                           f"{c['unsat']}/{c['total']} unsat, "
                           f"{c['unknown']}/{c['total']} unknown")

        return "\n".join(lines) if lines else ""
