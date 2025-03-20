"""
Benchmark tests for comparing solver performances against known best scores.

This module provides comprehensive benchmarking tests for different solver
implementations, measuring both accuracy (compared to best-score.txt) and execution time.
"""

import sys
import unittest
import time
import re
import csv
from io import StringIO

sys.path.append("code/")
from grid import Grid
from solver import Solver, SolverGreedy, SolverFulkerson, SolverHungarian


class TestSolverBenchmark(unittest.TestCase):
    """
    Benchmark test class for solver implementations.
    """

    @classmethod
    def setUpClass(cls):
        """
        Load the best scores from the best-score.txt file once for all tests.
        """
        cls.best_scores = {}
        
        # Parse the best-score.txt file
        with open("tests/best-score.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        current_grid = None
        for line in lines:
            line = line.strip()
            if line == "---":
                continue
            elif line.endswith(".in"):
                current_grid = line
            elif current_grid and line.isdigit():
                cls.best_scores[current_grid] = int(line)
    
    def _benchmark_solver(self, solver_class, grid_name):
        """
        Benchmark a solver on a specific grid, measuring time and accuracy.
        
        Parameters:
        -----------
        solver_class: class
            The solver class to benchmark
        grid_name: str
            The name of the grid file to use
            
        Returns:
        --------
        dict
            Benchmark results including time, score, best score, and deviation
        """
        if grid_name not in self.best_scores:
            return {
                "solver": solver_class.__name__,
                "grid": grid_name,
                "error": "No best score found"
            }
            
        best_score = self.best_scores[grid_name]
        
        try:
            # Load the grid
            grid = Grid.grid_from_file(f"input/{grid_name}", read_values=True)
            grid.cell_init()
            
            # Run the solver with timing
            start_time = time.time()
            solver = solver_class(grid)
            solver.run()
            end_time = time.time()
            
            # Calculate results
            execution_time = end_time - start_time
            actual_score = solver.score()
            
            # Calculate deviation from best score
            if best_score == 0:
                # Avoid division by zero
                deviation_percentage = 0 if actual_score == 0 else float('inf')
            else:
                deviation_percentage = ((actual_score - best_score) / best_score) * 100
            
            # Determine if the result is optimal or better than optimal
            if actual_score <= best_score:
                quality = "Better than optimal" if actual_score < best_score else "Optimal"
            else:
                quality = "Suboptimal"
                
            return {
                "solver": solver_class.__name__,
                "grid": grid_name,
                "time": execution_time,
                "score": actual_score,
                "best_score": best_score,
                "deviation": deviation_percentage,
                "pairs": len(solver.pairs),
                "grid_size": f"{grid.n}x{grid.m}",
                "quality": quality
            }
            
        except Exception as e:
            return {
                "solver": solver_class.__name__,
                "grid": grid_name,
                "error": str(e)
            }
    
    def test_benchmark_small_grids(self):
        """
        Benchmark all solvers on small grid files and generate a performance report.
        """
        # Get small grids (grid0x.in)
        small_grids = [name for name in self.best_scores.keys() 
                      if re.match(r'grid0[0-5]\.in', name)]
        
        if not small_grids:
            self.skipTest("No small grids found in best-score.txt")
        
        # Define solvers to benchmark
        solvers = [SolverGreedy, SolverFulkerson, SolverHungarian]
        
        # Run benchmarks
        results = []
        for grid_name in small_grids:
            for solver_class in solvers:
                result = self._benchmark_solver(solver_class, grid_name)
                results.append(result)
                
                # Skip benchmarking remaining solvers if an error occurred
                if "error" in result:
                    print(f"Error benchmarking {solver_class.__name__} on {grid_name}: {result['error']}")
        
        # Generate performance report
        self._generate_report(results, "Small Grid Benchmark")
    
    def test_benchmark_medium_grids(self):
        """
        Benchmark all solvers on medium grid files and generate a performance report.
        """
        # Get medium grids (grid1x.in)
        medium_grids = [name for name in self.best_scores.keys() 
                       if re.match(r'grid1[0-9]\.in', name)]
        
        if not medium_grids:
            self.skipTest("No medium grids found in best-score.txt")
        
        # Define solvers to benchmark
        solvers = [SolverGreedy, SolverFulkerson, SolverHungarian]
        
        # Run benchmarks
        results = []
        for grid_name in medium_grids:
            for solver_class in solvers:
                result = self._benchmark_solver(solver_class, grid_name)
                results.append(result)
                
                # Skip benchmarking remaining solvers if an error occurred
                if "error" in result:
                    print(f"Error benchmarking {solver_class.__name__} on {grid_name}: {result['error']}")
        
        # Generate performance report
        self._generate_report(results, "Medium Grid Benchmark")
    
    def test_hungarian_large_grid_benchmark(self):
        """
        Benchmark the Hungarian solver (generally the best) on large grid files.
        """
        # Get large grids (grid2x.in)
        large_grids = [name for name in self.best_scores.keys() 
                      if re.match(r'grid2[0-9]\.in', name)]
        
        if not large_grids:
            self.skipTest("No large grids found in best-score.txt")
        
        # Only benchmark Hungarian on large grids as it tends to be the most efficient
        solver_class = SolverHungarian
        
        # Run benchmarks
        results = []
        for grid_name in large_grids:
            result = self._benchmark_solver(solver_class, grid_name)
            results.append(result)
            
            # Print progress
            if "error" in result:
                print(f"Error benchmarking {solver_class.__name__} on {grid_name}: {result['error']}")
            else:
                print(f"Completed {solver_class.__name__} on {grid_name}: " 
                      f"Score={result['score']}, Best={result['best_score']}, "
                      f"Time={result['time']:.3f}s")
        
        # Generate performance report
        self._generate_report(results, "Large Grid Benchmark (Hungarian Only)")
    
    def _generate_report(self, results, title):
        """
        Generate and print a performance report from benchmark results.
        
        Parameters:
        -----------
        results: list
            List of benchmark result dictionaries
        title: str
            Title for the report
        """
        if not results:
            print("No benchmark results to report")
            return
        
        # Filter out results with errors
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            print("All benchmark attempts resulted in errors")
            return
            
        # Create a CSV-like string for the report
        output = StringIO()
        fieldnames = ["solver", "grid", "grid_size", "time", "score", "best_score", 
                     "deviation", "pairs", "quality"]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in valid_results:
            # Format time nicely
            result["time"] = f"{result['time']:.3f}s"
            # Format deviation nicely
            result["deviation"] = f"{result['deviation']:.2f}%"
            writer.writerow(result)
        
        # Print report
        print("\n" + "=" * 80)
        print(f"{title} Results")
        print("=" * 80)
        print(output.getvalue())
        
        # Summarize best solver for each grid
        grids = set(r["grid"] for r in valid_results)
        print("\nBest Solver by Grid:")
        print("-" * 40)
        
        # Count optimal solutions by solver
        solvers = set(r["solver"] for r in valid_results)
        optimal_counts = {solver: 0 for solver in solvers}
        better_than_optimal_counts = {solver: 0 for solver in solvers}
        
        for grid_name in sorted(grids):
            grid_results = [r for r in valid_results if r["grid"] == grid_name]
            
            # Find solver with lowest score
            best_solver = min(grid_results, key=lambda x: float(x["score"]))
            
            # Find solver with fastest time
            fastest_solver = min(grid_results, 
                               key=lambda x: float(x["time"].replace("s", "")))
            
            print(f"Grid: {grid_name}")
            print(f"  Most Accurate: {best_solver['solver']} "
                  f"(Score: {best_solver['score']}, "
                  f"Deviation: {best_solver['deviation']}, "
                  f"Quality: {best_solver['quality']})")
            
            print(f"  Fastest: {fastest_solver['solver']} "
                  f"(Time: {fastest_solver['time']})")
            
            # Update optimal counts
            for result in grid_results:
                if result["quality"] == "Optimal":
                    optimal_counts[result["solver"]] += 1
                elif result["quality"] == "Better than optimal":
                    better_than_optimal_counts[result["solver"]] += 1
        
        # Print solver statistics
        print("\nSolver Statistics:")
        print("-" * 40)
        for solver in sorted(solvers):
            total_grids = len([r for r in valid_results if r["solver"] == solver])
            optimal_percent = (optimal_counts[solver] / total_grids) * 100
            better_percent = (better_than_optimal_counts[solver] / total_grids) * 100
            
            print(f"{solver}:")
            print(f"  Optimal solutions: {optimal_counts[solver]}/{total_grids} ({optimal_percent:.1f}%)")
            print(f"  Better than optimal: {better_than_optimal_counts[solver]}/{total_grids} ({better_percent:.1f}%)")
            print(f"  Combined (optimal + better): {optimal_counts[solver] + better_than_optimal_counts[solver]}/{total_grids} "
                  f"({optimal_percent + better_percent:.1f}%)")
                  
        print("\n")


if __name__ == "__main__":
    unittest.main()